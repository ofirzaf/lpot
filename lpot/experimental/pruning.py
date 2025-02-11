#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from ..conf.config import Conf
from ..pruners import PRUNERS
from ..utils import logger
from ..utils.utility import singleton
from ..utils.create_obj_from_config import create_dataloader, create_train_func, create_eval_func
from ..model import BaseModel, MODELS
from .common import Model
from ..adaptor import FRAMEWORKS

@singleton
class Pruning:
    """This is base class of pruning object.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and pruning objectives (performance, memory footprint etc.).
       Pruning class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        pruning objective and related dataloaders etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)
        self.cfg = self.conf.usr_cfg
        self.framework = self.cfg.model.framework.lower()
        self._model = None
        self._pruning_func = None
        self._train_dataloader = None
        self._eval_func = None
        self._eval_dataloader = None
        self.adaptor = None
        self.pruners = []

    def on_epoch_begin(self, epoch):
        """ called on the begining of epochs"""
        for pruner in self.pruners:
            pruner.on_epoch_begin(epoch)

    def on_batch_begin(self, batch_id):
        """ called on the begining of batches"""
        for pruner in self.pruners:
            pruner.on_batch_begin(batch_id)

    def on_batch_end(self):
        """ called on the end of batches"""
        for pruner in self.pruners:
            pruner.on_batch_end()

    def on_epoch_end(self):
        """ called on the end of epochs"""
        for pruner in self.pruners:
            pruner.on_epoch_end()
        stats, sparsity = self._model.report_sparsity()
        logger.info(stats)
        logger.info(sparsity)

    def __call__(self):
        """The main entry point of pruning.

           This interface currently only works on pytorch
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in training and evaluation phases
              and pruning tuning settings.

              For this usage, only model parameter is mandotory.

           b) Partial yaml configuration: User specifies dataloaders used in training
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter to create lpot dataloader before calling this
              function.

              After that, User specifies fp32 "model", training dataset "p_dataloader"
              and evaluation dataset "eval_dataloader".

              For this usage, model, p_dataloader and eval_dataloader parameters are mandotory.

           c) Partial yaml configuration: User specifies dataloaders used in training phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The trained and pruned model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the pruned model meets
              the accuracy criteria. If not, the Tuner starts a new training and tuning flow.

              For this usage, model, q_dataloader and eval_func parameters are mandotory.

        Returns:
            pruned model: best pruned model found, otherwise return None

        """
        framework_specific_info = {'device': self.cfg.device,
                                   'random_seed': self.cfg.tuning.random_seed,
                                   'workspace_path': self.cfg.tuning.workspace.path,
                                   'q_dataloader': None}

        if self.framework == 'tensorflow':
            framework_specific_info.update(
                {"inputs": self.cfg.model.inputs, "outputs": self.cfg.model.outputs})

        self.adaptor = FRAMEWORKS[self.framework](framework_specific_info)

        assert isinstance(self._model, BaseModel), 'need set lpot Model for pruning....'

        for name in self.cfg.pruning.approach:
            assert name == 'weight_compression', 'now we only support weight_compression'
            for pruner in self.cfg.pruning.approach.weight_compression.pruners:
                if pruner.prune_type == 'basic_magnitude':
                    self.pruners.append(PRUNERS['BasicMagnitude'](\
                                            self._model, \
                                            pruner,
                                            self.cfg.pruning.approach.weight_compression))
            # TODO, add gradient_sensativity

        if self._train_dataloader is None and self._pruning_func is None:
            train_dataloader_cfg = self.cfg.pruning.train.dataloader
            assert train_dataloader_cfg is not None, \
                   'dataloader field of train field of pruning section ' \
                   'in yaml file should be configured as train_dataloader property is NOT set!'

            self._train_dataloader = create_dataloader(self.framework, train_dataloader_cfg)

        if self._eval_dataloader is None and self._eval_func is None:
            eval_dataloader_cfg = self.cfg.evaluation.accuracy.dataloader
            assert eval_dataloader_cfg is not None, \
                   'dataloader field of evaluation ' \
                   'in yaml file should be configured as eval_dataloader property is NOT set!'

            self._eval_dataloader = create_dataloader(self.framework, eval_dataloader_cfg)

        if self._pruning_func is None:
            # train section of pruning section in yaml file should be configured.
            train_cfg = self.cfg.pruning.train
            assert train_cfg, "train field of pruning section in yaml file must " \
                              "be configured for pruning if pruning_func is NOT set."
            hooks = {
                'on_epoch_start': self.on_epoch_begin,
                'on_epoch_end': self.on_epoch_end,
                'on_batch_start': self.on_batch_begin,
                'on_batch_end': self.on_batch_end,
            }
            self._pruning_func = create_train_func(self.framework, \
                                                   self.train_dataloader, \
                                                   self.adaptor, train_cfg, hooks=hooks)
        self._pruning_func(self._model.model)
        logger.info('Model pruning is done. Start to evaluate the pruned model...')
        if self._eval_func is None:
            # eval section in yaml file should be configured.
            eval_cfg = self.cfg.evaluation
            assert eval_cfg, "eval field of pruning section in yaml file must " \
                              "be configured for pruning if eval_func is NOT set."
            self._eval_func = create_eval_func(self.framework, \
                                            self.eval_dataloader, \
                                            self.adaptor, \
                                            eval_cfg.accuracy.metric, \
                                            eval_cfg.accuracy.postprocess, \
                                            fp32_baseline = False)
        score = self._eval_func(self._model)
        logger.info('Pruned model score is: ' + str(score))
        return self._model

    @property
    def train_dataloader(self):
        """ Getter to train dataloader """
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, dataloader):
        """Set Data loader for training for pruning.
           It is iterable and the batched data should consists of a tuple like
           (input, label) if the training dataset containing label, or yield (input, _)
           for label-free train dataset, the input in the batched data will be used for
           model inference, so it should satisfy the input format of specific model.
           In train process, label of data loader will not be used and
           neither the postprocess and metric. User only need to set
           train_dataloader when train_dataloader can not be configured from yaml file.

           Args:
               dataloader(generator): user are supported to set a user defined dataloader
                                      which meet the requirements that can yield tuple of
                                      (input, label)/(input, _) batched data. Another good
                                      practice is to use lpot.experimental.common.DataLoader
                                      to initialize a lpot dataloader object. Notice
                                      lpot.experimental.common.DataLoader is just a wrapper of the
                                      information needed to build a dataloader, it can't yield
                                      batched data and only in this setter method
                                      a 'real' train_dataloader will be created,
                                      the reason is we have to know the framework info
                                      and only after the Pruning object created then
                                      framework infomation can be known.
                                      Future we will support creating iterable dataloader
                                      from lpot.experimental.common.DataLoader
        """
        from .common import _generate_common_dataloader
        self._train_dataloader = _generate_common_dataloader(
            dataloader, self.framework)

    @property
    def eval_dataloader(self):
        """ Getter to eval dataloader """
        return self._eval_dataloader

    @eval_dataloader.setter
    def eval_dataloader(self, dataloader):
        """Set Data loader for evaluation of pruned model.
           It is iterable and the batched data should consists of yield (input, _).
           the input in the batched data will be used for model inference, so it 
           should satisfy the input format of specific model.
           User only need to set eval_dataloader when eval_dataloader can not be 
           configured from yaml file.

           Args:
               dataloader(generator): user are supported to set a user defined dataloader
                                      which meet the requirements that can yield tuple of
                                      (input, label)/(input, _) batched data. Another good
                                      practice is to use lpot.experimental.common.DataLoader
                                      to initialize a lpot dataloader object. Notice
                                      lpot.experimental.common.DataLoader is just a wrapper of the
                                      information needed to build a dataloader, it can't yield
                                      batched data and only in this setter method
                                      a 'real' train_dataloader will be created,
                                      the reason is we have to know the framework info
                                      and only after the Pruning object created then
                                      framework infomation can be known.
                                      Future we will support creating iterable dataloader
                                      from lpot.experimental.common.DataLoader
        """
        from .common import _generate_common_dataloader
        self._eval_dataloader = _generate_common_dataloader(
            dataloader, self.framework)

    @property
    def model(self):
        """ Getter of model in lpot.model  """
        return self._model

    @model.setter
    def model(self, user_model):
        """Only support PyTorch model, it's torch.nn.model instance.

        Args:
           user_model: user are supported to set model from original PyTorch model format
                       Best practice is to set from a initialized lpot.experimental.common.Model.

        """
        if not isinstance(user_model, Model):
            logger.warning('force convert user raw model to lpot model, ' +
                'better initialize lpot.experimental.common.Model and set....')
            user_model = Model(user_model)
        framework_model_info = {}
        cfg = self.conf.usr_cfg
        if self.framework == 'tensorflow':
            framework_model_info.update(
                {'name': cfg.model.name,
                 'input_tensor_names': cfg.model.inputs,
                 'output_tensor_names': cfg.model.outputs,
                 'workspace_path': cfg.tuning.workspace.path})

        self._model = MODELS[self.framework](\
            user_model.root, framework_model_info, **user_model.kwargs)

    @property
    def pruning_func(self):
        """ not support get pruning_func """
        logger.warning('pruning_func not support getter....')

    @pruning_func.setter
    def pruning_func(self, user_pruning_func):
        """Training function for pruning.

        Args:
            user_pruning_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If pruning_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed.
        """
        self._pruning_func = user_pruning_func

    @property
    def eval_func(self):
        """ not support get eval_func """
        logger.warning('eval_func not support getter....')

    @eval_func.setter
    def eval_func(self, user_eval_func):
        """Eval function for pruning.

        Args:
            user_eval_func: This function takes "model" as input parameter
                         and executes entire training process with self
                         contained training hyper-parameters. If pruning_func set,
                         an evaluation process must be triggered and user should
                         set eval_dataloader with metric configured or directly eval_func
                         to make evaluation of the model executed.
        """
        self._eval_func = user_eval_func
