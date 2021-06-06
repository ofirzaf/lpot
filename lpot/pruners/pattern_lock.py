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

import numpy as np
from .pruner import pruner_registry, Pruner
from ..utils import logger

@pruner_registry
class PatternLockPruner(Pruner):
    def __init__(self, model, local_config, global_config):
        super(PatternLockPruner, self).__init__(model, local_config, global_config)
        self.init = False
    
    def on_epoch_begin(self, epoch):
        # This method needs to be applied from the beginning of training, 
        # else the initial sparsity pattern will be lost
        if not self.init and epoch > 0:
            raise RuntimeError("Missed first epoch pruning step")
        if not self.init:
            logger.debug("Initializing pattern lock masks")
            self.compute_mask()
            self.init = True

    def on_batch_begin(self, batch_id):
        for weight in self.weights:
            if weight in self.masks:
                new_weight = self.masks[weight] * \
                    np.array(self.model.get_weight(weight))
                self.model.update_weights(weight, new_weight)

    def on_epoch_end(self):
        pass

    def on_batch_end(self):
        for weight in self.weights:
            if weight in self.masks:
                new_weight = self.masks[weight] * \
                    np.array(self.model.get_weight(weight))
                self.model.update_weights(weight, new_weight)

    def compute_mask(self):
        """compute masks according to current sparsity pattern"""
        for weight in self.weights:
            tensor = np.array(self.model.get_weight(weight))
            if len(tensor.shape) in self.tensor_dims:
                self.masks[weight] = tensor != 0.
