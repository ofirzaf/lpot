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

import copy
import os
from abc import abstractmethod
from lpot.utils.utility import LazyImport, compute_sparsity
from lpot.utils import logger
from lpot.conf.dotdict import deep_get, deep_set
from lpot.conf import config as cfg 
from lpot.model.base_model import BaseModel
from lpot.model.onnx_model import ONNXModel

torch = LazyImport('torch')
tf = LazyImport('tensorflow')
mx = LazyImport('mxnet')
onnx = LazyImport('onnx')
ort = LazyImport("onnxruntime")
yaml = LazyImport('yaml')
json = LazyImport('json')

tensor_to_node = lambda s: list(set([x.split(':')[0] for x in s]))

def get_model_type(model):
    """Get mode type

    Args:
        model (string or model object): model path or model object

    Returns:
        type (string): model type
    """
 
    from lpot.adaptor.tf_utils.util import is_saved_model_format, is_ckpt_format
    if isinstance(model, tf.Graph):
        return 'graph'
    elif isinstance(model, tf.compat.v1.GraphDef):
        return 'graph_def'
    elif isinstance(model, tf.keras.Model):
        return 'keras'
    elif isinstance(model, tf.estimator.Estimator):
        return 'estimator'
    elif isinstance(model, str):
        model = os.path.abspath(os.path.expanduser(model))
        if (model.endswith('.pb') and os.path.isfile(model)):
            if is_saved_model_format(os.path.dirname(model)):
                # Warning: TF compatibility issue to load saved model. TF 2.3 keras.load
                # can load saved model from TF backend, but TF 2.4 cannot.
                try:
                    model = tf.keras.models.load_model(model)
                    if isinstance(model, tf.keras.Model):
                        return 'keras'
                    else:
                        return 'saved_model'
                except:
                    # can't use keras load
                    return 'saved_model'
            else:
                return 'frozen_pb'
        elif model.endswith('.ckpt') and os.path.isfile(model):
            return 'slim'
        elif os.path.isdir(model):
            if is_ckpt_format(model):
                return 'checkpoint'
            elif is_saved_model_format(model):
                # it's very ugly tf version issue, in tf2.3 keras.load can
                # load saved model from tf backend, but tf2.4 it will crash
                try:
                    model = tf.keras.models.load_model(model)
                    if isinstance(model, tf.keras.Model):
                        return 'keras'
                    else:
                        return 'saved_model'
                except:
                    # can't use keras load
                    return 'saved_model'
        elif os.path.isfile(model + '.pb'):
            return 'frozen_pb'

    raise ValueError('model {} has not recognized model type....'.format(model))


def get_model_fwk_name(model):
    """Detect the input model belongs to which framework

    Args:
        model (string): framework name that supported by LPOT, if there's no available fwk info,
                        then return 'NA'.
    """
    def _is_onnxruntime(model):
        try:
            ort.InferenceSession(model)
        except:
            return False
        else:
            return True

    def _is_pytorch(model):
        try:
            is_torch = isinstance(model, torch.nn.Module)
        except:
            return False
        else:
            return is_torch

    def _is_tensorflow(model):
        try:
            get_model_type(model)
        except Exception as e:
            return False
        else:
            return True

    def _is_mxnet(model):
        try:
            is_mxnet = isinstance(model, mx.gluon.HybridBlock) or \
            (hasattr(model, '__len__') and len(model) > 1 and \
            isinstance(model[0], mx.symbol.Symbol))
        except:
            return False
        else:
            return is_mxnet

    checker = {'tensorflow': _is_tensorflow,
                'pytorch': _is_pytorch,
                'onnxruntime': _is_onnxruntime,
                'mxnet': _is_mxnet
                }

    for fwk_name, handler in checker.items():
        if handler(model):
            return fwk_name

    return 'NA'

def replace_graph_def_of_saved_model(input_model, output_model, graph_def):
    """update graph_def of saved model

    Args:
        input_model (string): the origin input model path
        output_model (string): output path
        graph_def (tf.compat.v1.GraphDef): processed graph_def
    """
 
    model_variables_dir = os.path.join(input_model, 'variables')
    os.makedirs(output_model, exist_ok=True)
    export_variables_dir = os.path.join(output_model, 'variables')
    export_saved_model = os.path.join(output_model,'saved_model.pb')

    checkpoint_file = os.path.join(export_variables_dir, 'checkpoint')
    os.makedirs(export_variables_dir, exist_ok=True)

    with open(checkpoint_file, 'w') as f:
        f.write("model_checkpoint_path: \"variables\"\n")
    from tensorflow.python.saved_model import loader_impl
    saved_model = loader_impl.parse_saved_model(input_model)
    meta_graph = saved_model.meta_graphs[0]
    # not all saved model have variables
    try:
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            loaded = tf.compat.v1.saved_model.loader.load(sess, ["serve"], input_model)
            # sess.run('init_all_tables')
            saver = tf.compat.v1.train.Saver()
            saver.save(sess,
                       os.path.join(export_variables_dir, 'variables'),
                       write_meta_graph=False,
                       write_state=False)
    except:
        logger.info('no variables in the saved model')
    meta_graph.graph_def.CopyFrom(graph_def)
    from tensorflow.python.lib.io import file_io
    file_io.write_string_to_file(export_saved_model, saved_model.SerializeToString())

def create_session_with_input_output(model, input_tensor_names, \
    output_tensor_names, **kwargs):
    """Create session

    Args:
        model (string or model object): model pah or model object
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
        
    """
 
    model_type = get_model_type(model)
    sess, input_tensor_names, output_tensor_names = SESSIONS[model_type](
        model, input_tensor_names, output_tensor_names, **kwargs)
    return sess, input_tensor_names, output_tensor_names

def validate_graph_node(graph_def, node_names):
    """Validate nodes exist in the graph_def

    Args:
        graph_def (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object
        node_names (list of string): node names to be validated
    """

    if len(node_names) == 0:
        return False
    all_node_name = [node.name for node in graph_def.node]
    for user_input_name in node_names:
        if user_input_name not in all_node_name:
            logger.info(str("Input node name {} doesn't exist in the model, " +
                "please check the yaml.").format(user_input_name))
            return False
    return True

def validate_and_inference_input_output(graph_def, \
    input_tensor_names, output_tensor_names):
    """validate and inference the input and output tensor names of graph_def

    Args:
        graph_def (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object
        input_tensor_names (list of string): input_tensor_names of graph_def
        output_tensor_names (list of string): output_tensor_names of graph_def

    Returns:
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """

    from lpot.adaptor.tf_utils.util import get_input_node_names
    input_tensor_names = input_tensor_names if validate_graph_node(\
        graph_def, tensor_to_node(input_tensor_names)) else \
        get_input_node_names(graph_def)

    from lpot.adaptor.tf_utils.util import get_output_node_names
    output_tensor_names = output_tensor_names if validate_graph_node(\
        graph_def, tensor_to_node(output_tensor_names)) else \
        get_output_node_names(graph_def)
    return input_tensor_names, output_tensor_names

def graph_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with tf.compat.v1.Graph

    Args:
        model (tf.compat.v1.Graph): tf.compat.v1.Graph object 
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=model, config=config)

    input_tensor_names, output_tensor_names = validate_and_inference_input_output(\
        model.as_graph_def(), input_tensor_names, output_tensor_names)

    return sess, input_tensor_names, output_tensor_names

def graph_def_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with tf.compat.v1.GraphDef

    Args:
        model (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object 
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    graph = tf.Graph()
    try:
        with graph.as_default():
            tf.import_graph_def(model, name='')
    except:
        input_tensor_names, output_tensor_names = validate_and_inference_input_output(\
            model, input_tensor_names, output_tensor_names)
        from lpot.adaptor.tf_utils.util import fix_ref_type_of_graph_def
        from lpot.adaptor.tf_utils.util import strip_unused_nodes
        model = fix_ref_type_of_graph_def(model)
        input_node_names = tensor_to_node(input_tensor_names)
        output_node_names = tensor_to_node(output_tensor_names)
        model = strip_unused_nodes(model, input_node_names, output_node_names)
        with graph.as_default():
            tf.import_graph_def(model, name='')

    return graph_session(graph, input_tensor_names, output_tensor_names, **kwargs)

def frozen_pb_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with frozen pb

    Args:
        model (string): model path 
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    graph_def = tf.compat.v1.GraphDef()
    model = model if model.endswith('.pb') else model + '.pb'
    with open(model, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def_session(graph_def, input_tensor_names, \
        output_tensor_names, **kwargs)

def keras_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with keras model

    Args:
        model (string or tf.keras.Model): model path or tf.keras.Model object
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    assert tf.version.VERSION > '2.1.0', 'keras model need tensorflow version > 2.1.0....'
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    if not isinstance(model, tf.keras.Model):
        model = tf.keras.models.load_model(model)
    kwargs = dict(zip(model.input_names, model.inputs))
    if tf.version.VERSION > '2.2.0':
        from tensorflow.python.keras.engine import keras_tensor
        if keras_tensor.keras_tensors_enabled():
            for name, tensor in kwargs.items():
                kwargs[name] = tensor.type_spec
    full_model = tf.function(lambda **kwargs: model(kwargs.values()))
    concrete_function = full_model.get_concrete_function(**kwargs)
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    from tensorflow.python.training import saver
    from tensorflow.core.protobuf import config_pb2
    from tensorflow.python.grappler import tf_optimizer
    from tensorflow.core.protobuf import meta_graph_pb2
    graph_def = frozen_model.graph.as_graph_def()
    input_names = [node.name for node in graph_def.node if node.op == 'Placeholder']
    output_names = [output.split(':')[0] for output in model.output_names]
    # replace the output name with squential
    for output_name in output_names:
        for node in graph_def.node[::-1]:
            if node.op == 'Identity' and output_name in node.input[0]:
                node.name = output_name
                break

    grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=graph_def, graph=frozen_model.graph)
    
    # Add a collection 'train_op' so that Grappler knows the outputs.
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in model.output_names:
      fetch_collection.node_list.value.append(array)
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
        fetch_collection)
    grappler_session_config = config_pb2.ConfigProto()
    rewrite_options = grappler_session_config.graph_options.rewrite_options
    rewrite_options.optimizers.append('constfold')
    rewrite_options.min_graph_nodes = -1
    graph_def = tf_optimizer.OptimizeGraph(grappler_session_config, \
                        grappler_meta_graph_def, graph_id=b"tf_graph")

    return graph_def_session(graph_def, input_names, output_names, **kwargs)

def slim_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with slim model

    Args:
        model (string): model path
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    assert tf.version.VERSION < '2.0.0', 'slim model only used in tensorflow 1.x'
    from .nets_factory import TFSlimNetsFactory
    factory = TFSlimNetsFactory()
    assert 'name' in kwargs, 'model name should be included in slim checkpoint....'
    assert kwargs['name'] in factory.default_slim_models, \
        'only support topology {}'.format(factory.default_slim_models)
    net = copy.deepcopy(factory.networks_map[kwargs['name']])
    model_func = net.pop('model')
    arg_scope = net.pop('arg_scope')()
    inputs_shape = net.pop('input_shape')
    kwargs = net
    images = tf.compat.v1.placeholder(name='input', dtype=tf.float32, \
        shape=inputs_shape)

    import tf_slim as slim
    with tf.compat.v1.Session() as sess:
        with slim.arg_scope(arg_scope) as scope:  # pylint: disable=not-context-manager
            model_func(images, is_training=False, **kwargs)
        graph_def = sess.graph.as_graph_def()
        output_tensor_names = output_tensor_names if len(output_tensor_names) > 0 \
            else [graph_def.node[-1].name]

        from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos
        graph_def = freeze_graph_with_def_protos(
            input_graph_def=graph_def,
            input_saver_def=None,
            input_checkpoint=model,
            output_node_names=','.join(output_tensor_names),
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='',
            clear_devices=True,
            initializer_nodes='')

    return graph_def_session(graph_def, ['input'], output_tensor_names)

def checkpoint_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with ckpt model

    Args:
        model (string): model path
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    assert output_tensor_names is not None and len(output_tensor_names) > 0, \
        'outputs should not be None of checkpoint....'

    ckpt_prefix = [os.path.splitext(i)[0] for i in os.listdir(model) \
        if i.endswith(".meta")][0]

    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    graph = tf.Graph()
    sess = tf.compat.v1.Session(graph=graph, config=config)
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(\
            os.path.join(model, ckpt_prefix + '.meta'), clear_devices=True)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, os.path.join(model, ckpt_prefix))

    from lpot.adaptor.tf_utils.util import get_input_node_names
    input_tensor_names = input_tensor_names if validate_graph_node(\
        sess.graph.as_graph_def(), tensor_to_node(input_tensor_names)) else \
        get_input_node_names(sess.graph.as_graph_def())

    return sess, input_tensor_names, output_tensor_names

def estimator_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with estimator model

    Args:
        model (tf.estimator.Estimator): tf.estimator.Estimator object
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model
        kwargs (dict): other required parameters, like input_fn

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    assert 'input_fn' in kwargs, 'input func should be supplied for estimator session....'
    with tf.Graph().as_default() as g:
      features, input_hooks = model._get_features_from_input_fn(
          kwargs['input_fn'], tf.estimator.ModeKeys.PREDICT)
      estimator_spec = model._call_model_fn(features, None,
          tf.estimator.ModeKeys.PREDICT, model.config)

      if len(output_tensor_names) == 0:
          outputs = [tensor.name for tensor in estimator_spec.predictions.values()] if\
              isinstance(estimator_spec.predictions, dict) else \
                  [estimator_spec.predictions.name]
      else:
          outputs = output_tensor_names

      logger.info('estimator output tensor names is {}'.format(outputs))
      with tf.compat.v1.Session(graph=g) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Freezing a graph requires output_node_names, which can be found in
        # estimator_spec.predictions that contains prediction tensors as a
        # dictionary
        # When a model uses Iterator, we need to have 'MakeIterator' (default
        # name used by TF) in the output_node_names as well.
        output_nodes = list(set([output.split(':')[0] for output in outputs]))
        if 'MakeIterator' in [node.op for node in g.as_graph_def().node]:
            output_nodes.append('MakeIterator')

        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
          g.as_graph_def(), output_nodes)

      return graph_def_session(graph_def, input_tensor_names, outputs)

def saved_model_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with saved model

    Args:
        model (string): model path
        input_tensor_names (list of string): input_tensor_names of model
        output_tensor_names (list of string): output_tensor_names of model

     Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
 
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)
    loader = tf.compat.v1.saved_model.loader.load(sess, ["serve"], model)
    input_tensor_names = [i.name for _, i in \
        loader.signature_def['serving_default'].inputs.items()]
    output_tensor_names = [i.name for _, i in \
        loader.signature_def['serving_default'].outputs.items()]

    return sess, input_tensor_names, output_tensor_names

# it's necessary that a session with input output tensors to run the model
SESSIONS = {'frozen_pb': frozen_pb_session,
            'graph_def': graph_def_session,
            'graph': graph_session,
            'saved_model': saved_model_session,
            'keras': keras_session,
            'checkpoint': checkpoint_session,
            'estimator': estimator_session,
            'slim': slim_session,}

class TensorflowModel(object):
    """Build LPOT TensorflowModel object

    Args:
        model (string or tensorflow model object): model path or model object
        framework_specific_info (dict): 
            information about model and framework, such as input_tensor_names, 
            input_tensor_names, workspace_path and name
        kwargs (dict): other required parameters, like input_fn

    Returns:
        TensorflowModel object
    """
 
    def __new__(cls, model, framework_specific_info={}, **kwargs):
        model_type = get_model_type(model)
        return TENSORFLOW_MODELS[model_type](model, framework_specific_info, **kwargs)

class TensorflowBaseModel(BaseModel):
    """Build TensorflowBaseModel object

    Args:
        model (string or tensorflow model object): model path or model object
        framework_specific_info (dict): 
            information about model and framework, such as input_tensor_names, 
            input_tensor_names, workspace_path and name
        kwargs (dict): other required parameters, like input_fn

    """
 
    def __init__(self, model, framework_specific_info={}, **kwargs):
        self.framework_specific_info = framework_specific_info
        input_tensor_names = deep_get(framework_specific_info, 'input_tensor_names', [])
        output_tensor_names = deep_get(framework_specific_info, 'output_tensor_names', [])
        self.workspace_path = deep_get(framework_specific_info, 'workspace_path', './')
        self.kwargs = copy.deepcopy(kwargs)
        kwargs.update({'name': deep_get(framework_specific_info, 'name')})
        self._model = model
        self.sess, self._input_tensor_names, self._output_tensor_names = \
            create_session_with_input_output(
                model, input_tensor_names, output_tensor_names, **kwargs)

        deep_set(framework_specific_info, 'input_tensor_names', self._input_tensor_names)
        deep_set(framework_specific_info, 'output_tensor_names', self._output_tensor_names)

        # sess_global_initialize(self.sess)
        self.iter_op = None
        if 'MakeIterator' in [node.op for node in self.sess.graph.as_graph_def().node]:
            self.iter_op = self.sess.graph.get_operation_by_name('MakeIterator')

        tf.compat.v1.get_variable_scope().reuse_variables()
        self._graph_info = {}

    @property
    def model(self):
        return self.sess.graph

    @property
    def graph_def(self):
        return self.sess.graph.as_graph_def()

    @property
    def graph_info(self):
        self._graph_info = {}
        for node in self.graph_def.node:
            self._graph_info[node.name] = node.op
        return self._graph_info

    @property
    def graph(self):
        return self.sess.graph

    @graph_def.setter
    def graph_def(self, graph_def):
        self.sess.close()
        self.sess, self._input_tensor_names, self._output_tensor_names = \
            create_session_with_input_output(graph_def, self._input_tensor_names, \
                self._output_tensor_names)
        if self.iter_op:
            self.iter_op = self.sess.graph.get_operation_by_name('MakeIterator')

    @property
    def input_tensor_names(self):
        return self._input_tensor_names

    @input_tensor_names.setter
    def input_tensor_names(self, tensor_names):
        assert validate_graph_node(\
            self.sess.graph.as_graph_def(), tensor_to_node(tensor_names)), \
                'tensor names should not be None or empty'
        self._input_tensor_names = tensor_names

    @property
    def output_tensor_names(self):
        return self._output_tensor_names

    @output_tensor_names.setter
    def output_tensor_names(self, tensor_names):
        assert validate_graph_node(\
            self.sess.graph.as_graph_def(), tensor_to_node(tensor_names)), \
                'tensor names should not be None or empty'
        self._output_tensor_names = tensor_names

    # input/output node names and input/output tensor
    # come from input/output tensor names, so do not support assign these values
    @property
    def input_node_names(self):
        return tensor_to_node(self._input_tensor_names)

    @property
    def output_node_names(self):
        output_node_names = tensor_to_node(self._output_tensor_names)
        if self.iter_op is not None:
            output_node_names.append('MakeIterator')
        return output_node_names

    @property
    def input_tensor(self):
        from lpot.adaptor.tf_utils.util import get_tensor_by_name
        return [get_tensor_by_name(\
            self.sess.graph, x) for x in self._input_tensor_names]

    @property
    def output_tensor(self):
        from lpot.adaptor.tf_utils.util import get_tensor_by_name
        return [get_tensor_by_name(\
            self.sess.graph, x) for x in self._output_tensor_names]

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace + '/save.pb'
        root = os.path.abspath(os.path.expanduser(root))
        # if not have suffix, default append .pb
        os.makedirs(os.path.dirname(root), exist_ok=True)
        pb_file = root if os.path.split(root)[-1].endswith('.pb') else root + '.pb'
        f = tf.io.gfile.GFile(pb_file, 'wb')
        f.write(self.graph_def.SerializeToString())
        logger.info("Save quantized model at %s" % pb_file)

class TensorflowSavedModelModel(TensorflowBaseModel):
 
    @property
    def graph_def(self):
        return self.sess.graph.as_graph_def() 

    @graph_def.setter
    def graph_def(self, graph_def):
        temp_saved_model_path = os.path.join(self.workspace_path, 'temp_saved_model')
        replace_graph_def_of_saved_model(self._model, temp_saved_model_path, graph_def)
        self._model = temp_saved_model_path
        self.sess.close()
        self.sess, self._input_tensor_names, self._output_tensor_names = \
            create_session_with_input_output(self._model, \
                self._input_tensor_names, self._output_tensor_names)

    @property
    def graph(self):
        return self.sess.graph

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        assert root != self._model, 'saved location should be different with original saved ' \
                                    'model path'
        import shutil
        file_names = os.listdir(self._model)
        for f in file_names:
            shutil.move(os.path.join(self._model, f), root)
        logger.info("Save quantized model at %s" % root)

class TensorflowKerasModel(TensorflowBaseModel):

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.exists(root):
            import shutil
            shutil.rmtree(root)

        os.makedirs(root, exist_ok=True)

        from tensorflow.python.saved_model import signature_constants
        from tensorflow.python.saved_model import tag_constants
        from lpot.adaptor.tf_utils.util import get_tensor_by_name
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(root)
        sigs = {}
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(self.sess.graph.as_graph_def(), name="")
            g = tf.compat.v1.get_default_graph()
            inp = [get_tensor_by_name(g, x) for x in self._input_tensor_names]
            out = [get_tensor_by_name(g, x) for x in self._output_tensor_names]
            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                {k: v for k, v in zip(self._input_tensor_names, inp)},
                {k: v for k, v in zip(self._output_tensor_names, out)})
            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map=sigs)
        builder.save()
        logger.info("Save quantized model at %s" % root)

class TensorflowCheckpointModel(TensorflowBaseModel):

    @property
    def graph_def(self):
        from lpot.adaptor.tf_utils.util import _parse_ckpt_bn_input
        from tensorflow.python.framework import graph_util
        graph_def = self.sess.graph.as_graph_def()
        graph_def = _parse_ckpt_bn_input(graph_def)
        return graph_util.convert_variables_to_constants(
            sess=self.sess,
            input_graph_def=graph_def,
            output_node_names=self.output_node_names)

    @graph_def.setter
    def graph_def(self, graph_def):
        self.sess.close()
        self.sess, self._input_tensor_names, self._output_tensor_names = \
            create_session_with_input_output(graph_def, self._input_tensor_names, \
                self._output_tensor_names)
        if self.iter_op:
            self.iter_op = self.sess.graph.get_operation_by_name('MakeIterator')

TENSORFLOW_MODELS = {'frozen_pb': TensorflowBaseModel,
                     'graph_def': TensorflowBaseModel,
                     'graph': TensorflowBaseModel,
                     'checkpoint': TensorflowCheckpointModel,
                     'estimator': TensorflowBaseModel,
                     'slim': TensorflowBaseModel,
                     'saved_model': TensorflowSavedModelModel,
                     'keras': TensorflowKerasModel,}


class PyTorchBaseModel(BaseModel):
    def __init__(self, model, framework_specific_info={}, **kwargs):
        self._model = model
        self.tune_cfg = None
        self.is_quantized = False

    def get_all_weight_names(self):
        """Get weight names

        Args:

        Returns:
            names (list): list of weight names

        """
        names = []
        for name, param in self._model.named_parameters():
            names.append(name)
        return names

    def get_weight(self, tensor_name):
        """Get weight value

        Args:
            tensor_name (string): weight name

        Returns:
            (object): weight tensor

        """
        state_dict = self._model.state_dict()
        for name in state_dict:
            if tensor_name == name:
                return state_dict[name]

    def update_weights(self, tensor_name, new_tensor):
        """Update weight value

        Args:
            tensor_name (string): weight name
            new_tensor (ndarray): weight value

        Returns:

        """
        new_tensor = torch.Tensor(new_tensor)
        state_dict = self._model.state_dict()
        for name in state_dict:
            if name == tensor_name:
                state_dict[name] = new_tensor
        self._model.load_state_dict(state_dict)

    def report_sparsity(self):
        """Get sparsity of the model

        Args:

        Returns:
            df (DataFrame): DataFrame of sparsity of each weight
            total_sparsity (float): total sparsity of model

        """
        import pandas as pd
        df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)', "Sparsity(%)",
                                   'Std', 'Mean', 'Abs-Mean'])
        pd.set_option('precision', 2)
        param_dims = [2, 4]
        params_size = 0
        sparse_params_size = 0
        for name, param in self._model.named_parameters():
            # Extract just the actual parameter's name, which in this context we treat
            # as its "type"
            if param.dim() in param_dims and any(type in name for type in ['weight', 'bias']):
                param_size, sparse_param_size, dense_param_size = compute_sparsity(
                    param.detach().numpy())
                density = dense_param_size / param_size
                params_size += param_size
                sparse_params_size += sparse_param_size
                df.loc[len(df.index)] = ([
                    name,
                    list(param.shape),
                    dense_param_size,
                    sparse_param_size,
                    (1 - density) * 100,
                    param.std().item(),
                    param.mean().item(),
                    param.abs().mean().item()
                ])

        total_sparsity = sparse_params_size / params_size * 100

        df.loc[len(df.index)] = ([
            'Total sparsity:',
            params_size,
            "-",
            int(sparse_params_size),
            total_sparsity,
            0, 0, 0])

        return df, total_sparsity


class PyTorchModel(PyTorchBaseModel):
    """Build PyTorchModel object

    Args:
        model (onnx model): model path 
        framework_specific_info (dict): information about model and framework
    """
 
    def __init__(self, model, framework_specific_info={}, **kwargs):
        super(PyTorchModel, self).__init__(model, framework_specific_info={}, **kwargs)

        from ..adaptor.pytorch import _cfg_to_qconfig, _propagate_qconfig
        if bool(framework_specific_info):
            workspace_path = framework_specific_info['workspace_path']
            tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(workspace_path)),
                                         'best_configure.yaml')
            weights_file = os.path.join(os.path.abspath(os.path.expanduser(workspace_path)),
                                        'best_model_weights.pt')
            assert os.path.exists(
                tune_cfg_file), "tune configure file %s didn't exist" % tune_cfg_file
            assert os.path.exists(
                weights_file), "weight file %s didn't exist" % weights_file
            self._model = copy.deepcopy(model.eval())
            with open(tune_cfg_file, 'r') as f:
                tune_cfg = yaml.safe_load(f)

            op_cfgs = _cfg_to_qconfig(tune_cfg)
            _propagate_qconfig(self._model, op_cfgs)
            # sanity check common API misusage
            if not any(hasattr(m, 'qconfig') and m.qconfig for m in self._model.modules()):
                logger.warn("None of the submodule got qconfig applied. Make sure you "
                            "passed correct configuration through `qconfig_dict` or "
                            "by assigning the `.qconfig` attribute directly on submodules")
            torch.quantization.add_observer_(self._model)
            torch.quantization.convert(self._model, inplace=True)
            weights = torch.load(weights_file)
            self._model.load_state_dict(weights)
        else:
            self._model = model
            self.tune_cfg= None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        try:
            with open(os.path.join(root, "best_configure.yaml"), 'w') as f:
                yaml.dump(self.tune_cfg, f, default_flow_style=False)
            torch.save(self._model.state_dict(), os.path.join(root, "best_model_weights.pt"))
            logger.info("Save config file and weights of quantized model at %s" % root)
        except IOError as e:
            logger.error("Unable to save configure file and weights. %s" % e)

    @property
    def graph_info(self):
        from ..adaptor.pytorch import get_ops_recursively
        op_map = {}
        get_ops_recursively(self._model, '', op_map)
        return op_map


class PyTorchIpexModel(PyTorchBaseModel):
    """Build PyTorchIpexModel object

    Args:
        model (onnx model): model path 
        framework_specific_info (dict): information about model and framework, like workspace_path
    """
 
    def __init__(self, model, framework_specific_info={}, **kwargs):
        super(PyTorchIpexModel, self).__init__(model, framework_specific_info={}, **kwargs)
        if bool(framework_specific_info):
            workspace_path = framework_specific_info['workspace_path']
            tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(workspace_path)),
                                         'best_configure.json')
            assert os.path.exists(
                tune_cfg_file), "tune configure file %s didn't exist" % tune_cfg_file

            with open(tune_cfg_file, 'r') as f:
                self.tune_cfg= json.load(f)

            self._model = model
        else:
            self._model = model
            self.tune_cfg= None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        try:
            with open(os.path.join(root, "best_configure.json"), 'w') as f:
                json.dump(self.tune_cfg, f)
            logger.info("Save config file of quantized model at %s" % root)
        except IOError as e:
            logger.error("Unable to save configure file and weights. %s" % e)

class MXNetModel(BaseModel):
    """Build MXNetModel object

    Args:
        model (mxnet model): model path 
        framework_specific_info (dict): information about model and framework
    """
 
    def __init__(self, model, framework_specific_info={}, **kwargs):
        self._model = model
        self.framework_specific_info = framework_specific_info

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def save(self, root=None):
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)

        if isinstance(self._model, mx.gluon.HybridBlock):
            self._model.export(root)
            logger.info('Save quantized hybrid block model at %s' % root)
        else:
            symbol, arg_params, aux_params = self._model
            symbol.save(root + '-symbol.json')
            save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
            save_dict.update(\
                {('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
            mx.nd.save(root + '-0000.params', save_dict)
            logger.info('Save quantized symbol model at %s' % root)

MODELS = {'tensorflow': TensorflowModel,
          'mxnet': MXNetModel,
          'pytorch': PyTorchModel,
          'pytorch_ipex': PyTorchIpexModel,
          'onnxrt_qlinearops': ONNXModel,
          'onnxrt_integerops': ONNXModel,}
