import json
import time
from collections import defaultdict
from functools import reduce
from operator import itemgetter, mul

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops

from greenness_track_toolkit.agent.core.colletor.collector_base import Collector
from greenness_track_toolkit.agent.models.flops import Flops
from greenness_track_toolkit.utils import get_logger


def copy_graph(graph: tf.Graph) -> tf.Graph:
    with tf.device('/CPU:0') and tf.Graph().as_default() as copied_graph:
        graph_def = graph.as_graph_def(add_shapes=True)
        meta_graph = tf.train.export_meta_graph(graph_def=graph_def,
                                                graph=graph,
                                                clear_devices=True,
                                                unbound_inputs_col_name=None)
        tf.train.import_meta_graph(meta_graph, clear_devices=True)
        return copied_graph


class TF1FlopsCollector(Collector):
    def close(self):
        pass

    def __init__(self, model, sess: tf.Session, batch_size: int, estimator=None):
        super().__init__()

        self._graph = copy_graph(sess.graph)
        get_logger().info("copy graph location:{}, origin location:{}".format(self._graph, sess.graph))
        self._batch_size = batch_size
        self._sess = sess
        self._model = model
        self._last_global_step = 0
        self.profiler = RuntimeProfiler(model, estimator, self._batch_size)
        self.batch_flops = self.profiler.profile(self._graph)
        get_logger().info("bs: {}, batch_flops: {}".format(batch_size, self.batch_flops))
        self._global_step_tensor = None

    def start(self):
        """
        开始收集
        :return:
        """
        with self._sess.graph.as_default():
            self._global_step_tensor = tf.train.get_global_step()  # native training loop
            if self._global_step_tensor is None:
                if self.profiler.mode == 'keras':
                    self._global_step_tensor = self._model.optimizer.iterations
        if self._global_step_tensor is None:
            get_logger().warning("The collector cannot calculate the flops, "
                                 "maybe global step is not set or the programming style is "
                                 "not supported")
        assert self._global_step_tensor is not None

    def delta(self, duration) -> Flops:
        current_flops = 0
        # there is not flops data output, if the session is closed and global_step
        # variable does not initialize
        if not self._sess._closed:
            current_global_step = self._sess.run(self._global_step_tensor)
            current_flops = self.batch_flops * (current_global_step - self._last_global_step)
            self._last_global_step = current_global_step
        return Flops(current_flops)


class FlopsDetail:
    def __init__(self, node=None, flops=None, input_shapes=None, ctx_type=None):
        self.node = node
        self.flops = flops
        self.input_shapes = input_shapes
        assert ctx_type in ('none', 'while', 'cond', None)
        self.ctx_type = ctx_type


class RuntimeProfiler():
    def __init__(self, model=None, estimator=None, bs=1, debug=False, repeat_counts=10):
        self.bs = bs
        self.debug = debug
        self.repeat_counts = repeat_counts
        self.mode = 'tf'
        if model is not None:
            if isinstance(model, keras.Model) or isinstance(model, keras.Sequential):
                self._model = model
                self.mode = 'keras'
            else:
                raise AttributeError("please input model with keras.Model or keras.Sequential")
        else:
            self._model = None

        if estimator is not None:
            self.mode = 'estimator'
            self._estimator = estimator

        self.tf_supported_unary_ops = {"ArgMax", "ArgMin", "L2Loss", "Log",
                                       "Mean", "Neg", "Pow",
                                       "Reciprocal", "Rsqrt", "Softmax", "Square",
                                       "Sum"}

        self.tf_supported_binary_ops = {"Add", "AvgPool", "BiasAdd", "Conv2D",
                                        "Conv2DBackpropInput", "DepthwiseConv2dNative", "Dilation2D", "Equal",
                                        "Greater",
                                        "GreaterEqual", "Less", "LessEqual", "MatMul",
                                        "Maximum", "Minimum", "Mul", "NotEqual",
                                        "RealDiv", "SquaredDifference", "Sub"}

        self.tf_supported_polytomy_ops = {"AddN"}

        self.extra_unary_ops = {"Abs", "Acos", "Acosh", "All", "Any", "Asin", "Asinh", "Atan", "Atanh", "AvgPool3D",
                                "BesselI0e", "BesselI1e", "Bincount", "Bucketize", "Cast", "Ceil", "Cholesky", "Cos",
                                "Cosh",
                                "Digamma", "Elu", "Erf", "Erfc", "Exp", "Expm1", "Floor", "FractionalAvgPool",
                                "FractionalMaxPool", "Inv", "IsFinite", "IsInf", "IsNan", "LeakyRelu", "Lgamma",
                                "LinSpace",
                                "Log1p", "LogMatrixDeterminant", "LogSoftmax", "LogicalNot", "Lu", "MatrixDeterminant",
                                "MatrixInverse", "MatrixSquareRoot", "Max", "MaxPool3D", "MaxPoolV2",
                                "MaxPoolWithArgmax",
                                "Min", "NthElement", "Prod", "Qr", "Relu", "Relu6", "Rint", "Round", "SelfAdjointEigV2",
                                "Selu", "Sigmoid", "Sign", "Sin", "Sinh", "Softplus", "Softsign", "Sqrt", "Svd", "Tan",
                                "Tanh", "TopKV2", "Unique"}
        self.extra_binary_ops = {"AddV2", "Angle", "ApproximateEqual", "Atan2", "BatchMatMul", "BiasAddV1", "Complex",
                                 "ComplexAbs", "Conj", "Conv3D", "Conv3DBackpropInputV2", "Cross", "Cumprod", "Cumsum",
                                 "Div",
                                 "DivNoNan", "DynamicStitch", "FloorDiv", "FloorMod", "Igamma", "Igammac", "Imag",
                                 "InTopK",
                                 "InTopKV2", "LogicalAnd", "LogicalOr", "MatrixSolve", "MatrixTriangularSolve", "Mod",
                                 "Polygamma", "Real", "SegmentMax", "SegmentMean", "SegmentMin", "SegmentProd",
                                 "SegmentSum",
                                 "SoftmaxCrossEntropyWithLogits", "SparseMatMul", "SparseSoftmaxCrossEntropyWithLogits",
                                 "TruncateDiv", "TruncateMod", "UnsortedSegmentMax", "UnsortedSegmentMin",
                                 "UnsortedSegmentProd", "UnsortedSegmentSum", "Xdivy", "Xlogy", "Zeta"}
        self.extra_polytomy_ops = {"AccumulateNV2", "Betainc", "SparseAdd", "SparseDenseCwiseAdd",
                                   "SparseDenseCwiseDiv",
                                   "SparseDenseCwiseMul", "SparseReduceMax", "SparseReduceMaxSparse", "SparseReduceSum",
                                   "SparseReduceSumSparse", "SparseSegmentMean", "SparseSegmentMeanWithNumSegments",
                                   "SparseSegmentSqrtN", "SparseSegmentSqrtNWithNumSegments", "SparseSegmentSum",
                                   "SparseSegmentSumWithNumSegments", "SparseSoftmax", "SparseSparseMaximum",
                                   "SparseSparseMinimum", "SparseTensorDenseAdd", "SparseTensorDenseMatMul"}

        self.unary_ops = self.tf_supported_unary_ops.union(self.extra_unary_ops)
        self.binary_ops = self.tf_supported_binary_ops.union(self.extra_binary_ops)
        self.polytomy_ops = self.tf_supported_polytomy_ops.union(self.extra_polytomy_ops)
        self.flops_ops = self.unary_ops.union(self.binary_ops).union(self.polytomy_ops)

        self.ctx_ops = {'Switch', 'Merge', 'Enter', 'Exit', 'NextIteration', 'LoopCond'}
        self.mem_ops = {'NoOp', 'StopGradient',
                        'StridedSlice', 'Const', 'Shape', 'Size', 'Pack', 'Identity', 'Merge', 'Assign', 'DiagPart',
                        'Reshape', 'ConcatV2', 'Stack', 'Fill', 'Tile', 'Pad', 'PadV2', 'Slice',
                        'Transpose', 'ExpandDims', 'Squeeze', 'Unpack', 'Where', 'GatherNd', 'ScatterNd',
                        'GatherV2', 'DynamicPartition', "DynamicStitch", "ParallelDynamicStitch",
                        'Range', 'ZerosLike', 'OnesLike',
                        'Select', 'SparseFillEmptyRows',
                        'TensorArrayReadV3', 'TensorArraySizeV3', 'TensorArrayV3',
                        'TensorArrayGatherV3', 'TensorArrayWriteV3', 'TensorArrayScatterV3',
                        'StringJoin', 'StringSplit', 'StringSplitV2', 'RegexReplace', 'Substr', 'StringStrip',
                        'DecodeBase64', 'ParseExample', 'ShardedFilename',
                        'RandomUniform', 'RandomStandardNormal', 'TruncatedNormal', 'RandomShuffle',
                        'ReadVariableOp', 'VarIsInitializedOp', 'IsVariableInitialized', 'VariableV2', 'VarHandleOp',
                        'HistogramSummary', 'ScalarSummary', 'Print',
                        'Placeholder', 'PlaceholderWithDefault',
                        'SaveV2', 'MergeV2Checkpoints', 'RestoreV2', 'AssignVariableOp',
                        'KvVariableExport',
                        'KvVariable', 'KvVariableIsInitializedV2', 'InitKvVariableV2', 'KvVariableImport',
                        'ReadKvVariableOpV2', 'KvVariableGatherOrInsertV2', 'KvVariableGatherOrZerosV2',
                        'KvVariableSizeV2', 'KvVariableFullOrDeltaImport', 'KvVariableFullOrDeltaExport'}
        self.dummy_ops = self.mem_ops.union(self.ctx_ops)

        self.unknown_ops = set()

        self.tensor_values = {}
        self.loop_count_values = {}

        self.flops_per_value = {"Expm1": 2, "Rsqrt": 2, "EuclideanNorm": 2, "Log1p": 2,
                                "Relu6": 2, "SquaredDifference": 2,
                                "Sigmoid": 4, "LogSoftmax": 4, "Elu": 4,
                                "Tanh": 5, "Selu": 5,
                                "GradientDescent": 1, "Momentum": 3, "Adagrad": 6, "Adadelta": 7, "Adam": 10,
                                "Ftrl": 9, "RMSProp": 7,
                                "GroupAdam": 15, "GroupFtrl": 14, "SparseGroupFtrl": 14, "GroupAMSGrad": 18,
                                "GroupAdadelta": 12, "GroupMomentum": 8}

    def profile(self, tf_graph, fe_path=None, mapping=True, fast=True):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count={'GPU': 0})
        with tf.device('/CPU:0'):
            self.sess = tf.Session(config=config, graph=tf_graph)
            with tf_graph.as_default(), self.sess.as_default():
                graph = self.sess.graph
                ts = time.time()
                te = time.time()

                ts = time.time()
                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.sess.run(tf.compat.v1.tables_initializer())

                feeds, output_tensor = self.get_tf_feeds_output(self.bs)
                te = time.time()
                output_names = [tensor.name for tensor in output_tensor]
                get_logger().info('\nfeature mock in {}s, bs={}'.format(round(te - ts, 1), self.bs))
                get_logger().info('\nfeeds.keys: {}'.format(list(feeds.keys())))
                if self.debug:
                    for k, v in feeds.items():
                        get_logger().info('feed after patch:', k, v.shape)
                get_logger().info('\noutput_names: {}'.format(output_names))
                ts = time.time()
                total_nodes = graph.get_operations()
                graph._unfetchable_ops.clear()
                forward_nodes = self.get_forward_nodes(total_nodes, output_names)
                serve_nodes = self.get_serve_nodes(feed_names=[t.name for t in feeds.keys()], output_names=output_names)
                kv_nodes = self.get_kv_nodes(forward_nodes)
                risk_nodes = self.get_risk_nodes(forward_nodes)
                io_nodes = self.get_io_nodes(forward_nodes)
                ph_nodes = self.get_ph_nodes(feeds)
                middle_nodes = serve_nodes.intersection(set(forward_nodes)) - kv_nodes - io_nodes - ph_nodes
                none_ctx_nodes = middle_nodes - risk_nodes
                none_ctx_flops_nodes = set([x for x in none_ctx_nodes if x.type in self.flops_ops])
                te = time.time()
                get_logger().info('parse graph in {}s'.format(round(te - ts, 1)))
                get_logger().info(
                    f'forward:{len(forward_nodes)} serve:{len(serve_nodes)} kv:{len(kv_nodes)} io:{len(io_nodes)} ph:{len(ph_nodes)} '
                    f'middle:{len(middle_nodes)} risk:{len(risk_nodes)} none_ctx_flops:{len(none_ctx_flops_nodes)}')
                get_logger().info('none-ctx node types: {}'.format(sorted(set([n.type for n in none_ctx_nodes]))))

                # 计算op的flops
                self.update_input_tensor_shapes(feeds)
                known_none_ctx_nodes = self.update_none_ctx_node_shapes(self.sess, none_ctx_flops_nodes, feeds)
                known_cond_ctx_nodes = self.update_cond_ctx_node_shapes(self.sess, middle_nodes, feeds)
                known_while_ctx_nodes = self.update_while_ctx_node_shapes(self.sess, middle_nodes, feeds)

                # flops_proto = self.flops_by_tf()
                # flop_details = self.parse_flops(flops_proto)
                # flop_details = self.patch_node_flops(flop_details, middle_nodes)
                flop_details = {}
                flop_details = self.patch_node_flops(flop_details, middle_nodes)

                self.weight_flops(known_none_ctx_nodes, flop_details, ctx_type='none')
                self.weight_flops(known_cond_ctx_nodes, flop_details, ctx_type='cond')
                self.weight_flops(known_while_ctx_nodes, flop_details, ctx_type='while')

                main_flops_details = [x for x in flop_details.values() if x.ctx_type is not None]
                for flops_detail in sorted(main_flops_details, key=lambda x: x.node.name):
                    node = flops_detail.node
                    get_logger().info(
                        f'op:{node.type} FLOPs:{flops_detail.flops} shape:{flops_detail.input_shapes} name:{node.name}')

                forward_flops = sum([x.flops for x in main_flops_details])

                backward_flops = 2*forward_flops
                update_flops = self.count_update_flops(output_names)

                sorted_flops_details = sorted(main_flops_details, key=lambda x: x.flops, reverse=True)
                get_logger().info('Nodes with Top-100 FLOPs')
                for i, detail in enumerate(sorted_flops_details[:100]):
                    node = detail.node
                    get_logger().info(
                        f'{i} op:{node.type} FLOPs:{detail.flops} shape:{detail.input_shapes} name:{node.name}')
                get_logger().info('forward flops:{}M, backward_flops:{}M, update_flops:{}M'.format(
                    round(forward_flops / 1000000, 2),
                    round(backward_flops / 1000000, 2),
                    round(update_flops / 1000000, 2))
                )
                total_flops = forward_flops + backward_flops + update_flops
                self.sess.close()
                return total_flops

    def get_tf_feeds_output(self, bs):
        def get_shape(tensor, bs=1):
            from tensorflow.python.framework.tensor_shape import TensorShape
            shape = tensor.get_shape()
            if shape.ndims is None:
                return [bs, 1]
            if len(shape) > 0 and shape[0].value is None:
                shape = TensorShape([bs] + shape[1:].dims)
            return shape.as_list()

        def get_default_input_output(bs):
            inputs = {}
            all_op = []
            op_inputs = set()
            nodes = tf.get_default_graph().get_operations()
            for op in nodes:
                if (op.type == 'Placeholder'):
                    key = op.outputs[0]
                    shape = get_shape(key, bs)
                    value = np.zeros(shape, dtype=np.float32)
                    inputs[key] = value
                op_inputs.update(set(op.inputs._inputs))
                if (len(op.outputs) > 0):
                    all_op.append(op.outputs[0])
            outputs = set(all_op) - set(op_inputs)
            return (inputs, outputs)

        inputs = {}
        outputs = []
        get_logger().info("self mode: {}".format(self.mode))
        if (self.mode == 'estimator'):
            nodes = self.sess.graph.get_operations()
            dense_tensors = list(
                filter(lambda x: x.dtype != tf.variant,
                       [item for item in nodes if item.type == 'IteratorGetNext'][0].outputs))
            for item in dense_tensors:
                shape = get_shape(item, bs)
                value = np.zeros(shape, dtype=np.float32)
                inputs[item] = value

            sparse_ops = [item for item in nodes if item.type == 'DeserializeSparse']
            for item in sparse_ops:
                indice_tensor = item.outputs[0]
                value_tensor = item.outputs[1]
                shape_tensor = item.outputs[2]

                sparse_shape = indice_tensor.shape[0]
                shape_value = 2 * np.ones(sparse_shape, dtype=np.int64)
                shape_value[0] = bs

                x_rand = np.random.rand(*shape_value)
                x_input = np.where(x_rand > 0.5, 1, 0)

                idx = tf.where(tf.not_equal(x_input, 0))
                sparse = tf.SparseTensor(idx, tf.gather_nd(x_input, idx), x_input.shape)

                inputs[indice_tensor] = sparse.indices
                inputs[value_tensor] = sparse.values
                inputs[shape_tensor] = sparse.dense_shape

            outputs = [output for x in ops.get_collection_ref(ops.GraphKeys.TRAIN_OP) for output in x.outputs]

        if (self.mode == 'tf'):
            inputs, outputs = get_default_input_output(bs)

        if (self.mode == 'keras'):
            if len(self._model.inputs) == 0:
                inputs, temp_outputs = get_default_input_output(bs)
                last_layer_name = self._model.layers[-1].name
                outputs = []
                for tensor in temp_outputs:
                    if (last_layer_name in tensor.name):
                        outputs.append(tensor)
            else:
                for input in self._model.inputs:
                    key = input
                    key = self.sess.graph.get_tensor_by_name(key.name)
                    shape = get_shape(key, bs)
                    value = np.zeros(shape, dtype=np.float32)
                    inputs[key] = value
                outputs = self._model.outputs

        if (len(inputs) == 0 or len(outputs) == 0):

            get_logger().info("inputs len {}, output len {}, use_default logic".format(len(inputs), len(outputs)))
            temp_inputs, temp_outputs = get_default_input_output(bs)
            if (len(inputs) == 0):
                inputs = temp_inputs
            if (len(outputs) == 0):
                outputs = temp_outputs
        return (inputs, outputs)

    def get_latency(self, sess, output_names, feeds):
        output_arrays = self.sess.run(output_names, feed_dict=feeds)
        if self.debug:
            for i, name in enumerate(output_names):
                get_logger().info(f'name:{name} shape:{output_arrays[i].shape}')
        ts = time.time()
        for i in range(self.repeat_counts):
            self.sess.run(output_names, feed_dict=feeds)
        te = time.time()
        latency = (te - ts) / self.repeat_counts * 1000
        return latency

    def load_model(self, model_path, sess, serving=True, fast=True):
        if serving:
            if fast:
                from tensorflow.python.saved_model.loader_impl import SavedModelLoader
                loader = SavedModelLoader(model_path)
                loader.load_graph(sess.graph, [tf.compat.v1.saved_model.tag_constants.SERVING])
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
            else:
                tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], model_path)
        else:
            latest = tf.train.latest_checkpoint('./' + model_path + '/')
            get_logger().info('tf.train.latest_checkpoint', latest)
            if latest is None:
                raise ValueError('latest is None')
            tf.train.import_meta_graph(latest + '.meta', clear_devices=True)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

    def weight_flops(self, nodes, flop_details, ctx_type=None):
        for node in nodes:
            if node.name not in flop_details:
                continue
            details = flop_details[node.name]
            details.ctx_type = ctx_type
            node_type_weight = self.flops_per_value.get(node.type, 1)
            details.flops *= node_type_weight
            if ctx_type == 'while':
                cfc = node._control_flow_context
                loop_count = self.loop_count_values.get(cfc, 1)
                details.flops *= loop_count

    def get_ph_nodes(self, feeds):
        nodes = set()
        graph = self.sess.graph
        for key in feeds.keys():
            nodes.add(graph.get_tensor_by_name(key.name).op)
        return nodes

    def update_none_ctx_node_shapes(self, sess, key_fetch_nodes, feeds):
        get_logger().info('start counting flops for trival nodes...')
        key_fetch_nodes = sorted(list(key_fetch_nodes), key=lambda x: x.name)

        ts = time.time()
        try:
            self.run_and_assign(sess, key_fetch_nodes, feeds)
        except Exception as e:
            get_logger().warning(f'try to batch fetch trivals with error, turn to partial run. error:{e}')
            ts = time.time()
            self.partial_run_and_assign(sess, key_fetch_nodes, feeds)
            te = time.time()
            get_logger().info('partial fetch trivals nodes in {}s'.format(round(te - ts, 1)))
        te = time.time()
        get_logger().info('finish counting flops for trival nodes...')
        get_logger().info('batch fetch trivals nodes in {}s'.format(round(te - ts, 1)))

        return key_fetch_nodes

    def update_cond_ctx_node_shapes(self, sess, nodes, feeds):
        get_logger().info('start counting flops for cond nodes...')
        ts = time.time()
        cond_fetchable_nodes, cond_unfetchable_nodes = self.fetch_cond_ctx_nodes(sess, nodes, feeds)

        cond_fetchable_nodes = sorted(list(cond_fetchable_nodes), key=lambda x: x.name)
        try:
            self.run_and_assign(sess, cond_fetchable_nodes, feeds)
        except Exception as e:
            get_logger().info(f'try to batch fetch cond with error, turn to partial run. error:{e}')
            ts = time.time()
            for i, t in enumerate(cond_fetchable_nodes):
                get_logger().info(f'cond digest:, {i}, {t.name}')
            cond_fetchable_nodes = self.keep_key_input_tensors(nodes, cond_fetchable_nodes)
            cond_fetchable_nodes = sorted(list(cond_fetchable_nodes), key=lambda x: x.name)
            cond_fetchable_nodes = self.partial_run_and_assign(sess, cond_fetchable_nodes, feeds)
            te = time.time()
            get_logger().info('partial fetch cond nodes in {}s'.format(round(te - ts, 1)))
        te = time.time()
        get_logger().info('finish counting flops for cond nodes...')
        get_logger().info('batch fetch cond nodes in {}s'.format(round(te - ts, 1)))
        return cond_fetchable_nodes

    def update_while_ctx_node_shapes(self, sess, nodes, feeds):
        get_logger().info('start counting flops for while nodes...')
        ts = time.time()
        while_ctx_nodes = self.fetch_while_ctx_nodes(sess, nodes, feeds)

        while_ctx_nodes = sorted(list(while_ctx_nodes), key=lambda x: x.name)
        key_tensors = set()
        for node in while_ctx_nodes:
            if node.type not in self.flops_ops:
                continue

            cfc = node._control_flow_context
            if cfc is None:
                continue

            if not isinstance(cfc, control_flow_ops.WhileContext):
                continue
            key_tensors.update(node.inputs._inputs)
            key_tensors.update(node.outputs)

        shapes = []
        for tensor in key_tensors:
            node = tensor.op
            cfc = node._control_flow_context
            cfc.Enter()
            ctx_shape = control_flow_ops.exit(tf.shape(tensor))
            cfc.Exit()
            shape = tf.identity(ctx_shape)
            shapes.append((tensor, shape))

        # TODO: recurse
        shape_values = sess.run([x[1] for x in shapes], feed_dict=feeds)
        for i, shape_value in enumerate(shape_values):
            tensor = shapes[i][0]
            tensor.set_shape(shape_value)
        te = time.time()
        get_logger().info('finish counting flops for while nodes...')
        get_logger().info('batch fetch while nodes in {}s'.format(round(te - ts, 1)))
        return while_ctx_nodes

    def patch_node_flops(self, flops_details, nodes):
        for node in nodes:
            if node.name in flops_details:
                continue
            if node.type in self.dummy_ops:
                continue
            if node.type not in self.flops_ops:
                self.unknown_ops.add(node.type)
                continue

            if node.type in self.unary_ops:
                flops = self.get_unary_flops(node)
            elif node.type in self.binary_ops:
                flops = self.get_binary_flops(node)
            elif node.type in self.polytomy_ops:
                flops = self.get_polytomy_flops(node)
            else:
                flops = 0
            input_shapes = [x.get_shape().as_list() for x in node.inputs._inputs if x.get_shape().ndims is not None]
            flops_detail = FlopsDetail(node=node, flops=flops, input_shapes=input_shapes)
            flops_details[node.name] = flops_detail
        return flops_details

    def count_update_flops(self, output_names):
        optimizer = output_names[0].split(':')[0].split('/')[-1]
        complexity = self.flops_per_value.get(optimizer, 1)
        elements = self.get_trainable_element_count()
        return elements * complexity

    def keep_key_input_tensors(self, nodes, tensors):
        key_tensors = set()
        for node in nodes:
            if node.type not in self.flops_ops:
                continue
            for i in node.inputs._inputs:
                if i in tensors:
                    key_tensors.add(i)
        return key_tensors

    def get_trainable_element_count(self):
        sizes = 0
        trainables = tf.trainable_variables()
        for v in trainables:
            if not isinstance(v, tf.Variable):
                continue
            shape = v.get_shape().as_list()
            if len(shape) > 0:
                size = reduce(mul, shape)
                sizes += size
        return sizes

    def get_forward_nodes(self, nodes, output_names):
        forwards = []

        for node in nodes:
            if 'gradients' in node.name:
                continue
            forwards.append(node)
        return forwards

    def get_feeds(self, model_path, serving, nodes=None, mapping=True, sess=None, defaults=None, fe_path=None):

        if serving:
            feeds = RuntimeProfiler.gen_feats(model_path + '/tf_signature.txt', bs=self.bs,
                                              serving=True, defaults=defaults)
            if feeds is None:
                return {}
            if fe_path is not None:
                self.patch_feeds_with_fe(feeds, defaults=defaults,
                                         fe_path=fe_path, sig_path=model_path + '/tf_signature.txt')
            else:
                self.patch_feeds_with_inference(sess, feeds, defaults=defaults, bs=self.bs)
        else:
            feeds = RuntimeProfiler.gen_feats(model_path + '/alps.meta', bs=self.bs,
                                              serving=False, defaults=defaults)
            if feeds is None:
                return {}
            if mapping:
                maps = self.get_train_ph_names('./' + model_path + '/alps.meta', nodes)
                map_feeds = {}
                for name, value in feeds.items():
                    map_feeds[maps[name]] = value
                feeds = map_feeds

        return feeds

    def patch_feeds_with_fe(self, feeds, defaults=None, fe_path=None, sig_path=None):

        sigs = json.loads(open(sig_path, 'r').read())
        inputs = sigs['inputs']
        name_to_tag = {}
        for input in inputs:
            name_to_tag[input['name']] = input['tag']

        for k, v in list(feeds.items()):
            if defaults is not None and k in defaults:
                continue
            if not isinstance(v, np.ndarray):
                continue
            dtype = v.dtype.type
            if dtype != np.str_:
                continue
            tag_name = name_to_tag[k]
            fef = f'{fe_path}/{tag_name}.conf'
            with open(fef, 'r') as f:
                line = f.readline()
                if 'class=BuildSeqFeatures' in line:
                    default_value = line.split(';')[-1].split('|')[1]
                else:
                    default_value = '0'

            c = reduce(mul, v.shape)
            values = np.reshape(np.array([default_value] * c), v.shape)
            feeds[k] = values

    def patch_feeds_with_inference(self, sess, feeds, defaults=None, bs=30):
        get_logger().info('start patch feeds')
        childrens = self.get_consumers(sess.graph.get_operations())

        for k, v in list(feeds.items()):
            if defaults is not None and k in defaults:
                continue
            if not isinstance(v, np.ndarray):
                continue
            dtype = v.dtype.type
            if dtype != np.str_:
                continue

            node = sess.graph.get_operation_by_name(k.split(':')[0])

            # graph_hit = False
            # default_graph_feats = {"ExtractFlatPathwiseGraphFeatureV1": "CgcSACoDCgEA"}
            # for graph_node_type in default_graph_feats.keys():
            #     graph_matched = self._seek_child_node_by_type(node, graph_node_type, childrens, max_deep=10)
            #     if len(graph_matched) > 0:
            #         feeds[k] = np.array([[default_graph_feats[graph_node_type]]] * bs)
            #         graph_hit = True
            #         break
            # if graph_hit:
            #     continue

            # log node types for debug
            unwalked = {node}
            walked = set()
            iter_idx = 0
            while True:
                if iter_idx > 8:
                    break
                iter_idx += 1
                for m in set(unwalked):
                    unwalked.remove(m)
                    if m in walked:
                        continue
                    walked.add(m)
                    unwalked.update(childrens[m])
            types = set([x.type for x in walked])
            get_logger().info('unknown patterns', types)

    def _seek_child_node_by_type(self, root, node_type, childrens, max_deep=5, dtypes=None):
        matched = set()
        unwalked = {root}
        for i in range(max_deep):
            for sub in set(unwalked):

                unwalked.remove(sub)
                if len(sub.outputs) == 0:
                    continue
                if sub.type == 'StringSplitV2':
                    output = sub.outputs[1]
                else:
                    output = sub.outputs[0]
                if dtypes is not None and output.dtype not in dtypes:
                    get_logger().info('seek', dtypes, output.dtype)
                    continue
                unwalked.update(childrens[sub])
                if isinstance(node_type, str):
                    if sub.type == node_type:
                        matched.add(sub)
                else:
                    assert len(node_type) == 2

                    if sub.type != node_type[0]:
                        continue
                    children = childrens[sub]
                    if len(children) != 1:
                        continue
                    child = list(children)[0]
                    if child.type != node_type[1]:
                        continue
                    matched.add(sub)
        return matched

    def update_input_tensor_shapes(self, feeds):
        graph = self.sess.graph
        for k, v in feeds.items():
            graph.get_tensor_by_name(k.name).set_shape(v.shape)

    def get_output_names(self, model_path, serving):
        if serving:
            output_names = RuntimeProfiler.get_outputs(model_path + '/tf_signature.txt')
        else:
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            sess_name = set()
            for x in train_op:
                for o in x.outputs:
                    sess_name.add(o.name)
            output_names = sorted(list(sess_name))
        return output_names

    def partial_run_and_assign(self, sess, nodes, feeds):

        fetches = [x for x in self.get_nodes_inputs(nodes) if x.name not in feeds]
        fetch_set = set(fetches)
        keys = list(feeds.keys())
        ts = time.time()
        h = self.sess.partial_run_setup(fetches, keys)
        te = time.time()
        get_logger().info('partial setup in {}s'.format(round(te - ts, 3)))

        unwalked = set(nodes)
        retry = 0
        fed = False
        idx = 0
        while True:
            walked = set()
            for node in sorted(list(unwalked), key=lambda x: x.name):
                try:
                    walked.add(node)
                    idx += 1
                    fetch = list(set(node.inputs._inputs) - fetch_set)
                    if len(fetch):
                        continue
                    if fed:
                        values = sess.partial_run(h, fetch)
                    else:
                        values = sess.partial_run(h, fetch, feeds)
                        fed = True

                    for i, tensor in enumerate(fetch):
                        value = values[i]
                        shape = value.shape if isinstance(value, np.ndarray) else []
                        tensor.set_shape(shape)

                except Exception as e:
                    retry += 1
                    get_logger().warning(f'exception with partial_run for node {node.type}:{node.name}, error:{e}')
                    break
                if idx % 10 == 0:
                    get_logger().info('partial_run {}/{}'.format(idx, len(nodes)))
            unwalked = unwalked - walked
            if len(unwalked) == 0:
                break

            fetches = [x for x in self.get_nodes_inputs(unwalked) if x.name not in feeds]
            fetch_set = set(fetches)
            h = self.sess.partial_run_setup(fetches, keys)
            fed = False

    def get_nodes_inputs(self, nodes):
        o = set()
        for node in nodes:
            o.update(node.inputs._inputs)
        return o

    def get_nodes_outputs(self, nodes):
        o = set()
        for node in nodes:
            o.update(node.outputs)
        return o

    def set_shape(self, tensor, value):
        if isinstance(value, np.ndarray):
            tensor.set_shape(value.shape)
        else:
            tensor.set_shape([])

    def run_and_assign(self, sess, nodes, feeds, cfc=False):
        tensors = list(self.get_nodes_inputs(nodes).union(self.get_nodes_outputs(nodes)))
        tensors = [x for x in tensors if
                   x.get_shape().ndims is None or any([d is None for d in x.get_shape().as_list()])]
        with tf.variable_scope('flops_shape'):
            if not cfc:
                fetches = [tf.shape(x) for x in tensors]
            else:
                fetches = tensors
        values = self.sess.run(fetches, feed_dict=feeds)
        for i, tensor in enumerate(tensors):
            if not cfc:
                tensor.set_shape(values[i])
            else:
                tensor.set_shape(values[i].shape)

    def sign(self, vs):
        return ','.join(sorted(list(vs)))

    def get_serve_nodes(self, feed_names=None, output_names=None):
        if not output_names:
            return set()

        parent_parents = self.get_producers(output_names, to_names=feed_names, include=True)
        return parent_parents

    def get_kv_nodes(self, nodes):
        kv_nodes = set()
        for node in nodes:
            if 'KvVariable' in node.type:
                kv_nodes.add(node)
                continue
            if any([x.op.type == 'KvVariable' or 'KvVariable' in x.name for x in node.inputs._inputs]):
                kv_nodes.add(node)
        return kv_nodes

    def get_io_nodes(self, nodes):
        io_nodes = set()
        trival_types = {'MakeIterator', 'MergeV2Checkpoints', 'RestoreV2', 'ReadVariableOp',
                        'SaveV2', 'SaveV3', 'VarHandleOp', 'VarIsInitializedOp', 'IteratorToStringHandle'}
        for node in nodes:
            if 'Dataset' in node.type:
                io_nodes.add(node)
            elif node.type in trival_types:
                io_nodes.add(node)
        return io_nodes

    def fetch_while_ctx_nodes(self, sess, nodes, feeds):

        graph = self.sess.graph
        ctx_nodes = set()
        contexts = set()
        for node in nodes:
            ctx = node._control_flow_context
            if ctx is None:
                continue
            if isinstance(ctx, control_flow_ops.WhileContext):
                if ctx in contexts:
                    continue

                for name in ctx._values:
                    tensor = graph.get_tensor_by_name(name)
                    if tensor.op not in nodes:
                        continue
                    ctx_nodes.add(tensor.op)
                    contexts.add(ctx)

        contexts = list(contexts)
        for ctx in contexts:
            loop = ctx.loop_exits[0]
            try:
                values = sess.run([loop] + ctx.loop_enters, feed_dict=feeds)
                self.loop_count_values[ctx] = values[0]
            except Exception as e:
                get_logger().warning(f'fetch while loop count error:{e}')

        return ctx_nodes

    def fetch_cond_ctx_nodes(self, sess, nodes, feeds):

        graph = self.sess.graph
        contexts = set()
        for node in nodes:
            ctx = node._control_flow_context
            if ctx is None:
                continue
            if isinstance(ctx, control_flow_ops.CondContext):
                contexts.add(ctx)

        unwalk = set(contexts)
        walked = set()
        fetchable_map = {None: True}
        while True:
            value_indices = {}
            for ctx in unwalk:
                out_ctx = ctx.outer_context
                if out_ctx is not None and not isinstance(out_ctx, control_flow_ops.CondContext):
                    walked.add(ctx)
                    fetchable_map[ctx] = False
                elif out_ctx in fetchable_map:
                    out_fetchable = fetchable_map[out_ctx]
                    if out_fetchable:
                        value_indices[ctx] = control_flow_ops.merge(
                            ctx.pivot.op.inputs._inputs[0].op.outputs).value_index
                    else:
                        fetchable_map[ctx] = False
                    walked.add(ctx)

            if len(value_indices) > 0:
                value_indices = list(zip(*value_indices.items()))
                try:
                    values = sess.run(value_indices[1], feed_dict=feeds)

                    for i, value in enumerate(values):
                        ctx = value_indices[0][i]
                        fetchable = ctx.pivot.name.endswith('switch_t:0') if value == 1 else ctx.pivot.name.endswith(
                            'switch_f:0')
                        fetchable_map[ctx] = fetchable
                except Exception as nc_vals:
                    get_logger().info(f'fetch cond switch error:{nc_vals}')

            unwalk = unwalk - walked

            if len(unwalk) == 0:
                break

        fetchable_nodes = set()
        unfetchable_nodes = set()
        for ctx, fetchable in fetchable_map.items():
            if ctx is None:
                continue
            names = ctx._values - set(list(ctx._external_values.keys()))
            if ctx._nested_contexts is not None:
                for nc in ctx._nested_contexts:
                    nc_vals = nc._values - set(list(nc._external_values.keys()))
                    names = names - nc_vals - set([x.name for x in nc.pivot.op.inputs[0].op.outputs])
            ts = []
            nodes = set()
            for name in names:
                t = graph.get_tensor_by_name(name)
                if t.op.type in self.ctx_ops:
                    continue
                ts.append(t)
                nodes.add(t.op)
            if self.detect_two_paths(ts):
                continue
            if fetchable:
                fetchable_nodes.update(nodes)
            else:
                unfetchable_nodes.update(nodes)

        fetchable_nodes = sorted([x for x in fetchable_nodes if x.type in self.flops_ops], key=lambda x: x.name)

        return set(fetchable_nodes), unfetchable_nodes

    def detect_two_paths(self, tensors):
        switch_node = None
        for t in tensors:
            if t.op.type != 'Switch':
                continue
            if t.op == switch_node:
                return True
            else:
                switch_node = t.op
        return False

    def get_consumers(self, nodes):
        input_to_output = defaultdict(set)
        for node in nodes:
            for tensor in node.inputs._inputs:
                input_to_output[tensor.op].add(node)
            for c in node.control_inputs:
                input_to_output[c].add(node)
            cfc = node._control_flow_context
            if cfc is not None:
                if isinstance(cfc, control_flow_ops.CondContext):
                    input_to_output[cfc.pivot.op].add(node)
                    input_to_output[cfc.pred.op].add(node)
                elif isinstance(cfc, control_flow_ops.WhileContext):
                    input_to_output[cfc.pivot.op].add(node)
                    for tensor in cfc.loop_enters:
                        input_to_output[tensor.op].add(node)
                    for tensor in cfc.loop_exits:
                        input_to_output[tensor.op].add(node)
        return input_to_output

    def get_risk_nodes(self, nodes):
        return self.get_ctx_nodes_by_subgroup(nodes).union(self.get_ctx_nodes_by_property(nodes))
       

    def get_ctx_nodes_by_subgroup(self, nodes):
        consumers = self.get_consumers(nodes)
        pairs = [('While', 'Enter', 'Exit'),
                 ('Cond', 'Switch', 'Merge')]

        ctx_nodes = set()

        for ctx_type, start_type, end_type in pairs:

            for node in nodes:
                if node.type == start_type:
                    if node in ctx_nodes:
                        continue
                    start_node = node
                    end_nodes = set()

                    walked_nodes = set()
                    unwalked_nodes = {start_node}
                    while True:
                        unwalk = set()
                        for n in unwalked_nodes:
                            if n not in consumers:
                                get_logger().info('op:{} {} not in consumers'.format(n.type, n.name))
                                continue
                            outputs = consumers[n]

                            for output in outputs:
                                if output.type == end_type:
                                    end_nodes.add(output)
                                    continue
                                if output in walked_nodes or output in ctx_nodes:
                                    continue
                                unwalk.add(output)

                        walked_nodes.update(unwalked_nodes)
                        ctx_nodes.update(walked_nodes)
                        unwalked_nodes = unwalk
                        if len(unwalk) == 0:
                            break
                    for node in walked_nodes:
                        consumers[node] = end_nodes
                    ctx_nodes.update(walked_nodes)

        return ctx_nodes

    def get_ctx_nodes_by_property(self, nodes):
        graph = self.sess.graph
        ctx_nodes = set()
        ctx_node_names = set()
        for node in nodes:
            ctx = node._control_flow_context
            if ctx is not None:
                ctx_nodes.add(node)
                ctx_node_names.update(ctx._values - set(list(ctx._external_values.keys())))
            if node._original_op is not None and node._original_op._control_flow_context is not None:
                ctx_nodes.add(node)
        for name in ctx_node_names:
            ctx_nodes.add(graph.get_tensor_by_name(name).op)
        return ctx_nodes

    def get_producers(self, from_names, to_names=None, include=True):

        graph = self.sess.graph
        start_nodes = set()
        walked_nodes = set()
        for name in from_names:
            start_nodes.add(graph.get_tensor_by_name(name).op)
        unwalked_nodes = set(start_nodes)
        end_nodes = set()
        if to_names is not None:
            for name in to_names:
                end_nodes.add(graph.get_tensor_by_name(name).op)

        while True:
            unwalk = set()
            for node in unwalked_nodes:
                for i in node.inputs._inputs:
                    if i.op in end_nodes:
                        continue
                    if i.op in walked_nodes:
                        continue
                    unwalk.add(i.op)
                for n in node.control_inputs:
                    if n in end_nodes:
                        continue
                    if n in walked_nodes:
                        continue
                    unwalk.add(n)
            walked_nodes.update(unwalked_nodes)
            unwalked_nodes = unwalk
            if len(unwalk) == 0:
                break
        if not include:
            for node in start_nodes:
                walked_nodes.remove(node)
        return walked_nodes

    def get_shape(self, tensor, bs=1):
        if tensor.get_shape().ndims is None:
            return [1]
        shape = tensor.get_shape().as_list()
        if any([x is None for x in shape]):
            shape = [bs if x is None else x for x in shape]
        if len(shape) == 0:
            shape = [1]
        return shape

    def flops_by_tf(self):
        options = tf.profiler.ProfileOptionBuilder.float_operation()
        options['output'] = 'none'
        flops_proto = tf.profiler.profile(
            self.sess.graph,
            run_meta=tf.RunMetadata(),
            cmd='graph',
            options=options)
        return flops_proto

    def parse_flops(self, proto):
        node_flops = {}
        unwalked = list(proto.children)
        graph = self.sess.graph
        while True:
            for node in list(unwalked):
                unwalked.remove(node)
                shape_values = []
                for idx, input_shape in node.input_shapes.items():
                    shape_value = [x.size for x in input_shape.dim]
                    shape_values.append(shape_value)
                node_name = node.name
                flops = node.float_ops
                try:
                    op = graph.get_operation_by_name(node_name)
                    node_flops[node_name] = FlopsDetail(node=op, flops=flops, input_shapes=shape_values, ctx_type=None)
                    for child in node.children:
                        if child not in unwalked:
                            unwalked.append(child)
                except:
                    get_logger().error("parse_flops error: {}".format(node))
            if len(unwalked) == 0:
                break
        return node_flops

    def get_unary_flops(self, x):
        a = x.inputs._inputs[0]
        shape = self.get_shape(a, self.bs)
        size = reduce(mul, shape)
        if x.type in ["Cholesky"]:
            flops = size * shape[-1] // 3
        elif x.type in ['Lu', 'Qr']:
            flops = size * shape[-1] * 2 // 3
        elif x.type in ["LogMatrixDeterminant", "MatrixDeterminant", "MatrixInverse",
                        "MatrixSquareRoot", "SelfAdjointEigV2", "Svd"]:
            flops = size * shape[-1]
        elif x.type == 'LRN':
            depth_radius = x.node_def.attr['depth_radius'].i
            flops = size * (2 * depth_radius + 1)
        elif x.type in ('AvgPool3D', 'MaxPool3D', 'MaxPoolV2'):
            # gen_nn_ops.avg_pool3d
            data_format = x.node_def.attr['data_format'].s
            ksize = x.node_def.attr['ksize'].list.i
            padding = x.node_def.attr['padding'].s
            strides = x.node_def.attr['strides'].list.i

            if data_format == 'NCHW':
                shape = itemgetter(0, 2, 3, 1)(shape)
                strides = itemgetter(0, 2, 3, 1)(strides)
                ksize = itemgetter(0, 2, 3, 1)(ksize)
            elif data_format == 'NCDHW':
                shape = itemgetter(0, 2, 3, 4, 1)(shape)
                strides = itemgetter(0, 2, 3, 4, 1)(strides)
                ksize = itemgetter(0, 2, 3, 4, 1)(ksize)

            r = 3 if '3D' in x.type else 2
            if padding == 'SAME':
                flops = reduce(mul, shape) / reduce(mul, strides) * reduce(mul, ksize)
            else:
                n_k = shape[0] * shape[-1] * reduce(mul, [max((shape[i] - ksize[i]) // strides[i] + 1, 1) for i in
                                                          range(1, 1 + r)])
                flops = n_k * reduce(mul, ksize)

        elif x.type in ('FractionalAvgPool', 'FractionalMaxPool'):
            # nn_ops.fractional_avg_pool_v2(d4, [1, 3, 3, 1])
            pooling_ratio = x.node_def.attr['pooling_ratio'].list.f
            overlapping = x.node_def.attr['overlapping'].b
            if overlapping:
                flops = size + (shape[1] / pooling_ratio[1] * shape[2] + shape[2] / pooling_ratio[2] * shape[1]) * \
                        shape[0] * \
                        shape[3]
            else:
                flops = size
        else:
            flops = size
        if x.type in self.flops_per_value:
            flops = flops * self.flops_per_value[x.type]
        return flops

    def get_binary_flops(self, x):
        if len(x.inputs._inputs) < 2:
            get_logger().info(f'error input size < 2{x.type, x.name}')
            return 0
        a = x.inputs._inputs[0]
        b = x.inputs._inputs[1]

        if a.op.type == 'KvVariable' or b.op.type == 'KvVariable':
            return 0

        default_dim = self.bs
        sa = self.get_shape(a, default_dim)
        sb = self.get_shape(b, default_dim)

        if x.type in ('MatMul', 'SparseMatMul', 'BatchMatMul', 'BatchMatMulV2'):
            # transpose_a = x.node_def.attr['transpose_a'].b
            transpose_b = x.node_def.attr['transpose_b'].b
            if x.type == 'BatchMatMulV2':
                sa = self.broadcast([sa[:-2], sb[:-2]]) + sa[-2:]
            flops = reduce(mul, sa) * (sb[-2] if transpose_b else sb[-1])
            # 1MACs = 2FLOPs
            flops *= 2
        elif x.type in ("MatrixSolve", "MatrixTriangularSolve"):
            # gen_linalg_ops.matrix_solve(d2,d2)
            flops = reduce(mul, sa) * sa[0]
        elif x.type in ('Conv2D', 'Conv3D', 'Conv3DBackpropInputV2', 'Dilation2D'):
            strides = x.node_def.attr['strides'].list.i
            data_format = x.node_def.attr['data_format'].s
            padding = x.node_def.attr['padding'].s
            if data_format == 'NCHW':
                sa = itemgetter(0, 2, 3, 1)(sa)
                strides = itemgetter(0, 2, 3, 1)(strides)
            elif data_format == 'NCDHW':
                sa = itemgetter(0, 2, 3, 4, 1)(sa)
                strides = itemgetter(0, 2, 3, 4, 1)(strides)
            if x.type in ('Conv2D', 'Conv3D'):
                if padding == 'SAME':
                    flops = reduce(mul, sa[:-1]) // reduce(mul, strides) * reduce(mul, sb)
                else:
                    r = 2 if x.type == 'Conv2D' else 3
                    n_f = sa[0] * reduce(mul, [max((sa[i] - sb[i - 1]) // strides[i] + 1, 1) for i in range(1, 1 + r)])
                    flops = n_f * reduce(mul, sb)
                flops *= 2
            elif x.type == 'Dilation2D':
                rates = x.node_def.attr['rates'].list.i
                if data_format == 'NCHW':
                    rates = itemgetter(0, 2, 3, 1)(rates)
                if padding == 'SAME':
                    flops = reduce(mul, sa[:-1]) // reduce(mul, strides) * reduce(mul, sb)
                else:
                    n_f = sa[0] * reduce(mul, [max((sa[i] - sb[i - 1]) // (strides[i] * rates[1]) + 1, 1) for i in
                                               range(1, 3)])
                    flops = n_f * reduce(mul, sb)
                flops *= 2
            else:
                flops = reduce(mul, sa) // reduce(mul, strides) * reduce(mul, sb)
        elif x.type in ('SparseSoftmaxCrossEntropyWithLogits', 'SoftmaxCrossEntropyWithLogits'):
            flops = reduce(mul, sb)
            flops *= 6
        else:
            # broadcast
            shape = self.broadcast([sa, sb])
            flops = reduce(lambda x, y: x * y, shape)
            # if x.type in self.flops_per_value:
            #     flops *= self.flops_per_value[x.type]
        return flops

    def get_polytomy_flops(self, x):
        default_dim = self.bs
        if x.type in ("SparseSegmentMean", "SparseSegmentMeanWithNumSegments",
                      "SparseSegmentSqrtN", "SparseSegmentSqrtNWithNumSegments", "SparseSegmentSum",
                      "SparseSegmentSumWithNumSegments"):
            sa = self.get_shape(x.inputs._inputs[0], default_dim)
            sb = self.get_shape(x.inputs._inputs[1], default_dim)
            flops = sa[1] * sb[0]
            if x.type in ('SparseSegmentSqrtNWithNumSegments', 'SparseSegmentSqrtN'):
                flops *= 2
            shapes = [sa, sb]
        elif x.type in ('AddN', 'AccumulateNV2'):
            shapes = [self.get_shape(x) for x in x.inputs._inputs]
            shape = self.broadcast(shapes)
            flops = reduce(mul, shape) * len(shapes)
        elif x.type in ('SparseAdd',):
            sa = self.get_shape(x.inputs._inputs[1], default_dim)
            sb = self.get_shape(x.inputs._inputs[4], default_dim)
            flops = sum(sa + sb)
            shapes = [sa, sb]
        elif x.type in ('SparseTensorDenseMatMul',):
            sa = self.get_shape(x.inputs._inputs[1], default_dim)
            sb = self.get_shape(x.inputs._inputs[3], default_dim)
            adjoint_b = x.node_def.attr['adjoint_b'].b
            if adjoint_b:
                flops = sa[0] * sb[0]
            else:
                flops = sa[0] * sb[1]
            flops *= 2
            shapes = [sa, sb]
        elif x.type in ('SparseTensorDenseAdd', "SparseDenseCwiseAdd", "SparseDenseCwiseDiv",
                        "SparseDenseCwiseMul"):
            # gen_sparse_ops.sparse_tensor_dense_add(indices,values,dense_shape,x)
            shape = self.get_shape(x.inputs._inputs[1], self.bs)
            flops = shape[0]
            shapes = [shape]
        elif x.type in ("SparseReduceMax", "SparseReduceMaxSparse", "SparseReduceSum", "SparseReduceSumSparse"):
            #  sparse_ops.sparse_reduce_max_v2(sp)
            shape = self.get_shape(x.inputs._inputs[1], self.bs)
            flops = shape[0]
            shapes = [shape]
        elif x.type in ('SparseReorder',):
            sa = self.get_shape(x.inputs._inputs[1], default_dim)
            flops = int(sa[0] * np.log2(sa[0]))
            shapes = [sa]
        elif x.type == 'SparseSparseMinimum':
            # gen_sparse_ops.sparse_sparse_minimum
            sa = self.get_shape(x.inputs._inputs[1], default_dim)
            sb = self.get_shape(x.inputs._inputs[4], default_dim)
            flops = sa[0] + sb[0]
            shapes = [sa, sb]
        elif x.type == 'Betainc':
            # gen_math_ops.betainc(d2,d2,d2)
            shape = self.get_shape(x.inputs._inputs[0], default_dim)
            flops = reduce(mul, shape) * 11
            shapes = [shape]
        else:
            flops = 0
            shapes = [1]
        return flops

    @staticmethod
    def gen_dense(shape, dtype, bs=1, default_value=None):

        np_dtype = RuntimeProfiler.get_np_dtype_from_tf_dtype(dtype)
        if default_value is not None:
            if isinstance(default_value, (list, tuple)):
                feat = np.array([default_value] * bs, dtype=np_dtype)
            else:
                feat = np.array([[default_value] * shape] * bs, dtype=np_dtype)
        else:
            if np_dtype == np.str_:
                feat = np.ones([bs, shape], dtype=np_dtype)
            else:
                feat = np.zeros([bs, shape], dtype=np_dtype)
        return feat

    @staticmethod
    def gen_sparse(shape, dtype, bs=1):

        feats = []
        for i, s in enumerate(shape):
            rows = np.arange(bs, dtype=np.int64)
            cols = np.zeros((bs,), dtype=np.int64)
            indices = np.stack([rows, cols], axis=1)
            # apps = np.array(indices)
            # apps[:, 0] = bs-1
            # apps[:, 1] += bs
            # indices = np.concatenate([indices, apps], axis=0)
            values = cols
            if dtype == 3:
                values = values.astype(np.int32)
            elif dtype == 1:
                values = values.astype(np.float32)
            dense_shape = np.array([bs, s], dtype=np.int64)
            sparse_tensor = tf.SparseTensorValue(indices=indices, values=values, dense_shape=dense_shape)
            feats.append(sparse_tensor)

        return feats

    @staticmethod
    def get_np_dtype_from_tf_dtype(dtype):
        #   DT_FLOAT = 1;
        #   DT_DOUBLE = 2;
        #   DT_INT32 = 3;
        #   DT_UINT8 = 4;
        #   DT_INT16 = 5;
        #   DT_INT8 = 6;
        #   DT_STRING = 7;
        #   DT_COMPLEX64 = 8;
        #   DT_INT64 = 9;
        #   DT_BOOL = 10;
        if dtype == 10:
            return np.bool
        elif dtype == 9:
            return np.int64
        elif dtype == 8:
            return np.complex64
        elif dtype == 7:
            return np.str_
        elif dtype == 6:
            return np.int8
        elif dtype == 5:
            return np.uint16
        elif dtype == 4:
            return np.uint8
        elif dtype == 3:
            return np.int32
        elif dtype == 2:
            return np.float64
        elif dtype == 1:
            return np.float32
        else:
            raise ValueError('not support dtype:', dtype)

    @staticmethod
    def get_tf_dtype_from_str(dtype):
        if dtype == 'bool':
            return 10
        elif dtype in ('int64', 'int'):
            return 9
        elif dtype == 'string':
            return 7
        elif dtype == 'int32':
            return 3
        elif dtype in ('float', 'float32', 'double'):
            return 1
        else:
            raise ValueError('unknown dtype:', dtype)

    @staticmethod
    def scope_names(feats, scope='freeze'):
        if isinstance(feats, dict):
            scopes = {}
            for k, v in feats.items():
                scopes[scope + '/' + k] = v
        else:
            scopes = []
            for k in feats:
                scopes.append(scope + '/' + k)
        return scopes

    @staticmethod
    def delete_useless_feeds(feeds, graph):
        nodes = graph.get_operations()
        phs = []
        for x in nodes:
            if x.type in ('Placeholder', 'PlaceholderWithDefault'):
                phs.append(x.name)
        phs = set(phs)

        outputs = {}
        for k, v in feeds.items():
            if k not in phs:
                continue
            outputs[k] = v
        return outputs

    @staticmethod
    def get_outputs(filename):
        sigs = json.loads(open(filename, 'r').read())
        outputs = sigs['outputs']['output']

        names = []
        if isinstance(outputs, str):
            names.append(outputs)
        else:
            for o in outputs:
                names.append(o['tensor_name'])
        return list(set(names))

    def broadcast(self, shapes):
        max_rank = max([len(s) for s in shapes])

        align_shapes = []
        for s in shapes:
            if len(s) < max_rank:
                s = [1] * (max_rank - len(s)) + s
            align_shapes.append(s)

        shape = []
        dims = list(zip(*align_shapes))
        for i in range(max_rank):
            shape.append(reduce(lambda x, y: max(x, y), dims[i]))
        return shape

    def to_pbtxt(self, pb_name, pbtxt_name, serving=True):
        config = tf.ConfigProto(device_count={"CPU": 8})
        sess = tf.Session(config=config)
        self.load_model(pb_name, pbtxt_name, serving=serving)
        tf.train.write_graph(sess.graph, "", pbtxt_name, as_text=True)