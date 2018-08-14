import numpy as np
import tensorflow as tf

class Save(object):
    def __init__(self, model, path):
        self.model = model
        self.export_path = path

    def save(self):
        g = self.model.core.graph
        with g.as_default():
            print('==============After training==============')
            print("Number of sets of all parameters: {}".format(len(tf.global_variables())))
            print("Number of all parameters: {}".format(
                np.sum([np.prod(v.shape.as_list()) for v in tf.global_variables()])))
            for v in tf.global_variables():
                print(v)

            print("Number of sets of trainable parameters: {}".format(len(tf.trainable_variables())))
            print("Number of trainable parameters: {}".format(
                np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])))
            for v in tf.trainable_variables():
                print(v)
            print('The type of model.core.b')
            print(type(self.model.core.b))
            print(self.model.core.b)
            print('The type of model.core.W[0]')
            print(type(self.model.core.w[0]))
            print(self.model.core.w[0])
            print('The type of model.core.W[1]')
            print(type(self.model.core.w[1]))
            print(self.model.core.w[1])

            builder = tf.saved_model.builder.SavedModelBuilder(self.export_path)
            raw_indices = tf.saved_model.utils.build_tensor_info(self.model.core.raw_indices)
            raw_data = tf.saved_model.utils.build_tensor_info(self.model.core.raw_values)
            raw_shape = tf.saved_model.utils.build_tensor_info(self.model.core.raw_shape)

            outputs = tf.saved_model.utils.build_tensor_info(
                self.model.core.outputs)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        'raw_indices': raw_indices,
                        'raw_data': raw_data,
                        'raw_shape': raw_shape,
                    },
                    outputs={
                        'outputs': outputs,
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            )
            legacy_init_op = tf.group(tf.tables_initializer(), name='laosiji_recsys')

            builder.add_meta_graph_and_variables(
                self.model.session, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict':
                        prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        prediction_signature,
                },
                legacy_init_op=legacy_init_op)
            builder.save()

            print('Done exporting!')