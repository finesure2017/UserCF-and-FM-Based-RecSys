from grpc.beta import implementations
import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import logging
from utils import hash_value
import time
class Predict(object):
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 9000
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.max_value_dict = {"communitylist":10000, "u_communitylist":10000, "snsid":100000,"uid":200000,"authorid":200000}
        self.sparse_columns = ['snsid','uid','authorid']
        self.label_columns = ['is_click']
        self.used_columns = ['snsid','uid','authorid']
    def predict_item_by_user(self,query_use_id, candidate_items):
        input= self.input_data_prepare(candidate_items)
        sns_ids = [item.get('snsid',0) for item in candidate_items]
        for item in candidate_items:
            item['uid'] = query_use_id
            #item.update(self.user_features.get(query_use_id,{}))
        logger = logging.getLogger()
        logger.info("candidate_items is {}".format(candidate_items[0]))
        logger.info("input is {}".format(input.tocsr()))
        input_parse = input.tocoo()
        raw_indices = np.hstack(
            (input_parse.row[:, np.newaxis], input_parse.col[:, np.newaxis])
        ).astype(np.int64)
        raw_data = input_parse.data.astype(np.float32)
        tf.contrib.util.make_tensor_proto(raw_data)
        raw_shape = np.array(input_parse.shape).astype(np.int64)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'laosiji_recsys'
        request.inputs['raw_indices'].CopyFrom(tf.contrib.util.make_tensor_proto(raw_indices))
        request.inputs['raw_data'].CopyFrom(tf.contrib.util.make_tensor_proto(raw_data))
        request.inputs['raw_shape'].CopyFrom(tf.contrib.util.make_tensor_proto(raw_shape))
        logger.info('Start to predict')
        start = time.time()
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        result = self.stub.Predict(request, timeout=0.3)  # 0.3 secs timeout
        logger.info('predict cost time {}'.format(time.time() - start))
        scores = np.array(result.outputs['outputs'].float_val).tolist()
        result = list(zip(sns_ids,list(scores),candidate_items))
        logger.info("return predict number {}".format(len(result)))
        return result
    @staticmethod
    def generate_sparse_column_vector(size, values):
        row = []
        col = []
        for loc, value in enumerate(values):
            if value:
                if isinstance(value, int):
                    row.append(loc)
                    col.append(hash_value(str(value)) % size)
                if isinstance(value, str):
                    detail_values = value.split(',')
                    row.extend([loc] * len(detail_values))
                    col.extend([hash_value(v) % size for v in detail_values])
        data = [1] * len(row)
        return sparse.csr_matrix((data, (row, col)), shape=(len(values), size))
    def input_data_prepare(self,data):
        if isinstance(data,dict):
            data = [data]
        feature_value_dict = {}
        for column in self.sparse_columns:
            feature_value_dict[column] = [item.get(column) for item in data]
        features = []
        for column in self.used_columns:
            if column in self.sparse_columns:
                features.append(self.generate_sparse_column_vector(self.max_value_dict[column], feature_value_dict[column]))
        inference_data = sparse.hstack(features)
        return inference_data
if __name__ == '__main__':
    items = [{'snsid': 367511, 'communitylist': '257', 'ishotshow': 0, 'publishtime': '2018-04-14 18:29:21', 'clickcount': 207, 'readcount': 2356, 'commentcount': 2, 'forward': 0, 'anonymouspraisenum': 0, 'gratuitycount': 0, 'praisecount': 3, 'uid': 174415}]
    pre = Predict()
    pre.predict_item_by_user(100,items)
        