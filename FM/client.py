import logging
import time
import eigen_config
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from simplex.du.laosiji_du import LaosijiProcess
from WebServer.webapp.common.utils import measure_time
from sklearn.metrics.pairwise import cosine_similarity
from simplex.utils import oss_to_memory
import pickle
from io import BytesIO
from simplex.model import LaosijiItemEmbedding

logger = logging.getLogger(__name__)
user_features_path = 'oss://modelzoo/dev/user_features.pkl'
user_features =  pickle.load(BytesIO(oss_to_memory(user_features_path).read()))

class laosiji:
    def __init__(self, host='laosiji-recsys.dev.svc.k8s.local', port=8500):
        online_version = eigen_config.get_global_configs().get('online_version')
        self.host = eigen_config.get_global_configs().get(online_version).get('tensorflow_serving_host')
        self.port = port
        self.channel = implementations.insecure_channel(self.host, int(self.port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.user_features = user_features
        self.processor = LaosijiProcess()
        self.embedding = LaosijiItemEmbedding("grpc://laosiji-item-embedding.dev.svc.k8s.local:8500")
        
    @measure_time
    def predict_item_by_user(self,query_use_id, candidate_items):
        sns_ids = [item.get('snsid',0) for item in candidate_items]
        for item in candidate_items:
            item['uid'] = query_use_id
            item.update(self.user_features.get(query_use_id,{}))
        input = self.processor.input_data_prepare(candidate_items)
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
        ## 下面这步在第一次调用时耗时三秒，程序中得调本函数预热
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
        # print(result)
        logger.info("return predict number {}".format(len(result)))
        return result

    @measure_time
    def predict_item_by_item(self, query_item, candidate_items):
        query_item = self.item_feature_qu(query_item)
        logger.info("query_item is {}".format(query_item))
        logger.info("candidate_items is {}".format(candidate_items[0]))
        sns_ids = [item.get('snsid',0) for item in candidate_items]
        query_item_input = self.processor.input_data_prepare(query_item)
        logger.info("query_item is {}".format(query_item))
        query_item_embedding = self.embedding.predict(query_item_input)
        candidate_items_input = self.processor.input_data_prepare(candidate_items)
        embedding_matrix = self.embedding.predict(candidate_items_input)
        scores = cosine_similarity(query_item_embedding,embedding_matrix)[0]
        result = list(zip(sns_ids,list(scores),candidate_items))
        return result

    def item_feature_du(self, item):
        return item

    def item_feature_qu(self, item):
        item['snsid'] = item['id']
        item['authorid'] = item['uid']
        #del item['id']
        del item['uid']
        return item

    def user_feature_du(self, item):
        return item

    def user_feature_qu(self, uid):
        return self.user_features.get(uid,{})

_INSTANCE = None
_VERSION = None
    
def get_client():
    global _INSTANCE
    global _VERSION
    online_version =  eigen_config.get_global_configs().get('online_version')
    if _VERSION != online_version:
        _INSTANCE = None
        _VERSION = online_version
        logger.info('*'*60)
        logger.info('version change to {}, client has been reset!'.format(online_version))
    if _INSTANCE is None:
        _INSTANCE = laosiji()
        logger.info('laosiji du version {}'.format(_VERSION))
    return _INSTANCE

def predict_item_by_user():
    lsj = laosiji()
    items = [{'id': 367511, 'communitylist': '257', 'ishotshow': 0, 'publishtime': '2018-04-14 18:29:21', 'clickcount': 207, 'readcount': 2356, 'commentcount': 2, 'forward': 0, 'anonymouspraisenum': 0, 'gratuitycount': 0, 'praisecount': 3, 'uid': 174415}]
    user_id = '100'
    result = lsj.predict_item_by_user(user_id, items)
    return result

def predict_item_by_item():
    lsj = laosiji()
    item_features = {'example_age': 23, 'commentcount': 0, 'gratuitycount': 0, 'clickcount': 1, 'id': 380169, 'praisecount': 0, 'week': 7, 'readcount': 1, 'hour_mapping': 5, 'publishtime': '2018-04-20 11:23:12', 'uid': 237224, 'embedding': np.array([[-0.37659507,  0.25975709,  0.54597071, -0.35006334,  1.80925135,
        -0.23168316,  1.84003456, -1.17873854, -1.45284694, -0.11932197]]), 'communitylist': '', 'forward': 0, 'anonymouspraisenum': 0, 'ishotshow': 0}
    query_item = {"item_features":item_features}
    items = [query_item]
    result = lsj.predict_item_by_item(query_item, items)
    return result

if __name__ == '__main__':
    print(predict_item_by_item())
