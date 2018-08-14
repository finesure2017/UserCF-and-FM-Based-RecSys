from scipy import sparse
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import hashlib
from multiprocessing import Pool,Queue,Manager,Array
dense_columns = []
sparse_columns = ['uid','snsid','authorid']
used_columns = ['uid','snsid','authorid']
max_value_dict = {"communitylist":10000, "u_communitylist":10000, "snsid":100000,"uid":200000,"authorid":200000}
items = [{'uid':'100','snsid': 367511, 'authorid': 174415}]
host = 'localhost'
port = '9000'
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
import pandas as pd 
testLog = pd.read_csv("laosiji_tmp_log_2018-06-01",usecols = ['uid','snsid','authorid','is_click'])
testset = (testLog.loc[testLog['is_click']==1]).pivot_table("is_click",'uid','snsid',fill_value = 0)
testset['click_count'] = testset.apply(lambda x:sum(x),axis =1)
testset.sort_values(by = "click_count",ascending = False,inplace = True)
del testset['click_count']

def hash_value(str):
    md5value = hashlib.md5(str.encode('utf-8')).hexdigest()
    return  int(md5value, 16)
@staticmethod
def generate_dense_column_vector(values):
    return values

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
def input_data_prepare(data):
    if isinstance(data,dict):
        data = [data]
    #print(data)
    feature_value_dict = {}
    for column in sparse_columns:
        feature_value_dict[column] = [item.get(column) for item in data]
    for column in dense_columns:
        feature_value_dict[column] = [item.get(column, 0) for item in data]
    features = []
    for column in used_columns:
        if column in sparse_columns:
            features.append(generate_sparse_column_vector(max_value_dict[column], feature_value_dict[column]))
        if column in dense_columns:
            features.append(generate_dense_column_vector(feature_value_dict[column]))
    inference_data = sparse.hstack(features)
    return inference_data


def predict(items):
    #print(items)
    Input = input_data_prepare(items)
    input_parse = Input.tocoo()
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
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result = stub.Predict(request, timeout=0.3)
    scores = np.array(result.outputs['outputs'].float_val).tolist()
    return scores
def Evaluation(num):
    sum_ctr10,sum_ctr20,sum_ctr30,avg_ctr10,avg_ctr20,avg_ctr30 = 0.0,0.0,0.0,0.0,0.0,0.0
    recommend_num = 0
    for uid in testset.head(num).index:
        candidate_items = []
        tmp = testLog.loc[testLog['uid'] == uid]
        if len(tmp) < 30:
            continue
        candidate_items = list(tmp[['uid','snsid','authorid']].T.to_dict().values())
        scores = predict(candidate_items)
        clicks = list(tmp['is_click'].values)
        ctr10,ctr20,ctr30 = ctr(scores,clicks)
        sum_ctr10 += ctr10
        sum_ctr20 += ctr20
        sum_ctr30 += ctr30
        #print("recommending",uid,ctr10,ctr20,ctr30)
        recommend_num += 1
    #print("recommend_num: ",recommend_num)
    avg_ctr10,avg_ctr20,avg_ctr30 = sum_ctr10/recommend_num,sum_ctr20/recommend_num,sum_ctr30/recommend_num
    print('Top10 ctr:',avg_ctr10,'Top20 ctr:',avg_ctr20,'Top30 ctr:',avg_ctr30)
def ctr(scores,clicks):
    z = list(zip(scores,clicks))
    z.sort(key = lambda x:x[0],reverse = True)
    z =[x[1] for x in z ]
    ctr10 = np.mean(z[:10])
    ctr20 = np.mean(z[:20])
    ctr30 = np.mean(z[:30])
    return ctr10,ctr20,ctr30



from multiprocessing import Pool,Queue,Manager,Array
def multi():
     p = Pool(10)
     for i in range(10):
             p.apply_async(task,args = (i*1,(i+1)*1,))
     print('Waiting for all subprocesses done...')
     p.close()
     p.join()
     print('All subprocesses done.')
def task(start,end):
    sum_ctr10,sum_ctr20,sum_ctr30,avg_ctr10,avg_ctr20,avg_ctr30 = 0.0,0.0,0.0,0.0,0.0,0.0
    recommend_num = 0
    for uid in testset.head(end).tail(end-start).index:
        candidate_items = []
        tmp = testLog.loc[testLog['uid'] == uid]
        if len(tmp) < 30:
            continue
        candidate_items = list(tmp[['uid','snsid','authorid']].T.to_dict().values())
        try:
            scores = predict(candidate_items)
        except:
            continue
        clicks = list(tmp['is_click'].values)
        ctr10,ctr20,ctr30 = ctr(scores,clicks)
        sum_ctr10 += ctr10
        sum_ctr20 += ctr20
        sum_ctr30 += ctr30
        recommend_num += 1
        #print("recommending",uid,ctr10,ctr20,ctr30)
    print("recommend_num: ",recommend_num)
    avg_ctr10,avg_ctr20,avg_ctr30 = sum_ctr10/recommend_num,sum_ctr20/recommend_num,sum_ctr30/recommend_num
    #print('Top10 ctr:',avg_ctr10,'Top20 ctr:',avg_ctr20,'Top30 ctr:',avg_ctr30)
    print ('[',avg_ctr10,',',avg_ctr20,',',avg_ctr30,'],')