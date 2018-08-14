# coding: utf-8
import json
from datetime import datetime
import re
import math
import numpy as np
from simplex.model import LaosijiItemEmbedding
from scipy import sparse
import hashlib
import eigen_config

def hash_value(str):
    md5value = hashlib.md5(str.encode('utf-8')).hexdigest()
    return  int(md5value, 16)

class LaosijiProcess(object):
    def __init__(self):
        self.rank = 10
        online_version = eigen_config.get_global_configs().get('online_version')
        self.online_version = online_version
        self.max_value_dict = eigen_config.get_global_configs().get(online_version).get('max_value_dict')
        self.dense_columns = eigen_config.get_global_configs().get(online_version).get('dense_columns')
        self.sparse_columns = eigen_config.get_global_configs().get(online_version).get('sparse_columns')
        self.label_columns = eigen_config.get_global_configs().get(online_version).get('label_columns')
        self.used_columns = eigen_config.get_global_configs().get(online_version).get('used_columns')

    @staticmethod
    def community_extract(community):
        if community:
            return ','.join(map(str, [j['id'] for j in community]))
        return ''

    @staticmethod
    def get_example_age(impress_time, publish_time):
        """返回帖子从生产到被展示的时间，0表示当天"""
        delta = impress_time - publish_time
        return delta.days

    @staticmethod
    def get_date(impress_time):
        # datetime.weekday()返回结果Monday=0,Sunday=6 所以+1作为返回结果
        return impress_time.weekday() + 1

    @staticmethod
    def get_hour(impress_time):
        """按照阅读时间对hour进行分类
        [1~6) :0
        [6~10):1
        [10~14):2
        [14~17):3
        [17~21):4
        [21~1):5
        """
        hour = impress_time.hour
        if 1 <= hour < 6:
            return 0
        elif 6 <= hour < 10:
            return 1
        elif 10 <= hour < 14:
            return 2
        elif 14 <= hour < 17:
            return 3
        elif 17 <= hour < 21:
            return 4
        else:
            return 5

    def get_features(self,item):
        impress_time = datetime.now()
        publish_time = item['publishtime']
        if publish_time.count(':') == 1:
            publish_time = publish_time + ':00'
        pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')
        publish_time = datetime(*map(int, pattern.match(publish_time).groups()))
        item['example_age'] = self.get_example_age(impress_time,publish_time)
        item['week'] = self.get_date(impress_time)
        item['hour_mapping'] = self.get_hour(impress_time)
        return item

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

    @staticmethod
    def generate_dense_column_vector(values, max_value = None):
        if max_value:
            values = [value/max_value for value in values]
        else:
            values = [math.log(value + 1) for value in values]
        return np.transpose(np.matrix(values))

    def input_data_prepare(self,data):
        if isinstance(data,dict):
            data = [data]
        feature_value_dict = {}
        for column in self.sparse_columns:
            feature_value_dict[column] = [item.get(column) for item in data]
        for column in self.dense_columns:
            feature_value_dict[column] = [item.get(column, 0) for item in data]
        features = []
        for column in self.used_columns:
            if column in self.sparse_columns:
                features.append(self.generate_sparse_column_vector(self.max_value_dict[column], feature_value_dict[column]))
            if column in self.dense_columns:
                features.append(self.generate_dense_column_vector(feature_value_dict[column],max_value=self.max_value_dict.get(column)))
        inference_data = sparse.hstack(features)
        return inference_data

    def get_item_embedding(self, item):
        item_feature = self.input_data_prepare(item)
        try:
            item_embedding = self.model.predict(item_feature)
        except:
            item_embedding = np.zeros(self.rank)
        item['embedding'] = item_embedding
        return item

    def raw_item_process(self, item):
        """
        parse the raw item
        :param item: json, raw info of item
        :return:
        """
        item = json.loads(item['result'])['body']['sns']
        if not item:
            return
        parsed_item = {}
        parsed_item['id'] = item['id']
        parsed_item['communitylist'] = self.community_extract(item['communitylist'])
        parsed_item['ishotshow'] = item['ishotshow']
        parsed_item['publishtime'] = item['publishtime']
        parsed_item['clickcount'] = item['clickcount']
        parsed_item['readcount'] = item['readcount']
        parsed_item['commentcount'] = item['commentcount']
        parsed_item['forward'] = item['forward']
        parsed_item['anonymouspraisenum'] = item['anonymouspraisenum']
        parsed_item['gratuitycount'] = item['gratuitycount']
        parsed_item['praisecount'] = item['praisecount']
        parsed_item['uid'] = item['user']['id']
        return parsed_item

    def process(self, item, raw = True):
        if raw:
            item = self.raw_item_process(item)
        item = self.get_features(item)
        return item