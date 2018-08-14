from scipy import sparse
import numpy as np
from utils import hash_value

class Featurelize(object):
    def __init__(self, features):
        self.max_value_dict = {"communitylist":10000, "u_communitylist":10000, "snsid":100000,"uid":200000,"authorid":200000}
        self.dense_columns = []
        self.sparse_columns = ['snsid','uid','authorid']
        self.label_columns = ['is_click']
        self.used_columns = features.split(',')

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
    def generate_dense_column_vector(values):
        return values

    def features_dict_prepare(self,df):
        feature_dict = {}
        train_columns = self.used_columns + self.label_columns
        for column in train_columns:
            print("handling feature {}".format(column))
            raw_data = df[column].tolist()
            feature = None
            if column in self.sparse_columns:
                feature = self.generate_sparse_column_vector(self.max_value_dict[column],raw_data)
            elif column in self.dense_columns:
                feature = self.generate_dense_column_vector(raw_data)
            elif column in self.label_columns:
                feature = self.generate_dense_column_vector(raw_data)
            feature_dict[column] = feature
        return feature_dict

    def input_data_prepare(self, feature_dict):
        features = []
        for column in self.used_columns:
            features.append(feature_dict[column])
        train_data = sparse.hstack(features).tocsr()
        label_data = np.array(feature_dict['is_click'])
        return train_data,label_data

    def get_feature_nums(self):
        feature_nums = 0
        for column in self.used_columns:
            if column in self.max_value_dict:
                feature_nums += self.max_value_dict[column]
            else:
                feature_nums += 1
        return feature_nums