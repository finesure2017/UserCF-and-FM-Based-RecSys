# coding: utf-8
import oss2

TRAINING_DATA_SETTINGS = {
    'OSS_HOST': 'oss-cn-hangzhou.aliyuncs.com',
    'OSS_ID': "LTAIACcC6aEwl3WB",
    'OSS_KEY': "AHLLPtdJnlutFAs9YRG9uGN2MDhjZF",
    'OSS_DBNAME': "fengchao-eigen"
}

OSS_MODEL_SETTINGS = {
    "OSS_HOST": 'oss-cn-hangzhou.aliyuncs.com',
    "OSS_ID": "LTAIACcC6aEwl3WB",
    "OSS_KEY": "AHLLPtdJnlutFAs9YRG9uGN2MDhjZF",
    "OSS_DBNAME": "modelzoo",
}

class OssDriver(object):
    def __init__(self, settings):
        """
        oss相关操作
        :param center: 配置中心句柄
        :param oss_config_root_path: oss配置根路径，在此路径下可以找到所有oss地址信息
        """
        self.bucket = self.initial_bucket(**settings)

    @staticmethod
    def initial_bucket(OSS_ID, OSS_KEY, OSS_HOST, OSS_DBNAME):
        """
        初始化oss bucket
        """
        #print(OSS_ID)
        return oss2.Bucket(oss2.Auth(OSS_ID, OSS_KEY), OSS_HOST, OSS_DBNAME)

    def is_exist(self, path):
        res = self.bucket.object_exists(path)
        if res:
            return True
        return False

    def download(self, path):
        return self.bucket.get_object(path).read()

    def upload(self,obj_key,path):
        return self.bucket.put_object_from_file(obj_key,path)