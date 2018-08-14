import os, time
import pandas as pd
import datetime
from multiprocessing import Pool
from oss import OssDriver, TRAINING_DATA_SETTINGS
from utils import path_delete

class Data(object):
    def __init__(self, date, gap, sample, reserve):
        self.oss = OssDriver(TRAINING_DATA_SETTINGS)
        self.date = date
        self.gap = gap
        self.sample = sample
        self.reserve = reserve

    def oss2hpc(self, prefix):
        ## prefix list:'user_logs/parsed/','item/','user/'
        if not self.oss.is_exist(prefix):
            print('{} doesn\'t exist!'.format(prefix))
            return 1
        tmp = self.oss.bucket.get_object(prefix).read()
        print('{} download successfully!'.format(prefix))

        base_prefix = prefix.split('/')[0]
        date = prefix.split('/')[-1]
        tmp_file_path = 'data/' + base_prefix + '_' + date

        with open(tmp_file_path, 'wb') as f:
            f.write(tmp)
        df = pd.read_csv(tmp_file_path)
        # remove file if not needed
        if not self.reserve:
            path_delete(tmp_file_path)
        return df

    def _load(self,date):
        log = self.oss2hpc('user_logs/parsed/'+date)
        if isinstance(log, int):
            return
        item = self.oss2hpc('item/'+date)
        if isinstance(item, int):
            return
        user = self.oss2hpc('user/'+date)
        if isinstance(user, int):
            return

        ## sample for test or not
        if self.sample:
            log = log.sample(1000)

        item = item.drop(columns=['tdate', 'publishtime'])
        #user = user.drop(columns=['tdate', 'u_example_age'])
        df = pd.merge(log, item, how='inner', left_on='snsid', right_on='snsid')
        #df = pd.merge(df, user, how='left', on='uid')
        df.to_csv('data/laosiji_tmp_log_{}'.format(date),index=False)
        return df

    def get_train_dates(self):
        end_date = datetime.datetime.strptime(self.date, '%Y-%m-%d')
        train_dates = []
        for time_diff in reversed(range(self.gap)):
            train_date = end_date - datetime.timedelta(time_diff)
            train_dates.append(train_date)
        train_dates = [train_date.strftime('%Y-%m-%d') for train_date in train_dates]
        return train_dates

    @staticmethod
    def raw_data_process(df):
        print("origin data shape is{}".format(df.shape))
        df = df.dropna(axis=0, subset=['snsid', 'uid'])
        df = df[df['uid'] != 0]
        print("data shape after moving illegal user is{}".format(df.shape))
        valid_user = df.groupby('uid')['is_click'].sum()
        user_list = list(valid_user[valid_user > 0].index)
        df = df[df['uid'].isin(user_list)]
        print("data shape after moving inactivate user is{}".format(df.shape))
        return df

    def load(self):
        train_dates = self.get_train_dates()
        test_date = train_dates[-1]
        p = Pool(10)
        for date in train_dates:
            p.apply_async(self._load, args=(date,))
            #p.apply(self._load, args=(date,))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')

        # union all data
        data_files = ['data/' + file for file in os.listdir('data') if file.startswith('laosiji_tmp_log_')]
        print("data files is {}".format(data_files))
        df = pd.concat([pd.read_csv(file) for file in data_files], axis=0)
        df = self.raw_data_process(df)
        #remove tmp file
        if not self.reserve:
            path_delete(data_files)
        return df,test_date

    def single_load(self):
        df = self._load(self.date)
        df = df.dropna(axis=0, subset=['snsid', 'uid'])
        df = df[df['uid'] != 0]
        return df
