def download(self,date):
        log = self.oss2hpc('user_logs/parsed/'+date)
        if isinstance(log, int):
            return
        item = self.oss2hpc('item/'+date)
        if isinstance(item, int):
            return
        ## sample for test or not
        if self.sample:
            log = log.sample(1000)
        item = item.drop(columns=['tdate', 'publishtime'])
        #user = user.drop(columns=['tdate', 'u_example_age'])
        df = pd.merge(log, item, how='inner', left_on='snsid', right_on='snsid')
        #df = pd.merge(df, user, how='left', on='uid')
        df.to_csv('data/laosiji_tmp_log_{}'.format(date),index=False)
        #return df