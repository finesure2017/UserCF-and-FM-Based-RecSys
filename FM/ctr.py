import datetime
import pickle

Host = "rm-bp1pyu5wq71ufm9a5.mysql.rds.aliyuncs.com"
User = "eigenrec"
Passwd = "eigenreceigen123"
Database = "eigenrec"

class Ctr(object):
    def __init__(self,df):
        self.df = df
        self.conn = pymysql.connect(host=Host, user=User, passwd=Passwd, db=Database, charset='utf8')

    @staticmethod
    def get_earilyest_date(date):
        date_time = datetime.datetime.strptime(date, '%Y-%m-%d')
        earliest_date = str((date_time - datetime.timedelta(days=60)).date())
        return earliest_date

    def get_selected_sns_ids(self,earliest_date):
        cursor = self.conn.cursor()
        sql_cmd = 'select id from parsed_item where priority= \'1\' and publishtime > \'{}\''.format(earliest_date)
        cursor.execute(sql_cmd)
        values = cursor.fetchall()
        cursor.close()
        snsids = [int(item[0]) for item in values]
        return snsids

    def get_beta_user_ids(self, file_path = 'data/beta_user.pkl'):
        beta_user = pickle.load(open(file_path, 'rb'))
        return beta_user

    def total_user_ctr(self):
        return self.df['is_click'].sum() / self.df.shape[0]

    def beta_user_ctr(self):
        df = self.df[self.df['uid'].isin(self.beta_user)]
        return df['is_click'].sum() / df.shape[0]

    def total_user_ctr_in_selected_articles(self, snsids):
        df = self.df[self.df['snsid'].isin(snsids)]
        return df['is_click'].sum() / df.shape[0]

    def beta_user_ctr_in_selected_articles(self,beta_user,snsids):
        df = self.df
        df = df[df['uid'].isin(beta_user)]
        df = df[df['snsid'].isin(snsids)]
        return df['is_click'].sum() / df.shape[0]

    def beta_user_ctr_in_predict_selected_articles(self,beta_user, predict_result, topk):
        click_count = 0
        for user in beta_user:
            res = predict_result[user][:topk]
            df = self.df
            df = df[df['uid'] == user]
            df = df[df['snsid'].isin(res)]
            click_count += df['is_click'].sum()
        return click_count / (topk * len(beta_user))