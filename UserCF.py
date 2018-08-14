import pandas as pd 
import numpy as np 
import os
import math
import random
from sklearn.cluster import KMeans
from multiprocessing import Pool,Queue,Manager,Array

#获得全部文章
def TotalArticles():
    file = '/data/public/yinghua/code/laosiji_recsys/data/item_2018-06-03'
    df = pd.read_csv(file,usecols= ['snsid','publishtime'])#,index_col = 'snsid')
    df.sort_values("publishtime",ascending=False,inplace=True)
    return df

def MakeTrainSet(Dir):
	#耗时6分钟
	files = [Dir+ log for log in list(os.listdir(Dir)) if log.startswith("laosiji_tmp_log_2018-05")]
	logs = []
	for log in files:
		tmp = pd.read_csv(log,usecols = ['uid','is_click','snsid'],dtype = {'snsid':'int64'})
		tmp = tmp.loc[tmp['is_click'] == 1]
		logs.append(tmp)
	Log = pd.concat(logs)
	#231052*6838 去除无点击的用户，无点击文章 #全部删除click_count列
	dataset = Log.pivot_table("is_click",'uid','snsid',fill_value = 0)
	dataset['click_count'] = dataset.apply(lambda x:sum(x),axis =1)
	dataset.sort_values(by = "click_count",ascending = False,inplace = True)
	#dataset.to_csv("TrainSet.csv")
	del dataset['click_count']
	return dataset
def MakeTestSet(Dir):
	files = [Dir+ log for log in list(os.listdir(Dir)) if log.startswith("laosiji_tmp_log_2018-06")]
	logs = []
	for log in files:
		tmp = pd.read_csv(log,usecols = ['uid','is_click','snsid'],dtype = {'snsid':'int64'})
		tmp = tmp.loc[tmp['is_click'] == 1]
		logs.append(tmp)
	testLog = pd.concat(logs)
	#112983*2803,#去除无点击的用户，无点击文章,#全部删除click_count列
	testset = testLog.pivot_table("is_click",'uid','snsid',fill_value = 0)
	testset['click_count'] = testset.apply(lambda x:sum(x),axis =1)
	testset.sort_values(by = "click_count",ascending = False,inplace = True)
	#dataset.to_csv("TestSet.csv")
	del testset['click_count']
	files = [Dir+ log for log in list(os.listdir(Dir)) if log.startswith("laosiji_tmp_log_2018-06")]
	logs = []
	for log in files:
		tmp = pd.read_csv(log,usecols = ['uid','is_click','snsid'],dtype = {'snsid':'int64'})
		logs.append(tmp)
	testLog = pd.concat(logs)
	return testLog,testset
def recommender(uid,num_of_neighbors=25):
	#uid 不在trainSet中？
	#neighbors = (topusers.corrwith(testset.loc[uid],axis = 1)).sort_values(ascending = False).head(25)
	neighbors = (topusers.corrwith(trainset.loc[uid],axis = 1)).sort_values(ascending = False).head(num_of_neighbors)  #???????
	recommend = pd.Series(0,index = trainset.columns)
	for neighbor in neighbors.index:
		recommend += neighbors[neighbor]*topusers.loc[neighbor]
	#筛选，做推荐
	series = testset.loc[uid]
	views = (testLog.loc[testLog['uid'] == uid])['snsid'].values    #给uid推荐的文章
	#snsid 不在trainSet中？
	recommend = recommend.loc[views]     #或者recommend = pd.Series(recommend,index= views)
	recommend.sort_values(ascending=False,inplace=True)
	recommend.fillna(0,inplace = True)
	return recommend
def Evaluation(num):
	sum_ctr10,sum_ctr20,sum_ctr30,avg_ctr10,avg_ctr20,avg_ctr30 = 0.0,0.0,0.0,0.0,0.0,0.0
	not_in_train = [uid for uid in testset.head(num).index if uid not in trainset.index]
	recommend_num = 0
	for uid in testset.head(num).index:
		if uid in not_in_train:
			continue
		series = testset.loc[uid]
		clicks = series.sort_values(ascending = False).head(sum(series)).index #uid点击的文章
		views = (testLog.loc[testLog['uid'] == uid])['snsid'].values
		#若views小于30怎么办
		if len(views)<30:
			continue
		try:
			recommend = recommender(uid)
		except:
			continue
		hit,ctr10,ctr20,ctr30 = 0,0.0,0.0,0.0
		for i,snsid in enumerate(recommend.index):
			if snsid in clicks:
				hit += 1
			if i == 9:
				ctr10 = 1.0*hit/10
				sum_ctr10 += ctr10
			elif i == 19:
				ctr20 = 1.0*hit/20
				sum_ctr20 += ctr20
			elif i == 29:
				ctr30 = 1.0*hit/30
				sum_ctr30 += ctr30
				break
		print "recommend user:",uid,'Top10 ctr:',ctr10,'Top20 ctr:',ctr20,'Top30 ctr:',ctr30
		recommend_num += 1
	avg_ctr10,avg_ctr20,avg_ctr30 = sum_ctr10/recommend_num,sum_ctr20/recommend_num,sum_ctr30/recommend_num
	print 'Top10 ctr:',avg_ctr10,'Top20 ctr:',avg_ctr20,'Top30 ctr:',avg_ctr30
#使用多线程
def task(start,end):
	#[0,100)
	#[start,end)
	if end < start:
		print("Error :end < start")
		exit()
	sum_ctr10,sum_ctr20,sum_ctr30,avg_ctr10,avg_ctr20,avg_ctr30 = 0.0,0.0,0.0,0.0,0.0,0.0
	not_in_train = [uid for uid in testset.head(end).tail(end - start).index if uid not in trainset.index]
	recommend_num = 0
	for uid in (testset.head(end).tail(end - start)).index:#random.sample((testset.head(end).tail(end - start)).index,100):#(testset.head(end).tail(end - start)).index:
		if uid in not_in_train:    #uid not in trainset 
			continue
		#if uid%10 != 7:
		#	continue   #抽样，减少测试时间
		series = testset.loc[uid]
		clicks = series.sort_values(ascending = False).head(sum(series)).index #uid点击的文章
		views = (testLog.loc[testLog['uid'] == uid])['snsid'].values
		#若views小于30怎么办
		if len(views)<30:
			continue
		try:
			recommend = recommender(uid)    #snsid not in trainset
		except:
			continue
		hit,ctr10,ctr20,ctr30 = 0,0.0,0.0,0.0
		for i,snsid in enumerate(recommend.index):
			if snsid in clicks:
				hit += 1
			if i == 9:
				ctr10 = 1.0*hit/10
				sum_ctr10 += ctr10
			elif i == 19:
				ctr20 = 1.0*hit/20
				sum_ctr20 += ctr20
			elif i == 29:
				ctr30 = 1.0*hit/30
				sum_ctr30 += ctr30
				break
		#print "recommend user:",uid,'Top10 ctr:',ctr10,'Top20 ctr:',ctr20,'Top30 ctr:',ctr30
		recommend_num += 1
	avg_ctr10,avg_ctr20,avg_ctr30 = sum_ctr10/recommend_num,sum_ctr20/recommend_num,sum_ctr30/recommend_num
	#print 'Top10 ctr:',avg_ctr10,',','Top20 ctr:',avg_ctr20,'Top30 ctr:',avg_ctr30
	print '[',avg_ctr10,',',avg_ctr20,',',avg_ctr30,'],'
def multiprocess():
     p = Pool(10)
     for i in range(10):
             p.apply_async(task,args = (i*1000,(i+1)*1000,))
     print('Waiting for all subprocesses done...')
     p.close()
     p.join()
     print('All subprocesses done.')


#统计原始log中的结果
##6.1-6.3每天的testLog,testset 获取每天的testLog,testset
#6.1-6.3全部的Log
#Log = pd.merge(testLog,articles)
def OriginalTest(i): # i= 1 即6.1 ...
	file = "/home/lintaofang/data/laosiji_tmp_log_2018-06-0"+str(i)
	Log = pd.read_csv(file,usecols = ['uid','is_click','snsid'],dtype = {'snsid':'int64'})
	tmp = Log.loc[Log['is_click'] == 1]
	tmp = tmp.pivot_table("is_click",'uid','snsid',fill_value = 0)
	tmp['click_count'] = tmp.apply(lambda x:sum(x),axis =1)
	tmp.sort_values(by = "click_count",ascending = False,inplace = True)
	del tmp['click_count']
	return Log,tmp
def LogSet():
	testlogs = []
	testsets = []
	for i in range(1,4):
		testlog,testset = OriginalTest(i)
		Log = pd.merge(testLog,articles)
		testlogs.append(Log)
		testsets.append(testset)
	return testlogs,testsets
#6.1 -6.3 每天测试点击率
def original(start,end):
	sum_ctr10,sum_ctr20,sum_ctr30 = 0.0,0.0,0.0
	for i in range(1,2):
		testLog,testset = testlogs[i],testsets[i]
		recommend_num = 0
		for uid  in (testset.head(end).tail(end - start)).index:
			#ctr10,ctr20,ctr30 = 0,0.0,0.0,0.0
			userlog = testLog.loc[testLog['uid']==uid]
			if len(userlog)<30:
				continue
			userlog.sort_values('publishtime',ascending = False)
			sum_ctr10 += sum(userlog.head(10)['is_click'])
			sum_ctr20 += sum(userlog.head(20)['is_click'])
			sum_ctr30 += sum(userlog.head(30)['is_click'])
			recommend_num += 1
			# series = testset.loc[uid]
			# views = len(set((testLog.loc[testLog['uid'] == uid])['snsid']))
			# total_click_rate += sum(series)*1.0/views
		#print '06-0'+str(i),
		print '[',sum_ctr10*1.0/(recommend_num*10),',',sum_ctr20*1.0/(recommend_num*20),',',sum_ctr30*1.0/(recommend_num*30),'],'

if __name__ == '__main__':
	Dir = "/home/lintaofang/data/"
	trainset =  MakeTrainSet(Dir)
	testLog,testset =MakeTestSet(Dir)
	topusers = trainset.head(1000)
	articles = TotalArticles()
	task(1,1000)
	#统计原始log中的结果
	testlogs,testsets = LogSet()
	original(1,1000)








'''
#做个聚类
#耗时太长时间
def clusters():
	km = KMeans(n_clusters = 100,random_state = 1)
	km.fit(dataset)#dataset = dataset.applymap(lambda x:1 if x>0 else 0)
	return km.cluster_centers_
clusters = pd.DataFrame(km.cluster_centers_,columns = trainset.columns)
clusters.to_csv("100clusters.csv",index = False)

#Load 重命名列 str->int
clusters = pd.read_csv("/home/lintaofang/100clusters.csv")
for i in clusters.columns:
	d['i'] = int(i)
clusters.rename(columns = d,inplace = True)


#更为简单的 
#Load 重命名列 str->int
clusters.columns = list(map(int,list(clusters.columns)))
'''

