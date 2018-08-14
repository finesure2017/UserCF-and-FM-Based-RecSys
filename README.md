# UserCF-and-FM-Based-RecSys
实现了一个汽车资讯推荐系统，使用UserCF与FM模型
### 基于CF与FM的汽车资讯推荐系统

### 1.数据
* 认识数据
* 测试集与测试集    
 
### 2.模型
*  UserCF 
*  FM

### 3.评测
### 4.备注















#### 1.数据

* 1.1 认识数据  
数据按天存储，每一天的log数据记录着每一个用户的行为。
体现用户与资讯文章之间的交互。  
其中包括用户对文章的点击，在该文章界面的停留时间，当前行为的发生时间，文章所属于的community等等。
![](https://ws1.sinaimg.cn/large/0069RVTdgy1fu98ptlhz0j30bg058glq.jpg)


每天的log数据，是一个二维表。每一行是一个行为记录。  

	uid : 用户id   (int)  
	snsid : 资讯文章id  （int）  
	is_click : 是否点击 （0：未点击，1：点击）  
	duration : 停留时间  （s,未点击为-1.0）  
	tdate : 事务时间  （YY-MM-DD）  
	comuntitylist : 文章所属community   
	ishotshow : 是否是hotshow  
	click_count : 文章点击总数量  
	read_count ; 文章阅读总数量   
	comment_count ： 文章评论总数  
	forward : 是否转发  
	authorid : 文章作者id  

![](https://ws1.sinaimg.cn/large/0069RVTdgy1fu98p7xontj30ds027q3a.jpg)

* 1.2 训练集  
 选取5月11日到6月3的最活跃1000个用户作为偏好集.  其他用户与该1000用户计算相似度
 去除无点击的用户，无点击文章 ，得到uid×snsid即
231052×6838的数据集  
按照每个用户的点击量排序 ,选择点击量最多的即最活跃的1000个用户。作为参照。测试集中的每一个用户与这1000个用户计算相似度。
 
 * 去除无点击(僵尸)用户，无点击文章
 
  ![](https://ws2.sinaimg.cn/large/0069RVTdgy1fu99j3d43sj30s60d1wfi.jpg)

* 1.3 测试集  
使用6月1日到6月3日的log作为测试数据。  
对6月1日到6月3日的N个活跃用户作为测试用户进行UserCF推荐，  N取1000，5000，10000，50000   
若用户在此日中点击了推荐的文章，则表明推荐正确。  



#### 2.模型

#### 2.1 UserCF
基于用户的协同过滤系统，通过其近邻(即偏好相似的用户)的爱好向用户做推荐。  

* 寻找相似用户  
* 通过其相似爱好向用户做推荐。  
* 在用户看到文章集中选取30篇文章，排序后作为推荐结果  
 
采用pearson相似度，通过近邻25个邻居获得其偏好
![](https://ws3.sinaimg.cn/large/006tKfTcgy1ft4lvsbk9uj30rv074tas.jpg)
 
 *  pearson相似度
 *  25个邻居 (可调整)
 *  在用户看得到的文章中做推荐

#### 2.2 FM
[什么是FM？](https://tracholar.github.io/machine-learning/2017/03/10/factorization-machine.html)    
使用FM的一个开源实现tffm 
( TensorFlow  Factorization Machine）[https://github.com/geffy/tffm]  
调用方法：
![](https://ws4.sinaimg.cn/large/0069RVTdgy1fu9a547nx8j30eo06tt9h.jpg)

#### 3.评测

##### 评测指标
* TopK 点击率
![](https://ws4.sinaimg.cn/large/0069RVTdgy1fu9a1lfx78j30vo0mdn26.jpg)

#### 3.1 UserCF
对6月1日到6月3日的N个活跃用户作为测试用户评测   
 N取1000，5000，10000，50000
	计算推荐的文章中的Top 10,Top20,Top30的点击率
 ![](https://ws3.sinaimg.cn/large/0069RVTdgy1fu99uhasc6j30o606f3zb.jpg)


* 原始log中的Total点击率  

![](https://ws4.sinaimg.cn/large/0069RVTdgy1fu99v73owtj30qk07vwfq.jpg)
 
 可见，相对与原始推荐系统而言，再采用UserCF作为推荐模型时，其TopK点击率有了显著的提升，尤其是对于那些相对活跃的用户。  
#### 3.2 FM 
 
 ![](https://ws2.sinaimg.cn/large/0069RVTdgy1fu99xm9aptj30qq07lwfb.jpg)
  可见，相对与原始推荐系统而言，再采用FM作为推荐模型时，其TopK点击率提升更加明显。
  
FM vs UserCF

* 性能：FM推荐速度更快
* TopK 点击率 FM相对于UserCF约提升30%左右


#### 4.备注

* 从OSS上下载数据
![](https://ws4.sinaimg.cn/large/0069RVTdgy1fu9abuiyprj30pt0ahacq.jpg)
* 使用多线程
	见
* FM训练，使用tensorflow-serving，以及调用
	
	
```
python run.py --gpu 2 --gap 30 --date 2018-06-01 --feature uid,snsid,authorid

tensorflow_model_server --port=9000 --model_base_path=/root/export

调用见test_cliet.py
