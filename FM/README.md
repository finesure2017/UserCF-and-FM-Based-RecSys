# laosiji_recsys
>为老司机APP进行个性化推荐

## 推荐架构
* 第一步：FM
> 利用协同过滤进行粗筛
 
* 第二步：deep-and-wide
> 进行精确化推荐与多样性性调节

### 数据
* user_logs

	* raw\_laosiji_userlog
 
		>见excel
	* dcrawl\_parsed\_laosiji_userlogs
		>见下表
 
		列名|类型|说明|
		---- | --- | --- |
		begin | string | 暂时没用
		duration | float | 暂时没用
		is_click|int|1:点击；0:未点击|
		snsid|int|帖子编号|
		uid|int|用户编号|
		tdate|string(TODO...)|日志时间|

* topic
	*  dcrawl\_raw\_laosiji_items
	> 见下表
	
		列名|类型|说明|
		---- | --- | --- |
		id | string | 
		json | string | 
		tdate|string|分区|
	*  dcrawl\_parsed\_laosiji_userlogs
	
		> 从raw表中取出item所有属性，但因目前item很多属性为空，此表暂时未用，目前直接将数据写入algo表
	
	*  dcrawl\_algo\_laosiji_items
		
		> 见下表

		
		列名|类型|说明|
		---- | --- | --- |
		id | int | 
		communitylist | string | 所属圈子信息
		ishotshow | int | 0/1
		cilckcount | int |
		commentcount | int |
		anonymouspraisenum | int |
		gratuitycount | int | 
		praise | int |
		forward | int |
		readcount | int |

* community
>文件不大，定期更新，存放在laosiji\_recsys/support\_data/community


###数据流
* 帖子
> 每天执行
> 
> /ETL/lib/workflows/laosiji\_items_workflow.py

	order| stage |intput_table| intput_table | 说明
	--- | ---- | --- | --- | --- |
	1 | sql2hive_full | sql: item| dcrawl\_raw\_laosiji_items | 将当天的全量item原始信息以日期分片存放 
	2 | extract_item | dcrawl\_raw\_laosiji_items | dcrawl\_algo\_laosiji_items | 直接解析至algo表
	3 | hive2sql | dcrawl\_algo\_laosiji_items| sql: parsed_item| 全量更新至sql供inference使用
	4 | hive2oss | dcrawl\_algo\_laosiji_items | osskey: /item/{tdate} | 全量更新至sql供算法使用
	
* 日志
> 每天执行
> 
> /ETL/lib/workflows/laosiji\_userlogs_workflow.py

	order| stage |intput_table| intput_table | 说明
		--- | ---- | --- | --- | --- |
		1 | oss2hdfs | osskey: /user_logs/{tdate}| hdfs | 将昨天的log原始信息存放至hdfs 
		2 | join2one | hdfs | dcrawl\_raw_laosiji_userlogs | 增加分区信息
		3 | extract_userlogs | dcrawl\_raw_laosiji\_userlogs| dcrawl\_parsed\_laosiji_userlogs | 抽取部分有用log（目前只关心点击数据，TODO...增加阅读量）
		4 | parsed2algo | dcrawl\_parsed\_laosiji_userlogs | dcrawl\_algo\_laosiji_userlogs | 结合item，生成算法可使用的数据（目前只关心点击数据，TODO...增加阅读量）
		5 | 
	
	