import json
import time
import requests
import hashlib
import os
#from sklearn.utils.murmurhash import murmurhash3_32

def ding_ding_info(text):
    url = "https://oapi.dingtalk.com/robot/send?access_token=f36ddc9c0874eebd06a24df12c51c89f349130e1b72404a59c6598ac7be69562"
    HEADERS = {
    "Content-Type": "application/json ;charset=utf-8 "
    }
    String_textMsg = {"msgtype": "text","text": {"content": "Time: {0}\n Message:{1}".format(time.strftime("%H:%M:%S"),text)}}
    String_textMsg = json.dumps(String_textMsg)

    try:
        requests.post(url, data=String_textMsg, headers=HEADERS, timeout=0.5)
    except:
        print("Fail to send message to dingding")

#def hash_value(str):
#    return  murmurhash3_32(str)

def hash_value(str):
    md5value = hashlib.md5(str.encode('utf-8')).hexdigest()
    return  int(md5value, 16)

def path_create(path_name, base_path = os.getcwd()):
    path = os.path.join(base_path,path_name)
    if not os.path.exists(path):
        os.makedirs(path)

def path_delete(paths):
    if isinstance(paths,str):
        paths = [paths]
    for path in paths:
        os.system('rm {}'.format(path))
        print('{} removed successfully!'.format(path))

def get_date(gap):
    today = datetime.date.today()
    oneday = datetime.timedelta(days=gap)
    day = today - oneday
    return day.__str__()