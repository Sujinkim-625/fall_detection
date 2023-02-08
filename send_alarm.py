#!/usr/bin/env python
# coding: utf-8

# In[52]:


import requests
from dotenv import load_dotenv
import os

url = 'https://kauth.kakao.com/oauth/token'
rest_api_key = '' # 추후 env 파일 불러오기
redirect_uri = 'https://naver.com'






# json 저장
import json
import requests

with open("C:\\Users\\marui\\kakao_code4.json","r") as fp:
    tokens = json.load(fp)

url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

person = ["홍길동", "010-1234-5678", "서울시 종로구"]
alarm = "낙상 감지후 일어서지 못함"


headers={
    "Authorization" : "Bearer " + tokens["access_token"]
}


data={
    "template_object": json.dumps({
        "object_type":"text",
        "text": "이름 : " + person[0] 
        + "\n" + "전화번호 : " + person[1] 
        + "\n" + "주소 : " + person[2] 
        + "\n" +"현재상황 : " + alarm,
        "link":{
            "web_url":"www.naver.com"
        }
    })
}

def send_alarm_kakao():
    response = requests.post(url, headers=headers, data=data)
    response.status_code
    print(response.status_code)
    if response.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))



