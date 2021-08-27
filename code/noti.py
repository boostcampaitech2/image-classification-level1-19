import json
import sys
import random
import requests

class notion:
    def __init__(self):
        pass
    
    def send_message(self,m:str):
        return m

if __name__ == '__main__':
    note = notion()
    url = 'https://hooks.slack.com/services/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'# 웹후크 URL 입력
    message = note.send_message() # 메세지 입력
    title = (f"New Incoming Message :zap:") # 타이틀 입력
    slack_data = {
        "username": "NotificationBot", # 보내는 사람 이름
        "icon_emoji": ":satellite:",
        #"channel" : "#somerandomcahnnel",
        "attachments": [
            {
                "color": "#9733EE",
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ]
            }
        ]
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
