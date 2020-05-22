import requests

headers = {
    'Host': 'qunstats.wooyide.com',
    'Content-Type': 'application/json',
    'Access-Token': 'e21fc102a0734b90908aa3097c9af136',
    'Accept': '*/*',
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E217 MicroMessenger/6.8.0(0x16080000) NetType/WIFI Language/en Branch/Br_trunk MiniProgramEnv/Mac',
    'Referer': 'https://servicewechat.com/wx18bc891a3ffd9159/135/page-frame.html',
    'Accept-Language': 'en-us',
}

data = '{"qiandao_id":15656,"text":"刘靖，36.5，正常","images":[],"problems":[],"password":"","index_in_date":100604}'

response = requests.post('https://qunstats.wooyide.com/api/qiandao/v1/checkin/do', headers=headers, data=data.encode(), verify=False)
print(response.text)