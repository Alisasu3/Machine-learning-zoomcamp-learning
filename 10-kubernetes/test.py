# import requests

# url = 'http://localhost:9696/predict'

# data = {'url': 'http://bit.ly/mlbookcamp-pants'}

# result = requests.post(url, json=data).json()
# print(result)

import requests

url = "http://localhost:9696/predict"
data = {"url": "http://bit.ly/mlbookcamp-pants"}

resp = requests.post(url, json=data)

print("STATUS:", resp.status_code)
print("HEADERS:", resp.headers.get("Content-Type"))
print("TEXT (first 500 chars):")
print(resp.text[:500])
