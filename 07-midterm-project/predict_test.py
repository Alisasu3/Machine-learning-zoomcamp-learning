#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

people = {
    "highbp":1,
    "highchol":0,
    "cholcheck":1,
    "bmi":38,
    "smoker":1,
    "stroke":0,
    "heartdiseaseorattack":0,
    "physactivity":1,
    "fruits":1,
    "veggies":1,
    "hvyalcoholconsump":0,
    "anyhealthcare":1,
    "nodocbccost":0,
    "genhlth":3,
    "menthlth":0,
    "physhlth":0,
    "diffwalk":0,
    "sex":"female",
    "age":12,
    "education":5,
    "income":8
}


response = requests.post(url,json=people).json()
print(response)


if response['diabetes'] == True:
    print('this people has high diabetes risk')
else:
    print('this person has low diabetes risk')





