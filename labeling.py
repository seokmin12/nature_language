from tqdm import tqdm
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame

with open("/Users/seokmin/Desktop/python/nature_language/data/positive_words.txt", encoding='utf-8') as pos:
    positive = pos.readlines()
with open("/Users/seokmin/Desktop/python/nature_language/data/negative_words.txt", encoding='utf-8') as neg:
    negative = neg.readlines()
positive = [pos.replace("\n", "") for pos in positive]
negative = [neg.replace("\n", "") for neg in negative]

day_time = datetime.today() - timedelta(5001)
labels = []
title_list = []
for i in tqdm(range(5000)):
    day = day_time - timedelta(days=i)
    url = f"https://finance.naver.com/news/mainnews.naver?date={str(day.strftime('%Y-%m-%d'))}"
    request = requests.get(url)
    soup = BeautifulSoup(request.text, 'lxml')

    titles = soup.select("#contentarea_left > div.mainNewsList > ul > li > dl > dd.articleSubject > a")

    for title in titles:
        title_data = title.text
        clean_title = re.sub("[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]", "", title_data)
        negative_flag = False
        label = 0

        for n in range(len(negative)):
            if negative[n] in clean_title:
                label = -1
                negative_flag = True
                break
        if not negative_flag:
            for n in range(len(positive)):
                if positive[n] in clean_title:
                    label = 1
                    break
        title_list.append(clean_title)
        labels.append(label)

raw_data = {'title': title_list,
            'label': labels}
data = DataFrame(raw_data)
print(data)
data.to_csv('/Users/seokmin/Desktop/python/sentiment_stock/machine_learning/data/test_data.csv', index=False)
