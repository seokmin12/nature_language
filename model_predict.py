from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import re
import pandas as pd
from tqdm import tqdm

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

okt = Okt()

max_len = 20
max_words = 35000

model = load_model('/Users/seokmin/Desktop/python/nature_language/news_model.h5')

train_data = pd.read_csv("/Users/seokmin/Desktop/python/nature_language/data/train_data.csv")

train_data.drop_duplicates(subset=['title'], inplace=True)  # title 열에서 중복인 내용이 있다면 중복 제거
train_data['title'] = train_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data['title'] = train_data['title'].str.replace('^ +', "")  # white space 데이터를 empty value로 변경
train_data['title'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')


X_train = []
for sentence in tqdm(train_data['title']):
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_train.append(temp_X)


def sentiment_predict(new_sentence):
    clean_sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》·]', '', new_sentence)
    clean_sentence = okt.morphs(clean_sentence, stem=True)  # 토큰화
    clean_sentence = [word for word in clean_sentence if not word in stopwords]  # 불용어 제거
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences([clean_sentence])  # 정수 인코딩
    pad_new = pad_sequences(sequences, maxlen=max_len)  # 패딩
    score = model.predict(pad_new)
    predict_score = np.argmax(score)
    if predict_score == 1:
        print(new_sentence)
        print("부정적인 기사입니다.")
    else:
        print(new_sentence)
        print("긍정적인 기사입니다.")


sentiment_predict('SM C&C·SM Life Design, 모두 상승세로…특히 SM C&C 가파른 상승세로')
