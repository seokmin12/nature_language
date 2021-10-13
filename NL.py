import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

train_data = pd.read_csv("/Users/seokmin/Desktop/python/nature_language/data/train_data.csv")
test_data = pd.read_csv("/Users/seokmin/Desktop/python/nature_language/data/test_data.csv")

train_data.drop_duplicates(subset=['title'], inplace=True)  # title 열에서 중복인 내용이 있다면 중복 제거
train_data['title'] = train_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data['title'] = train_data['title'].str.replace('^ +', "")  # white space 데이터를 empty value로 변경
train_data['title'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')

test_data.drop_duplicates(subset=['title'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
test_data['title'] = test_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['title'] = test_data['title'].str.replace('^ +', "")  # 공백은 empty 값으로 변경
test_data['title'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

from konlpy.tag import Okt

okt = Okt()

X_train = []
for sentence in tqdm(train_data['title']):
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_train.append(temp_X)

X_test = []
for sentence in tqdm(test_data['title']):
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_test.append(temp_X)

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 35000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

import numpy as np

y_train = []
y_test = []
for i in tqdm(range(len(train_data['label']))):
    if train_data['label'].iloc[i] == 1:
        y_train.append([0, 0, 1])
    elif train_data['label'].iloc[i] == 0:
        y_train.append([0, 1, 0])
for i in range(len(test_data['label'])):
    if test_data['label'].iloc[i] == 1:
        y_test.append([0, 0, 1])
    elif test_data['label'].iloc[i] == 0:
        y_test.append([0, 1, 0])

y_train = np.array(y_train)
y_test = np.array(y_test)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

max_len = 20  # 전체 데이터의 길이를 20로 맞춘다
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('/Users/seokmin/Desktop/python/nature_language/news_model.h5', monitor='val_acc', mode='max',
                     verbose=1, save_best_only=False)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=20, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

predict = model.predict(X_test)

import numpy as np
predict_labels = np.argmax(predict, axis=1)
original_labels = np.argmax(y_test, axis=1)

for i in range(30):
    print()
    print("기사제목 : ", test_data['title'].iloc[i], "/\t 원래 라벨 : ", original_labels[i], "/\t예측한 라벨 : ", predict_labels[i])
