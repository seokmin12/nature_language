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

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

total_cnt = len(tokenizer.word_index)
rare_cnt = 0
threshold = 3

for key, value in tokenizer.word_counts.items():
    if value < threshold:
        rare_cnt = rare_cnt + 1

vocab_size = total_cnt - rare_cnt + 1
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)

import pickle
# saving tokenizer
with open('/Users/seokmin/Desktop/python/nature_language/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    elif train_data['label'].iloc[i] == -1:
        y_train.append([1, 0, 0])
for i in range(len(test_data['label'])):
    if test_data['label'].iloc[i] == 1:
        y_test.append([0, 0, 1])
    elif test_data['label'].iloc[i] == 0:
        y_test.append([0, 1, 0])
    elif test_data['label'].iloc[i] == -1:
        y_test.append([1, 0, 0])

y_train = np.array(y_train)
y_test = np.array(y_test)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

max_len = 15  # 전체 데이터의 길이를 15로 맞춘다
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('/Users/seokmin/Desktop/python/nature_language/news_model.h5', monitor='val_acc', mode='max',
                     verbose=1, save_best_only=False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=20, validation_split=0.4)

print(f"\n 테스트 정확도: {round(model.evaluate(X_test, y_test)[1] * 100, 2)}%")
