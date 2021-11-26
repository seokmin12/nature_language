from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import re
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

okt = Okt()

max_len = 20
max_words = 35000

# load model
model = load_model('/Users/seokmin/Desktop/python/nature_language/news_model.h5')

# loading tokenizer
with open('/Users/seokmin/Desktop/python/nature_language/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def sentiment_predict(new_sentence):
    clean_sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》·]', '', new_sentence)
    clean_sentence = okt.morphs(clean_sentence, stem=True)  # 토큰화
    clean_sentence = [word for word in clean_sentence if not word in stopwords]  # 불용어 제거
    sequences = tokenizer.texts_to_sequences([clean_sentence])  # 정수 인코딩
    pad_new = pad_sequences(sequences, maxlen=max_len)  # 패딩
    score = model.predict(pad_new)
    predict_score = np.argmax(score)
    if predict_score == 0:
        print(new_sentence)
        print("부정적인 기사입니다.")
    elif predict_score == 1:
        print(new_sentence)
        print("중립적인 기사입니다.")
    elif predict_score == 2:
        print(new_sentence)
        print("긍정적인 기사입니다.")


sentiment_predict('오지환, 쇄골 골절 진단…역전 우승 노리는 LG의 대형악재')

# 오지환, 쇄골 골절 진단…'역전 우승' 노리는 LG의 대형악재   0
# CS "중국 부동산업 불황, 철광석·석탄 전망에는 호재"  2
# 이재명, 이번엔 ‘52조 돈살포’… 野 “매표행위”  1
# [특징주]엔피, 네이버·위지윅·YG와 메타버스 동맹…자이언트스텝과 키맞추기  1
# 플레이그램, 새 주인 맞아 900억 현금 확보… NFT·소셜카지노 사업 시동  1
