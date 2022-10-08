#!/usr/bin/env python
# coding: utf-8

# ## summarization_exam

# In[5]:


get_ipython().system('pip install BeautifulSoup4 lxml requests')


# In[41]:


import requests
from bs4 import BeautifulSoup
import bs4.element
import datetime

# BeautifulSoup 객체 생성
def get_soup_obj(url):
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'}
    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.text,'lxml')
    
    return soup

# 뉴스의 기본 정보 가져오기
def get_top3_news_info(sec, sid):
    # 임시 이미지
    default_img = "https://search.naver.com/search.naver?where=image&sm=tab_jum&query=naver#"
    
     # 해당 분야 상위 뉴스 목록 주소
    sec_url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec"                 + "&sid1="                 + sid
    print("section url : ", sec_url)
    
    # 해당 분야 상위 뉴스 HTML 가져오기
    soup = get_soup_obj(sec_url)
  
    # 해당 분야 상위 뉴스 3개 가져오기
    news_list3 = []
    lis3 = soup.find('ul', class_='type06_headline').find_all("li", limit=3)
    for li in lis3:
        # title : 뉴스 제목, news_url : 뉴스 URL, image_url : 이미지 URL
        news_info = {
            "title" : li.img.attrs.get('alt') if li.img else li.a.text.replace("\n", "").replace("\t","").replace("\r","") , 
            "date" : li.find(class_="date").text,
            "news_url" : li.a.attrs.get('href'),
            "image_url" :  li.img.attrs.get('src') if li.img else default_img
        }
        news_list3.append(news_info)
        
    return news_list3

# 뉴스 본문 가져오기
def get_news_contents(url):
    soup = get_soup_obj(url)
    body = soup.find('div', class_="go_trans _article_content")

    news_contents = ''
    for content in body:
        if type(content) is bs4.element.NavigableString and len(content) > 50:
            # content.strip() : whitepace 제거 (참고 : https://www.tutorialspoint.com/python3/string_strip.htm)
            # 뉴스 요약을 위하여 '.' 마침표 뒤에 한칸을 띄워 문장을 구분하도록 함
            news_contents += content.strip() + ' '
         
    return news_contents
        
    
# '정치', '경제', '사회' 분야의 상위 3개 뉴스 크롤링
def get_naver_news_top3():
    # 뉴스 결과를 담아낼 dictionary
    news_dic = dict()
    
    # sections : '정치', '경제', '사회'
    sections = ["pol", "eco","soc"]
    # section_ids :  URL에 사용될 뉴스  각 부문 ID
    section_ids = ["100", "101","102"]
    
    for sec, sid in zip(sections, section_ids):   
        # 뉴스의 기본 정보 가져오기
        news_info = get_top3_news_info(sec, sid)
        #print(news_info)
        for news in news_info:
            # 뉴스 본문 가져오기
            news_url = news['news_url']
            news_contents = get_news_contents(news_url)
            
            # 뉴스 정보를 저장하는 dictionary를 구성
            news['news_contents'] = news_contents

        news_dic[sec] = news_info
    
    return news_dic
    
# 함수 호출 - '정치', '경제', '사회' 분야의 상위 3개 뉴스 크롤링
news_dic = get_naver_news_top3()
# 경제의 첫번째 결과 확인하기 
news_dic['eco'][0]


# In[2]:


get_ipython().system('pip install transformers')


# ## mT5모델 

# In[8]:


import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# In[42]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')


# In[43]:


def summarization_t5(text):
    tokenized_text = tokenizer(text, return_tensors='pt', truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            tokenized_text['input_ids'],                #text
            do_sample=True,                           #
            eos_token_id = tokenizer.eos_token_id, #eos 토큰 : 1
            max_length = 512,                          #생성할 시퀀스의 최대 길이
            top_p = 0.7,                                   #샘플링을 위해 보관할 매개변수 가장 높은 확률의 어휘 토큰의 누적 확률
            top_k = 20,                                   #필터링을 위해 보관할 확률이 가장 높은 어휘 토큰의 수
            num_beams = 20,
            num_return_sequences = 1,
            no_repeat_ngram_size = 2,
            early_stopping = True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[26]:


# def summarization_t5(sents):
#     article_text = sents
#     input_ids = tokenizer(
#         [WHITESPACE_HANDLER(article_text)],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )["input_ids"]

#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=512,
#         no_repeat_ngram_size=2,
#         num_beams=7
#     )[0]

#     summary = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     return summary


# In[44]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #cpu사용

# 섹션 지정
my_section = 'eco'
news_list3 = news_dic[my_section]
# 뉴스 요약하기
for news_info in news_list3:
    # 뉴스 본문이 10 문장 이하일 경우 결과가 반환되지 않음.
    # 이때는 요약하지 않고 본문에서 앞 3문장을 사용함.
    try:
        snews_contents = summarization_t5(news_info['news_contents'])

    except:
        snews_contents = None

#     if not snews_contents:
#         news_sentences = news_info['news_contents'].split('.')

#         if len(news_sentences) > 3:
#             snews_contents = '.'.join(news_sentences[:3])
#         else:
#             snews_contents = '.'.join(news_sentences)

    news_info['snews_contents'] = snews_contents
    
## 요약 결과 - 첫번째 뉴스
print("==== 첫번째 뉴스 원문 ====")
print(news_list3[0]['news_contents'])
print("\n==== 첫번째 뉴스 요약문 ====")
print(news_list3[0]['snews_contents'])

## 요약 결과 - 두번째 뉴스
print("==== 두번째 뉴스 원문 ====")
print(news_list3[1]['news_contents'])
print("\n==== 두번째 뉴스 요약문 ====")
print(news_list3[1]['snews_contents'])


# ## kobart 모델1

# In[45]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, BartForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
model = BartForConditionalGeneration.from_pretrained('MrBananaHuman/kobart-base-v2-summarization').to('cuda')


# In[80]:


def summarization_kobart1(text):
    tokenized_text = tokenizer(text, return_tensors='pt', truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            tokenized_text['input_ids'],                #text
            do_sample=True,                           #
            eos_token_id = tokenizer.eos_token_id, #eos 토큰 
            max_length = 512,                          #생성할 시퀀스의 최대 길이
            top_p = 0.7,                                   #샘플링을 위해 보관할 매개변수 가장 높은 확률의 어휘 토큰의 누적 확률
            top_k = 20,                                   #필터링을 위해 보관할 확률이 가장 높은 어휘 토큰의 수
            num_beams = 20,
            num_return_sequences = 1,
            no_repeat_ngram_size = 2,
            early_stopping = True
        )
        print('tokenized_text = ',tokenized_text)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[78]:


print(tokenizer.eos_token_id)

print(tokenizer.bos_token_id)


# In[79]:


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf
# import time

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:  # gpu가 있다면, 용량 한도를 5GB로 설정
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], 
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])

# 섹션 지정
my_section = 'eco'
news_list3 = news_dic[my_section]
# 뉴스 요약하기
for news_info in news_list3:

    try:
        snews_contents = summarization_kobart1(news_info['news_contents'])

    except:
        snews_contents = None

#     if not snews_contents:
#         news_sentences = news_info['news_contents'].split('.')

#         if len(news_sentences) > 3:
#             snews_contents = '.'.join(news_sentences[:3])
#         else:
#             snews_contents = '.'.join(news_sentences)

    news_info['snews_contents'] = snews_contents
    
## 요약 결과 - 첫번째 뉴스
print("==== 첫번째 뉴스 원문 ====")
print(news_list3[0]['news_contents'])
print("\n==== 첫번째 뉴스 요약문 ====")
print(news_list3[0]['snews_contents'])

## 요약 결과 - 두번째 뉴스
print("==== 두번째 뉴스 원문 ====")
print(news_list3[1]['news_contents'])
print("\n==== 두번째 뉴스 요약문 ====")
print(news_list3[1]['snews_contents'])


# ## koT5 모델

# In[48]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("psyche/KoT5-summarization")

model = AutoModelForSeq2SeqLM.from_pretrained("psyche/KoT5-summarization").to('cuda')


# In[49]:


def summarization_kot5(text):
    tokenized_text = tokenizer(text, return_tensors='pt', truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            tokenized_text['input_ids'],                #text
            do_sample=True,                           #
            eos_token_id = tokenizer.eos_token_id, #eos 토큰 : 1
            max_length = 512,                          #생성할 시퀀스의 최대 길이
            top_p = 0.7,                                   #샘플링을 위해 보관할 매개변수 가장 높은 확률의 어휘 토큰의 누적 확률
            top_k = 20,                                   #필터링을 위해 보관할 확률이 가장 높은 어휘 토큰의 수
            num_beams = 20,
            num_return_sequences = 1,
            no_repeat_ngram_size = 2,
            early_stopping = True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[50]:


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf
# import time

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:  # gpu가 있다면, 용량 한도를 5GB로 설정
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], 
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])

# 섹션 지정
my_section = 'eco'
news_list3 = news_dic[my_section]
# 뉴스 요약하기
for news_info in news_list3:

    try:
        snews_contents = summarization_kot5(news_info['news_contents'])

    except:
        snews_contents = None

#     if not snews_contents:
#         news_sentences = news_info['news_contents'].split('.')

#         if len(news_sentences) > 3:
#             snews_contents = '.'.join(news_sentences[:3])
#         else:
#             snews_contents = '.'.join(news_sentences)

    news_info['snews_contents'] = snews_contents
    
## 요약 결과 - 첫번째 뉴스
print("==== 첫번째 뉴스 원문 ====")
print(news_list3[0]['news_contents'])
print("\n==== 첫번째 뉴스 요약문 ====")
print(news_list3[0]['snews_contents'])

## 요약 결과 - 두번째 뉴스
print("==== 두번째 뉴스 원문 ====")
print(news_list3[1]['news_contents'])
print("\n==== 두번째 뉴스 요약문 ====")
print(news_list3[1]['snews_contents'])


# ## kobart 모델2

# In[51]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
#  Load Model and Tokenize
tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news").to('cuda')


# In[52]:


# def summarization_kobart(text):
#     tokenized_text = tokenizer(text, return_tensors='pt', truncation=True).to('cuda')
#     with torch.no_grad():
#         outputs = model.generate(
#             tokenized_text['input_ids'],                #text
#             do_sample=True,                           #
#             eos_token_id = tokenizer.eos_token_id, #eos 토큰 : 1
#             max_length = 512,                          #생성할 시퀀스의 최대 길이
#             top_p = 0.7,                                   #샘플링을 위해 보관할 매개변수 가장 높은 확률의 어휘 토큰의 누적 확률
#             top_k = 20,                                   #필터링을 위해 보관할 확률이 가장 높은 어휘 토큰의 수
#             num_beams = 20,
#             num_return_sequences = 1,
#             no_repeat_ngram_size = 2,
#             early_stopping = True
#         )
#         return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[64]:


def summarization_kobart(text):
    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to('cuda')
    # Generate Summary Text Ids
    summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=2.0,
        max_length=142,
        min_length=56,
        num_beams=4)
# Decoding Text
    return tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)


# In[65]:



# 섹션 지정
my_section = 'eco'
news_list3 = news_dic[my_section]
# 뉴스 요약하기
for news_info in news_list3:

    try:
        snews_contents = summarization_kobart(news_info['news_contents'])

    except:
        snews_contents = None

#     if not snews_contents:
#         news_sentences = news_info['news_contents'].split('.')

#         if len(news_sentences) > 3:
#             snews_contents = '.'.join(news_sentences[:3])
#         else:
#             snews_contents = '.'.join(news_sentences)

    news_info['snews_contents'] = snews_contents
    
## 요약 결과 - 첫번째 뉴스
print("==== 첫번째 뉴스 원문 ====")
print(news_list3[0]['news_contents'])
print("\n==== 첫번째 뉴스 요약문 ====")
print(news_list3[0]['snews_contents'])

## 요약 결과 - 두번째 뉴스
print("==== 두번째 뉴스 원문 ====")
print(news_list3[1]['news_contents'])
print("\n==== 두번째 뉴스 요약문 ====")
print(news_list3[1]['snews_contents'])


# In[ ]:


from pororo import P


# In[6]:


print("CUDA version: {}".format(torch.version.cuda))


# In[66]:


model.config.bos_token_id


# ## 뉴스탭 함수

# In[10]:


get_ipython().system('pip install selenium')
get_ipython().system('pip install beautifulsoup4')

from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver as wb


# In[12]:


def webc_news(word):
    baseurl = 'https://search.naver.com/search.naver?where=news&ie=utf8&sm=nws_hty&query='
    url = baseurl + quote_plus(word) #plusUrl의 경우 한글이 허용안되므로 quote_plus를 사용

    cpath = 'C:/Python39/chromedriver_win32/chromedriver.exe'
    driver = wb.Chrome(cpath) #앞으로 사용할 웹 = chrome
    driver.get(url) #url의 주소를 읽음

    html = driver.page_source #주소로부터 읽어온 소스를 html에 저장
    soup = BeautifulSoup(html) #불러온 정보를 Beautifulsoup를 통해 쥬피터노트북에 품

    news_titles = soup.select("a.news_tit")
   

    links = []
    titles = []
    cnt = 0

    for i in news_titles[:5]: #select_one을 쓴이유는 select가 리스트로 받아오기때문
    #.LC20lb.DKYOMd의 경우 LC20lb와 DKYOMd사이가 띄어쓰기되어있는데 .으로 대체한다
        link = i.attrs['href']
        title = i.get_text()
        print(title,link)  


# ## KWS

# In[4]:


get_ipython().system('pip install konlpy')


# In[13]:


get_ipython().system('pip install sentence_transformers')


# In[14]:


import numpy as np
import itertools
from konlpy.tag import Mecab
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# In[15]:


def KWS(doc):

    tokenized_doc = mecab.pos(doc)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'NNG'])

#     print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
#     print('명사 추출 :',tokenized_nouns)

    n_gram_range = (1, 2)

    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names_out()

# print('trigram 개수 :',len(candidates))
# print('trigram 다섯개만 출력 :',candidates[:5])

    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    
    top_n = 2
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return doc_embedding,candidate_embeddings,candidates
    


# In[16]:


def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


# In[17]:


def keyword_preprocess(need_scaled):
    
    spl =[]
    result =''
    
    for i in need_scaled:
        spl += i.split()
    

    drop_duplicate = set(spl)
    

    for d in drop_duplicate:
        result += d
        
    return result


# ## base

# In[18]:


from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver as wb

question = input('무엇을 도와드릴까요?: ')

print('입력하신 검색어는 {}입니다.'.format(question)) 
    #검색어에 오타가 들어갔을 경우 예외처리할것

docs,embed,candidate = KWS(question)
keyword = mmr(docs,embed,candidate, top_n=3, diversity=0.7)
corpus = keyword_preprocess(keyword)


while True:
    
    select = input('카테고리를 선택해주십시오. 영상정보: 1  블로그정보: 2 신문기사: 3  논문정보: 4  종료: 5')
    #카테고리에 해당하는 숫자이외의 숫자 또는 다른 문자가 들어갔을 경우 예외처리
    if not select.isnumeric():  
            print("입력하신 문자:{} \n 잘못 입력하셨습니다.".format(select))
            continue
        
    if select == '1': #키워드바탕으로 영상출력(웹에서 해결)
        print('영상정보 카테고리입니다')
        webc_video(corpus)
        continue
    elif select == '2': #키워드 바탕으로 블로그탭 출력
        print('블로그정보 카테고리입니다')
        webc_blog(corpus)
        continue
    elif select == '3': # 키워드 바탕으로 뉴스탭 출력
        print('뉴스정보 카테고리입니다')
        webc_news(corpus)
        continue
    elif select == '4': #키워드바탕으로 논문탭 출력
        print('논문정보 카테고리입니다')
        webc_paper(corpus)
        continue
    elif select == '5': #종료
        print('종료합니다')
        break
    else:
        print("입력하신 문자:{} \n 해당 카테고리가 없습니다.".format(select))
    
        


# In[ ]:




