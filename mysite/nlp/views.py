from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
# from django.shortcuts import HttpResponse
import joblib
import jieba
import jieba.analyse as analyse
import json
import os
from statsmodels.tsa.api import ExponentialSmoothing
import statsmodels.api as sm
import numpy as np

# Create your views here.
curPath = os.path.dirname(__file__)  # Python/mysite/nlp
parPath = os.path.abspath(os.path.join(curPath, os.pardir))  # Python/mysite
parPath2 = os.path.abspath(os.path.join(parPath, os.pardir))  # Python/mysite
# print(curPath, parPath, parPath2)
stopwords_path = curPath + '/stopword.txt'
# stopwords_path = r'/root/workspace/mysite/nlp/stopword.txt'
stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
TFIDF_model = joblib.load(parPath2 + '/cli/TFIDF.model')
model = joblib.load(parPath2 + '/cli/bayes.model')

minLen=5 # 允许展示词云图的最小长度
arimaModel = None

def del_stopwords(sentence):
    result = []
    for word in sentence:
        if word in stopwords:
            continue
        else:
            result.append(word)
    return result


def cut_words(sentence_list):
    result = []
    for sentence in sentence_list:
        word_list = jieba.cut(sentence)
        word_list = del_stopwords(word_list)
        sentence_cut = ' '.join(word_list)
        result.append(sentence_cut)
    return result


def getScoreListAndTag(sentenceList):
    sentenceCut = cut_words(sentenceList)
    sentence_all = ' '.join(sentenceCut)
    dataTFIDF = TFIDF_model.transform(sentenceCut)
    predictPro = model.predict_proba(dataTFIDF)
    temp = []
    tags = analyse.extract_tags(sentence_all, topK=10, withWeight=False)
    for predict in predictPro:
        temp.append(predict[1])
    # print(temp)
    return temp, tags


def UploadFile(request):
    if request.method == "POST":
        f = request.FILES['TxtFile']
        sentence = f.read()
        sentence = sentence.decode("utf-8")
        sentenceList = sentence.split('\n')
        List, tags = getScoreListAndTag(sentenceList)
        sum = 0.0
        for score in List:
            sum += score * 5
        Star = round(sum / len(List), 1)  # 这里返回满分为五分的评分
        temp = {'Star':Star, 'Tags':tags}
        content = json.dumps(temp)
        response = HttpResponse(content=content, content_type='application/json')
        return response


def UploadText(request):
    if request.method == "POST":
        sentence = request.POST.get("sentence", None)
        sentenceList = []
        sentenceList.append(sentence)
        scoreList, tags = getScoreListAndTag(sentenceList)
        star = round(scoreList[0] * 5, 1)
        temp = {'Star':star, 'Tags':tags}
        content = json.dumps(temp)
        response = HttpResponse(content=content, content_type='application/json')
        return response


def inputForm(request):
    return render(request, "inputForm.html")


def getPost(request):
    if request.method == 'POST':
        try:
            req = json.loads(request.body)
            sentence = req['text']
            sentenceList = []
            sentenceList.append(sentence)
            scoreList, tags = getScoreListAndTag(sentenceList)
            return JsonResponse({'state':1, 'score':scoreList[0], 'tags':tags})
        except Exception as e:
            return JsonResponse({'state':0})
    else:
        return JsonResponse({'state':0})


def forecast_es(request):
    if request.method == "POST":
        try:
            req = json.loads(request.body)
            # print(req)
            data = req["data"]
            period = req["period"]
            num = req["num"]
            model = ExponentialSmoothing(np.asarray(data),
                             seasonal_periods=period,
                             trend='add',
                             seasonal='add').fit()
            forecast = model.forecast(num)
            # print(data, period, num, forecast)
            return JsonResponse({'state':0, 'forecast': list(forecast)})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500})
    else:
        return JsonResponse({'state': 400})


def forecast_es_index(request):
    if request.method == "POST":
        try:
            req = json.loads(request.body)
            data = req["data"]
            period = req["period"]
            index = req["index"]
            model = ExponentialSmoothing(np.asarray(data),
                             seasonal_periods=period,
                             trend='add',
                             seasonal='add').fit()
            forecast = model.forecast(index + 1)
            return JsonResponse({'state':0, 'forecast': forecast[index]})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500})
    else:
        return JsonResponse({'state': 400})


def forecast_arima_214(request):
    if request.method == "POST":
        try:
            req = json.loads(request.body)
            data = req["data"]
            period = req["period"]
            num = req["num"]
            model = sm.tsa.statespace.SARIMAX(data,
                                 order=(2, 1, 4),
                                 seasonal_order=(0, 1, 1,period)).fit()
            forecast = model.forecast(num)
            # print(data, period, num, forecast)
            return JsonResponse({'state':0, 'forecast': list(forecast)})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500})
    else:
        return JsonResponse({'state': 400})


def initModel_arima_010(request):
    if request.method == "POST":
        try:
            req = json.loads(request.body)
            data = req["data"]
            period = req["period"]
            num = req["num"]
            arimaModel = sm.tsa.statespace.SARIMAX(data,
                                 order=(0, 1, 0),
                                 seasonal_order=(0, 1, 1,period)).fit()
            # print(data, period, num, forecast)
            return JsonResponse({'state':0})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500})
    else:
        return JsonResponse({'state': 400})


def forecast_arima_010(request):
    if request.method == "POST":
        try:
            req = json.loads(request.body)
            data = req["data"]
            period = req["period"]
            num = req["num"]
            if arimaModel == None:
                return JsonResponse({'state': 1})
            forecast = arimaModel.forecast(num)
            # print(data, period, num, forecast)
            return JsonResponse({'state':0})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500, 'forecast': list(forecast)})
    else:
        return JsonResponse({'state': 400})
    

# cd mysite
# python3 manage.py runserver 0.0.0.0:8080
