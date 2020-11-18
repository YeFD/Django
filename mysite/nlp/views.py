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
path = curPath + "/static/dict_small.txt"
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
        sentenceList = sentence.split("。")
        words = []
        flags = []
        for s in sentenceList:
            w, f = BiMM(s)
            words.append(w)
            flags.append(f)
        temp = {'words':words, 'flags':flags}
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

# path = "./static/dict_small.txt"
maxSize = 5
print(path)
flagList = {"n": "n", "f": "n", "s": "n", "t": "n",
            "nr": "n", "ns": "n", "nt": "n", "nw": "n",
            "nz": "n", "v": "v", "vd": "v", "vn": "vn",
            "a": "adj", "ad": "adj", "an": "adj", "d": "adv",
            "m": "adj", "q": "adj", "r": "r", "p": "介词",
            "c": "连词", "u": "助词", "xc": "助词", "w": "标点符号",
            "uj": "助词", "ul": "le", "uz": "助词", "uv": "助词",
            "ud": "助词", "ug": "助词", "e": "语气助词", "k": "n",
            "b": "adj", "y": "语气助词", "j": "n", "ng": "n",
            "h": "adj", "df": "v", "z": "n", "l": "n",
            "vg": "v", "tg": "n", "nrt": "n", "rz": "r",
            "nrfg": "n", "ag": "adj", "g": "n", "mq": "r",
            "x": "n", "dg": "n", "i": "n", "o": "adv",
            "rr": "r", "vq": "v", "rg": "n", "mg": "n",
            "vi": "v",
            "zg": "n", "none": "符号"}

def read_dict(path):
    word_dict = []
    word_freq = []
    word_flag = []
    with open(path) as file:
        line = file.readlines()
        for i in line:
            word = i.strip().split(' ')
            word_dict.append(word[0])
            word_freq.append(int(word[1]))
            word_flag.append(word[2])
            # if word[2] not in flags:
            # flags.append(word[2])
        # print(len(flagList), len(flags), flags)
    return word_dict, word_freq, word_flag

word_dict, word_freq, word_flag = read_dict(path)

def FMM(sentence):
    words = []
    flags = []
    freq = 0
    index = 0
    mismatch = 0
    sentence_len = len(sentence)
    size = maxSize
    if sentence_len < maxSize:
        size = sentence_len
    while index < sentence_len:
        match = False
        for i in range(size, 0, -1):
            # print(index, index + i, sentence[index: index + i])
            cur_str = sentence[index: index + i]
            if cur_str in word_dict:
                curIndex = word_dict.index(cur_str)
                words.append(cur_str)
                freq += word_freq[curIndex]
                flags.append(flagList[word_flag[curIndex]])
                index += i
                match = True
                break
        if not match:
            words.append(sentence[index])
            index += 1
            mismatch += 1
            flags.append("符号")
    return words, freq, mismatch, flags


def BMM(sentence):
    words = []
    flags = []
    freq = 0
    index = len(sentence)
    mismatch = 0
    size = maxSize
    if index < maxSize:
        size = index
    while index > 0:
        match = False
        for i in range(size, 0, -1):
            # print(index, index + i, sentence[index: index + i])
            cur_str = sentence[index - i: index]
            if cur_str in word_dict:
                curIndex = word_dict.index(cur_str)
                words.append(cur_str)
                freq += word_freq[curIndex]
                flags.append(flagList[word_flag[curIndex]])
                # print(cur_str, flagList[word_flag[curIndex]])
                index -= i
                match = True
                break
        if not match:
            # print(sentence[index - 1])
            words.append(sentence[index - 1])
            index -= 1
            mismatch += 1
            flags.append("符号")
    words.reverse()
    flags.reverse()
    return words, freq, mismatch, flags


def BiMM(s):
    words_FMM, freq_FMM, mismatch_FMM, flags_FMM = FMM(s)
    words_BMM, freq_BMM, mismatch_BMM, flags_BMM = BMM(s)
    if words_FMM == words_BMM:
        return words_FMM, flags_FMM
    socre_FMM = 0
    socre_BMM = 0
    # 词频越多越好
    if freq_FMM > freq_BMM:
        socre_FMM += 1
    elif freq_FMM < freq_BMM:
        socre_BMM += 1
    # 不匹配单词越少越好
    if mismatch_FMM < mismatch_BMM:
        socre_FMM += 1
    elif mismatch_FMM > mismatch_BMM:
        socre_BMM += 1
    # 总词数越少越好
    if len(words_FMM) < len(words_BMM):
        socre_FMM += 1
    elif len(words_FMM) > len(words_BMM):
        socre_BMM += 1
    # print(socre_FMM, socre_BMM)
    if socre_FMM > socre_BMM:
        return words_FMM, flags_FMM
    else:
        return words_BMM, flags_BMM


def cut(request):
    if request.method == "POST":
        try:
            req = json.loads(request.body)
            sentence = req["sentence"]
            sentenceList = sentence.split("。")
            words = []
            flags = []
            for s in sentenceList:
                w, f = BiMM(s)
                words.append(w)
                flags.append(f)
            return JsonResponse({'state': 0, 'words': words, 'flags': flags})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500})
    else:
        return JsonResponse({'state': 400})
            
