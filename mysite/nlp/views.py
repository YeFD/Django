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

def index(request):
    return render(request, "index.html")

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

flagList = {"n": "n", "f": "n", "s": "n", "t": "n",
            "nr": "n", "ns": "n", "nt": "n", "nw": "n",
            "nz": "n", "v": "v", "vd": "v", "vn": "vn",
            "a": "adj", "ad": "adj", "an": "adj", "d": "adv",
            "m": "adj", "q": "adj", "r": "r", "p": "p",
            "c": "连词", "u": "助词", "xc": "助词", "w": "标点符号",
            "uj": "助词", "ul": "助词", "uz": "助词", "uv": "助词",
            "ud": "助词", "ug": "助词", "e": "语气助词", "k": "n",
            "b": "adj", "yw": "yw", "j": "n", "ng": "n",
            "h": "adj", "df": "v", "z": "n", "l": "n",
            "vg": "v", "tg": "n", "nrt": "n", "rz": "r",
            "nrfg": "n", "ag": "adj", "g": "n", "mq": "r",
            "x": "n", "dg": "n", "i": "n", "o": "adv",
            "rr": "r", "vq": "v", "rg": "n", "mg": "n", "etc": "etc",
            "vi": "v", "qs": "qs", "gt": "gt", "bei": "bei", "ba": "ba",
            "zg": "n", "none": "符号", "de": "de", "le": "le", "sf": "sf",
            "di": "di", "cc": "cc", "#": "#"}
flagList2 = {
    "n": "名词", "r": "代词", "adj": "形容词", "de": "的", "adv": "副词", "sf": "疑问词", "etc": "省略词",
    "di": "地", "cc": "并列连词", "ba": "把词", "bei": "被词", "qs": "祈使词", "vn": "名词",
    "gt": "感叹词", "v": "动词", "p": "介词", "yw": "疑问词", "#": "结束", "le": "助词", "符号": "符号"
}
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


def BiMM_extra(s, extra_word, extra_flag):
    if extra_word in s:
        s_list = s.split(extra_word)
        words, flags = BiMM(s_list[0])
        words.append(extra_word)
        flags.append(extra_flag)
        t1, t2 = BiMM(s_list[1])
        words += t1
        flags += t2
        return words, flags
    else:
        return BiMM(s)


class Sentence:
    words = []
    flags = []
    token = ""
    index = 0
    type = ""
    result = []
    error = []
    sf_flag = False

    def __init__(self):
        pass

    def analyze(self, words, flags):
        self.words = words
        self.flags = flags
        for i in range(1, len(words)):
            if words[i] == "的":
                self.flags[i - 1] = "adj"
            elif words[i] == "地":
                self.flags[i - 1] = "adv"
        print(self.words, self.flags)
        self.index = 0
        self.error = []
        self.result = []
        self.type = ""
        self.sf_flag = False
        self.token = self.getToken(self.index)
        self.index += 1
        if "yw" in flags or "sf" in flags:
            self.type = "疑问句"
            self.questions()  # 疑问句
        elif "qs" == flags[0]:
            self.type = "祈使句"
            self.imperative()  # 祈使句
        else:
            self.declarative_sentence()  # 陈述句
        if len(self.result) < len(self.words) and len(self.error) == 0:
            self.error.append("含多余成分")
        return self.result, self.error, self.type

    def questions(self):
        # 疑问句
        if "yw" in self.flags:
            self.declarative_sentence()
            self.Y()
        elif "sf" in self.flags:
            # self.SF()
            self.sf_flag = True
            self.declarative_sentence()
        if self.token[0] == "?" or self.token[0] == "？":
            self.result.append("符号")
            self.match("?")

    def imperative(self):
        self.QS()
        self.declarative_sentence()
        self.GT()

    def declarative_sentence(self):
        # 陈述句
        if "ba" in self.flags:
            if self.type == "":
                self.type = "把式陈述句"
            self.sentence_ba()
        elif "bei" in self.flags:
            if self.type == "":
                self.type = "被动陈述句"
            self.sentence_bei()
        else:
            if self.type == "":
                self.type = "普通陈述句"
            self.sentence_normal()

    def sentence_ba(self):
        self.Attributive()  # 定语
        self.Subject()  # 主语
        self.SF()
        self.BA()
        self.Attributive()  # 定语
        self.Object()  # 宾语
        self.Adverbial()  # 状语
        self.Predicate()  # 谓语
        self.Complement()  # 补语

    def sentence_bei(self):
        self.Attributive()  # 定语
        self.Object()  # 宾语
        self.SF()
        self.BEI()
        self.Attributive()  # 定语
        self.Subject()  # 主语
        self.Adverbial()  # 状语
        self.Predicate()  # 谓语
        self.Complement()  # 补语

    def sentence_normal(self):
        self.Attributive()  # 定语
        self.Subject()  # 主语
        self.SF()
        self.Adverbial()  # 状语
        self.Predicate()  # 谓语
        self.Complement()  # 补语
        self.Attributive()  # 定语
        self.Object()  # 宾语

    def getToken(self, i):
        if i < len(self.words):
            # result = (words[i], flags[i])
            return self.words[i], self.flags[i]
        else:
            return "#", "#"

    def match(self, type):
        if self.index > 0:
            pass
            # print(self.token)
        self.token = self.getToken(self.index)
        self.index += 1
        # return self.token

    def BA(self):
        if self.token[1] == "ba":
            self.match("ba")
            self.result.append("把词")
        else:
            print("不是把")

    def Y(self):
        if self.token[1] == "yw":
            self.match("yw")
            self.result.append("疑问词")
        else:
            print("not yw")

    def SF(self):
        if self.token[1] == "sf":
            self.match("sf")
            self.result.append("疑问词")
        else:
            print("not sf")

    def QS(self):
        if self.token[1] == "qs":
            self.match("qs")
            self.result.append("祈使词")
        else:
            print("not qs")

    def GT(self):
        if self.token[1] == "gt":
            self.match("gt")
            self.result.append("助词")
        else:
            print("no gt")

    def BEI(self):
        if self.token[1] == "bei":
            self.match("bei")
            self.result.append("被词")
        else:
            print("not bei")

    def Attributive(self):
        # 定语
        if self.token[1] == "adj":
            self.ADJP()
            # print("====定语")

    def ADJP(self):
        while self.token[1] == "adj":
            self.ADJ()

    def ADJ(self):
        self.match("adj")
        self.result.append("定语")
        if self.token[1] == "de":
            self.match("de")
            self.result.append("定语")

    def Subject(self):
        # 主语
        if self.token[1] == "n" or self.token[1] == "vn" \
                or self.token[1] == "r":
            self.NP()
            # print("====主语")
            # print("====主语")
        else:
            self.error.append("缺少主语")
            print("缺少主语")

    def NP(self):
        self.NN()
        while self.token[1] == "cc":
            self.match("cc")
            self.result.append("主语")
            self.NN()
        if self.token[1] == "etc":
            self.match("etc")
            self.result.append("主语")

    def NN(self):
        while self.token[1] == "n" or self.token[1] == "vn" \
                or self.token[1] == "r":
            self.match("n")
            self.result.append("主语")

    def Adverbial(self):
        # 状语
        if self.token[1] == "adv":
            self.ADVP()
            # print("====状语")

    def ADVP(self):
        while self.token[1] == "adv":
            self.ADV()

    def ADV(self):
        self.match("adv")
        self.result.append("状语")
        if self.token[1] == "di":
            self.match("di")
            self.result.append("状语")

    def Predicate(self):
        # 谓语
        if self.token[1] == "v" or self.token[1] == "vn":
            self.VP()
            # print("====谓语")
        else:
            self.error.append("缺少谓语")
            print("缺少谓语")

    def VP(self):
        while self.token[1] == "v" or self.token[1] == "vn":
            self.match("v")
            self.result.append("谓语")

    def Complement(self):
        # 补语
        if self.token[1] == "le":
            self.match("le")
            self.result.append("补语")
            # print("====补语")

    def Object(self):
        # 宾语
        if self.token[1] == "n" or self.token[1] == "vn" \
                or self.token[1] == "p":
            self.P()
            self.NP2()
            # print("====宾语")
        elif self.token[1] == "r":
            self.match("r")
            self.result.append("宾语")
            # print("====宾语")
        else:
            self.error.append("缺少宾语")
            print("缺少宾语")

    def P(self):
        if self.token[1] == "p":
            self.match("p")
            self.result.append("宾语")

    def NP2(self):
        self.NN2()
        while self.token[1] == "cc":
            self.match("cc")
            self.result.append("宾语")
            self.NN2()
        if self.token[1] == "etc":
            self.match("etc")
            self.result.append("宾语")

    def NN2(self):
        while self.token[1] == "n" or self.token[1] == "vn":
            self.match("n")
            self.result.append("宾语")


def getFlag(flags):
    flags2 = []
    for flag in flags:
        if flag not in flagList2.keys():
            flags2.append("")
            print(flag)
        else:
            flags2.append(flagList2[flag])
    return flags2

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
            
def analyze(request):
    if request.method == "POST":
        try:
            s = request.POST.get("sentence", None)
            extra_word = request.POST.get("extra_word", None)
            extra_flag = request.POST.get("extra_flag", None)
            # req = json.loads(request.body)
            # sentence = req["sentence"]
            # sentenceList = sentence.split("。")
            # print(extra_flag)
            if len(extra_word) == 0:
                words, flags = BiMM(s)
            else:
                words, flags = BiMM_extra(s, extra_word, extra_flag)
            # print(words, flags)
            sentence = Sentence()
            result, error, type = sentence.analyze(words, flags)
            # print(result, error, type)
            flags2 = getFlag(flags)
            return JsonResponse({'state': 0, 'words': words, 'flags': flags, 'result': result, 'error': error, 'flags2': flags2, 'type': type})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500})
    else:
        return JsonResponse({'state': 400})

def analyze_file(request):
    if request.method == "POST":
        try:
            f = request.FILES['TxtFile']
            s = f.read()
            s = s.decode("utf-8")
            words, flags = BiMM(s)
            sentence = Sentence()
            result, error, type = sentence.analyze(words, flags)
            # print(result, error, type)
            flags2 = getFlag(flags)
            return JsonResponse({'state': 0, 'words': words, 'flags': flags, 'result': result, 'error': error, 'flags2': flags2, 'type': type})
        except Exception as e:
            print(e)
            return JsonResponse({'state': 500})
    else:
        return JsonResponse({'state': 400})