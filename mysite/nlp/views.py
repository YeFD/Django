from django.shortcuts import render
from django.http import HttpResponse
# from django.shortcuts import HttpResponse
import joblib
import jieba
# Create your views here.
stopwords_path = r'/home/coding/workspace/mysite/nlp/stopword.txt'
commentPath = r'/home/coding/workspace/mysite/some/file/comment.txt'
stopwords = [line.strip() for line in open(
    stopwords_path, 'r', encoding='utf-8').readlines()]
TFIDF_model = joblib.load(r'/root/coding/workspace/cli/TFIDF.model')
model = joblib.load(r'/root/coding/workspace/cli/bayes.model')


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


def getScore(sentence):
    sentence_list = []
    sentence_list.append(sentence)
    sentence_cut = cut_words(sentence_list)
    data_TFIDF = TFIDF_model.transform(sentence_cut)

    predict_pro = model.predict_proba(data_TFIDF)
    print(predict_pro)
    return predict_pro[0][1]


def index(request):
    scoreList = []
    if request.method == "POST":
        sentence = request.POST.get("sentence", None)
        # print(sentence)
        score = getScore(sentence)
        score = '%.0f' % (score * 100)
        scoreList.append(score)
        # print(scoreList)
        # print(getStar(sentence))
    return render(request, "index.html", {"data": scoreList})


def UploadFile(request):
    if request.method == "POST":
        f = request.FILES['TxtFile']
        sentence = f.read()
        sentence = sentence.decode("utf-8")
        sentenceList = sentence.split('\n')
        List = getScoreList(sentenceList)
        sum = 0.0
        for score in List:
            sum += score * 5
        Star = round(sum / len(List), 1)
        response = HttpResponse()
        response.content = Star  # 这里返回满分为五分的评分
        return response
    # return render(request, "UploadFile.html", {"data": "0"})
    # from django import forms
    # proList = []
    # if request.method == "POST":
    #     f = request.FILES['TxtFile']
    #     sentence = f.read()
    #     sentence = sentence.decode("utf-8")
    #     sentenceList = sentence.split('\n')
    #     List = getScoreList(sentenceList)
    #     # print(List)
    #     for score in List:
    #         score = '%.0f' % (score * 100)
    #         proList.append(score)
    #     # print(proList)
    #     # with open('some/file/comment.txt', 'wb+') as destination:
    #     #     for chunk in f.chunks():
    #     #         destination.write(chunk)

    #     # sentence = open(obj).read()
    #     # score = getScore(sentence)
    #     # scoreList.append(score)
    #     # scoreList = txtScore()

    #     # proList.append(score)
    #     # print(proList)
    #     # Lists = txtScore()
    #     # for List in Lists:
    #     #     proList.append(List[0])
    #     # print(proList)
    # return render(request, "UploadFile.html", {"data": proList})


def txtScore():
    sentenceList = [line.strip() for line in open(
        commentPath, 'r', encoding='utf-8').readlines()]
    sentenceCut = cut_words(sentenceList)
    dataTFIDF = TFIDF_model.transform(sentenceCut)
    predictPro = model.predict_proba(dataTFIDF)
    # print(predictPro)
    return predictPro


def getScoreList(sentenceList):
    sentenceCut = cut_words(sentenceList)
    dataTFIDF = TFIDF_model.transform(sentenceCut)
    predictPro = model.predict_proba(dataTFIDF)
    temp = []
    for predict in predictPro:
        temp.append(predict[1])
    # print(temp)
    return temp


def UploadText(request):
    if request.method == "POST":
        sentence = request.POST.get("sentence", None)
        score = getStar(sentence)
        response = HttpResponse()
        response.content = score
        return response


def inputForm(request):
    # score = 0
    # if request.method == "POST":
    #     sentence = request.POST.get("sentence", None)
    #     # print(sentence)
    #     #sentence = sentence.decode("utf-8")
    #     score = getStar(sentence)
    #     #score = '%.0f' % (score * 100)
    #     # scoreList.append(score)
    #     # print(scoreList)
    return render(request, "inputForm.html")


def getStar(sentence):
    sentence_list = []
    sentence_list.append(sentence)
    sentence_cut = cut_words(sentence_list)
    data_TFIDF = TFIDF_model.transform(sentence_cut)

    predict_pro = model.predict_proba(data_TFIDF)
    score = '%.1f' % (predict_pro[0][1] * 5)
    # print(score)
    return score


# cd mysite
# python3 manage.py runserver 0.0.0.0:8080
