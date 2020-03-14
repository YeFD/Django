from django.shortcuts import render
from django.http import HttpResponse
# from django.shortcuts import HttpResponse
import joblib
import jieba
import jieba.analyse as analyse
import json
# Create your views here.
stopwords_path = r'/root/workspace/mysite/nlp/stopword.txt'
commentPath = r'/root/workspace/mysite/some/file/comment.txt'
stopwords = [line.strip() for line in open(
    stopwords_path, 'r', encoding='utf-8').readlines()]
TFIDF_model = joblib.load(r'/root/workspace/cli/TFIDF.model')
model = joblib.load(r'/root/workspace/cli/bayes.model')


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
        print(temp)
        content = json.dumps(temp)
        response = HttpResponse(content=content, content_type='application/json')
        return response


def inputForm(request):
    return render(request, "inputForm.html")


# cd mysite
# python3 manage.py runserver 0.0.0.0:8080
