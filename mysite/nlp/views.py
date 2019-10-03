from django.shortcuts import render
# from django.shortcuts import HttpResponse
import joblib
import jieba
# Create your views here.
scoreList = [60.9, 15.0]
stopwords_path = r'/home/coding/workspace/mysite/nlp/stopword.txt'
stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
TFIDF_model = joblib.load(r'/home/coding/workspace/cli/TFIDF.model')
model = joblib.load(r'/home/coding/workspace/cli/bayes.model')


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
    return predict_pro[0][0]


def index(request):
    if request.method == "POST":
        sentence = request.POST.get("sentence", None)
        print(sentence)
        score = getScore(sentence)
        scoreList.append(score)
    return render(request, "index.html", {"data": scoreList})


# cd mysite
# python3 manage.py runserver 0.0.0.0:8080
