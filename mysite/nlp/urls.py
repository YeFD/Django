from django.urls import path
from . import views
app_name = 'nlp'
urlpatterns = [
    path('', views.index, name=''),  # 这里是默认首页
    path('nlp', views.inputForm, name='nlp'),  # 这里是默认首页
    # path('index/', views.index, name='index'),
    path('UploadFile/', views.UploadFile, name='UploadFile'),
    path('UploadText/', views.UploadText, name='UploadText'),
    # path('inputForm/', views.inputForm, name='inputForm'),
    path('getPost/', views.getPost),
    path('forecast_es/', views.forecast_es),
    path('forecast_es_index/', views.forecast_es_index),
    path('forecast_arima_214/', views.forecast_arima_214),
    path('initModel_arima_010/', views.initModel_arima_010),
    path('forecast_arima_010/', views.forecast_arima_010),
    path('analyze/', views.analyze, name="analyze"),
    path('analyze_file/', views.analyze_file, name="analyze_file")
    ]
