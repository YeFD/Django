from django.urls import path
from . import views
app_name = 'nlp'
urlpatterns = [
    path('', views.inputForm, name=''),  # 这里是默认首页
    # path('index/', views.index, name='index'),
    path('UploadFile/', views.UploadFile, name='UploadFile'),
    path('UploadText/', views.UploadText, name='UploadText'),
    # path('inputForm/', views.inputForm, name='inputForm'),
    path('api/', views.getPost),
    path('forecastEs/', views.forecast_es),
    path('forecastEsIndex/', views.forecast_es_index),
    path('forecastArima214/', views.forecast_arima_214),
    path('initModelArima010/', views.initModel_arima_010),
    path('forecastArima010/', views.forecast_arima_010)
    ]
