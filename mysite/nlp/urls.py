from django.urls import path
from . import views
app_name = 'nlp'
urlpatterns = [
    path('', views.inputForm, name=''),  # 这里是默认首页
    # path('index/', views.index, name='index'),
    path('UploadFile/', views.UploadFile, name='UploadFile'),
    path('UploadText/', views.UploadText, name='UploadText'),
    # path('inputForm/', views.inputForm, name='inputForm'),
]
