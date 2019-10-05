from django.urls import path
from . import views
app_name = 'nlp'
urlpatterns = [
    path('', views.index, name=''),  # 这里是默认首页
    path('index/', views.index, name='index'),
    path('UploadFile/', views.UploadFile, name='UploadFile'),
]
