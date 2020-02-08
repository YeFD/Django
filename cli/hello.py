#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import time
import jieba


def get_time():
    return time.strftime('%Y-%m-%d',time.localtime(time.time()))


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


print("您好，欢迎来到 Cloud Studio")
print("当前时间是：" + get_time())
print("您的IP是：" + get_host_ip())
# pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple sklearn --user
# pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip --user
# django-admin.py startproject mysite
# python3 manage.py startapp nlp
# 2
# sklearn joblib jieba django
