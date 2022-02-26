## 评论情感分析
网页使用Django开发，本环境已安装Python3.5.2，jieba，sklearn，joblib等。

`pip3 install -r requirements.txt`

1. 请先在终端中输入`cd mysite`切换到mysite目录
2. 在终端输入`python3 manage.py runserver 0.0.0.0:8080`启动程序
3. ~~在右侧边栏【访问链接】创建访问链接，监听 `8080` 端口，点击链接访问。~~
    1. 点击右下角弹窗打卡预览窗口
    2. `cmd+shift+p`输入`preview.start`打开预览窗口
    3. 通过下方端口栏打开

uwsgi测试
```
uwsgi --http :8080 --chdir Python/mysite/ --module mysite.wsgi --static-map=/static=Python/mysite/nlp/static
```
uwsgi托管、维护
```
uwsgi --ini Python/mysite/mysite.ini
killall -9 uwsgi
uwsgi --reload master.pid
uwsgi --stop master.pid
```

## 
## 
## 
##
##
##
##
## 
## 
## 
## 
## 欢迎来到 Cloud Studio

这是一个展现 Cloud Studio 功能的 Python 示例。
当前环境（Maching Leaning）下安装了 python2 和 Python3。

##  Client Demo

### Python2 Demo
cli 目录下有可以获取当前时间与 IP 的代码，我们以此为例展示 python2 的使用。

1. 请先在终端中输入 `cd cli` 切换到 cli 示例目录。

2. 直接在下方终端中输入 `python hello.py` 查看运行效果吧。

### Python3 Demo

cli 目录下有贪吃蛇小游戏的代码，我们以此为例展示 python3 的使用。

1. 请先在终端中输入 `cd cli` 切换到 cli 示例目录，若已在该目录下请忽略。

2. 在终端中输入 `python3 snake.py` 开始玩贪吃蛇小游戏吧。

##  Web Demo

web 目录下有用 Python 编写的 Web 代码，我们以此为例展示如何用 Cloud Studio 进行一个网站的开发及预览。

1. 请先在终端中输入 `cd web` 切换到 web 示例目。

2. 在终端中运行 `cat requirements.txt | xargs sudo pip install -i http://pypi.douban.com/simple --trusted-host pypi.douban.com` 安装依赖。
    `pip3 install -i https://pypi.douban.com/simple statsmodels`

3. 在终端中运行 `python app.py` 启动程序。

4. 在右侧边栏【访问链接】创建访问链接，监听 `5000` 端口，如下图所示。

![图片](https://dn-coding-net-production-pp.codehub.cn/8872855a-e08c-4ba1-81fc-31e7a2f04bee.png)


5. 点击此访问链接即可查看网站！

## 其他语言支持

想要切换其他语言环境，可以在右边的【运行环境】中进行切换。

您也可以自己安装需要的环境，切换环境时会为您自动保存。


Happy Coding！
