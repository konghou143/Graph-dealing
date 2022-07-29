from django.shortcuts import render,HttpResponse

# Create your views here.

def index(request):
        return HttpResponse("欢迎使用")


def login(request):
    if request.method == "GET":
        return render(request, "login.html")
    else:
        username = request.POST.get("user")
        password = request.POST.get("pwd")
        if username == "root" and password == "123":
            return render(request, "start.html")
        else:
            return render(request, 'login.html', {"error_msg": "用户名或密码错误"})


def start(request):
    return render(request, "start.html")


def deal(request):
    if request.method == "GET":
        return render(request, "deal.html")
    else:
        file_object = request.FILES.get("avatar")
        f = open("app01/a1.png", mode='wb')

        for chunk in file_object.chunks():
            f.write(chunk)
        f.close()
        return render(request, "deal.html", {"error_msg": "提交完成，点击下方选项选择操作"})


def show(request):
    return render(request, "show.html")