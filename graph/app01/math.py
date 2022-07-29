from django.shortcuts import render, HttpResponse
import cv2
import numpy as np

def sztx(request):
    if request.method == "GET":
        return render(request, "sztx.html")
    else:
        mtd = request.POST.get("mtd")

        if mtd == "腐蚀":
            pic = 'app01/a1.png'
            src = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            erosion = cv2.erode(src, kernel)
            cv2.imwrite("app01/static/erosion.jpg", erosion)
            print(np.sum(erosion))
        elif mtd == "膨胀":
            pic = 'app01/a1.png'
            src = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            dilation = cv2.dilate(src, kernel)
            cv2.imwrite("app01/static/erosion.jpg", dilation)
            print(np.sum(dilation))
        elif mtd == "开运算":
            pic = 'app01/a1.png'
            src = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            open = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
            cv2.imwrite("app01/static/erosion.jpg", open)
            print(np.sum(open))
        elif mtd == "闭运算":
            pic = 'app01/a1.png'
            src = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
            close = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite("app01/static/erosion.jpg", close)
            print(np.sum(close))
        else:
            return render(request, "sztx.html")
    return render(request, "show.html")


def txbh(request):
    if request.method == "GET":
        return render(request, "txbh.html")
    else:
        mtd = request.POST.get("mtd")

        if mtd == "水平翻转":
            pic = 'app01/a1.png'
            src = cv2.imread(pic)
            horizontal = cv2.flip(src, 1, dst=None)
            cv2.imwrite("app01/static/erosion.jpg", horizontal)
        elif mtd == "垂直翻转":
            pic = 'app01/a1.png'
            src = cv2.imread(pic)
            horizontal = cv2.flip(src, 0, dst=None)
            cv2.imwrite("app01/static/erosion.jpg", horizontal)
        elif mtd == "垂直翻转":
            pic = 'app01/a1.png'
            src = cv2.imread(pic)
            horizontal = cv2.flip(src, -1, dst=None)
            cv2.imwrite("app01/static/erosion.jpg", horizontal)
        elif mtd == "垂直翻转":
            pic = 'app01/a1.png'
            src = cv2.imread(pic)
            horizontal = cv2.flip(src, -1, dst=None)
            cv2.imwrite("app01/static/erosion.jpg", horizontal)
        elif mtd == "缩放":
            pic = 'app01/a1.png'
            x = request.POST.get("cs1")
            y = request.POST.get("cs2")
            src = cv2.imread(pic)
            src = cv2.resize(src, (0, 0), fx=int(x), fy=int(y), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite("app01/static/erosion.jpg", src)
        elif mtd == "平移":
            pic = 'app01/a1.png'
            x = request.POST.get("cs1")
            y = request.POST.get("cs2")
            src = cv2.imread(pic)
            height, width, channel = src.shape
            M = np.float32([[1, 0, int(x)], [0, 1, int(y)]])
            src = cv2.warpAffine(src, M, (width, height))
            cv2.imwrite("app01/static/erosion.jpg", src)
        elif mtd == "中心旋转":
            pic = 'app01/a1.png'
            x = request.POST.get("cs1")
            src = cv2.imread(pic)
            height, width, channel = src.shape
            M = cv2.getRotationMatrix2D((width / 2, height / 2), int(x), 1)
            src = cv2.warpAffine(src, M, (width, height))
            cv2.imwrite("app01/static/erosion.jpg", src)
        elif mtd == "仿射变换":
            pic = 'app01/a1.png'
            src = cv2.imread(pic)
            rows, cols = src.shape[: 2]
            post1 = np.float32([[50, 50], [200, 50], [50, 200]])
            post2 = np.float32([[10, 100], [200, 50], [100, 250]])
            M = cv2.getAffineTransform(post1, post2)
            src = cv2.warpAffine(src, M, (rows, cols))
            cv2.imwrite("app01/static/erosion.jpg", src)
        else:
            return render(request, "txbh.html")
    return render(request, "show.html")


def txms(request):
    if request.method == "GET":
        return render(request, "txms.html")
    else:
        mtd = request.POST.get("mtd")
        cs = request.POST.get("cs")
        if mtd == "HSV色彩空间":
            pic = cv2.imread('app01/a1.png')
            hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            if cs == "h":
                cv2.imwrite("app01/static/erosion.jpg", h)
            elif cs == "s":
                cv2.imwrite("app01/static/erosion.jpg", s)
            elif cs == "v":
                cv2.imwrite("app01/static/erosion.jpg", v)
        elif mtd == "RGB色彩空间":
            pic = cv2.imread('app01/a1.png')
            b = pic[:, :, 0]
            g = pic[:, :, 1]
            r = pic[:, :, 2]
            if cs == "b":
                cv2.imwrite("app01/static/erosion.jpg", b)
            elif cs == "g":
                cv2.imwrite("app01/static/erosion.jpg", g)
            elif cs == "r":
                cv2.imwrite("app01/static/erosion.jpg", r)
        else:
            return render(request, "txms.html")
    return render(request, "show.html")


def byjc(request):
    if request.method == "GET":
        return render(request, "byjc.html")
    else:
        mtd = request.POST.get("mtd")
        if mtd == "1":
            pic = 'app01/a1.png'
            src = cv2.imread(pic)
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            image = cv2.copyMakeBorder(src, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
            image = cv2.GaussianBlur(image, (3, 3), 0, 0)
            m1 = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
            rows = image.shape[0]
            cols = image.shape[1]
            image1 = np.zeros(image.shape)
            for k in range(0, 2):
                for i in range(2, rows - 2):
                    for j in range(2, cols - 2):
                        image1[i, j] = np.sum((m1 * image[i - 2:i + 3, j - 2:j + 3, k]))
            image1 = cv2.convertScaleAbs(image1)

            cv2.imwrite("app01/static/erosion.jpg", image1)
        else:
            pic = 'app01/a1.png'
            src = cv2.imread(pic)
            greyImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
            kernely = np.array([[0, -1], [1, 0]], dtype=int)
            x = cv2.filter2D(greyImage, cv2.CV_16S, kernelx)
            y = cv2.filter2D(greyImage, cv2.CV_16S, kernely)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            cv2.imwrite("app01/static/erosion.jpg", Roberts)
        return render(request, "show.html")

def txzq(request):
    if request.method == "GET":
        return render(request, "txzq.html")
    else:
        mtd = request.POST.get("mtd")
        mtd2 = request.POST.get("mtd2")
        if mtd == "空域" and mtd2 == "平滑":
            pic = 'app01/1.png'
            src = cv2.imread(pic)
            cv2.imwrite("app01/static/erosion.jpg", src)
        if mtd == "空域" and mtd2 == "锐化":
            pic = 'app01/2.png'
            src = cv2.imread(pic)
            cv2.imwrite("app01/static/erosion.jpg", src)
        if mtd == "频域" and mtd2 == "平滑":
            pic = 'app01/3.png'
            src = cv2.imread(pic)
            cv2.imwrite("app01/static/erosion.jpg", src)
        if mtd == "频域" and mtd2 == "锐化":
            pic = 'app01/4.png'
            src = cv2.imread(pic)
            cv2.imwrite("app01/static/erosion.jpg", src)
    return render(request, "show.html")


def zscl(request):
    if request.method == "GET":
        return render(request, "zscl.html")
    else:
        mtd = request.POST.get("mtd")
        if mtd == "添加噪声":
            pic = 'app01/a1.png'
            image = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
            output = np.zeros(image.shape, np.uint8)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if image[i][j] < 100:
                        output[i][j] = 255
                    elif image[i][j] > 200:
                        output[i][j] = 0
                    else:
                        output[i][j] = image[i][j]
            cv2.imwrite("app01/static/erosion.jpg", output)
        elif mtd == "消除噪声":
            pic = 'app01/a1.png'
            image = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
            output = np.zeros(image.shape, np.uint8)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    ji = 1.0
                    for m in range(0, 1):
                        for n in range(-1, 2):
                            if 0<= i + m < image.shape[0] and 0 <= j + n < image.shape[1]:
                                ji *= image[i + m][j + n]
                    output[i][j] = pow(ji, 1/3)
            cv2.imwrite("app01/static/erosion.jpg", output)
        else:
            return render(request, "zscl.html")
    return render(request, "show.html")

