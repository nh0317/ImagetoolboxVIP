from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
import qimage2ndarray
import numpy as np
import math
from numpy import pi, exp, sqrt

#가우시안 마스크
def gaussian(scale, sigma):
    mask = np.ones(shape=(scale, scale))
    sum = 0
    for i in range(0, scale):
        for j in range(0, scale):
            mask[i, j] = exp((-i * i - j * j) / (2 * sigma * sigma))
            mask[i, j] = mask[i, j] / (2 * 3.14 * sigma * sigma)
            sum += mask[i, j]
    mask = mask / sum
    return mask

#필터적용
def filter(mask, scale, image):
    # 패딩
    npad = int(scale / 2)
    pad = ((npad, npad), (npad, npad))
    image_padding = np.pad(image, pad, 'constant', constant_values=(128))

    #필터 적용
    x, y = image.shape
    filter_arr = np.zeros(shape=(x + 2 * npad, y + 2 * npad))
    for i in range(npad, x + npad):
        for j in range(npad, y + npad):
            for k in range(0, scale):
                for s in range(0, scale):
                    filter_arr[i, j] = filter_arr[i, j] + (mask[k, s]) * image_padding[i - npad + k, j - npad + s]

    #여백제거
    for i in range(1, npad+1):
        filter_arr=np.delete(filter_arr,0,axis=0)
        filter_arr = np.delete(filter_arr, 0, axis=1)

        m,n=filter_arr.shape
        filter_arr = np.delete(filter_arr, m-1, axis=0)
        filter_arr = np.delete(filter_arr, n-1, axis=1)

    return filter_arr

def EdgeDetection(image):
    #grayscale
    image_arr=qimage2ndarray.rgb_view(image)
    gray_weights=[0.2989, 0.5870, 0.1140]
    grayscale=np.dot(image_arr[...,:3],gray_weights)

    #가우시안 마스크 생성 및 적용
    scale = 5 #로딩시간이 오래걸린다면 scale=3으로 변경
    gaussian_mask = gaussian(scale, 1)
    gaussian_arr=filter(gaussian_mask,scale,grayscale)

    #라플라시안 마스크 및 적용
    laplacian_mask1=np.array([[-1,-1,-1],
                              [-1,8,-1],
                              [-1,-1,-1]])
    laplacian_mask2 = np.array([[0, 0, -1, 0, 0],
                                [0, -1, -2, -1, 0],
                                [-1, -2, 16, -2, -1],
                                [0, -1, -2, -1, 0],
                                [0, 0, -1, 0, 0]])

    #시간이 오래 걸린다면 laplacian_arr=filter(laplacian_mask1,3,gaussian_arr)으로 변경
    laplacian_arr=filter(laplacian_mask2,5,gaussian_arr)
    image = qimage2ndarray.array2qimage(laplacian_arr, normalize=False)
    qPixmapVar = QPixmap.fromImage(image)
    return qPixmapVar

def CornerDetection(image):
    #grayscale 변환
    image_arr = qimage2ndarray.rgb_view(image)
    gray_weights = [0.2989, 0.5870, 0.1140]
    grayscale = np.dot(image_arr[..., :3], gray_weights)

    # 윈도우 크기 결정
    scale = 3

    #코너 검출
    npad = int(scale / 2)
    image_padding = np.pad(grayscale, npad, 'constant', constant_values=(128))
    x, y = grayscale.shape
    corner_arr = image_arr.copy()
    for i in range(npad, x + npad):
        for j in range(npad, y + npad):
            M = image_padding[i - npad:i + npad + 1, j - npad:j + npad + 1]
            R = np.linalg.det(M) - 0.04 * (np.trace(M) * np.trace(M))
            if R > 0:
                corner_arr[i - npad, j - npad] = [255, 0, 0]

    image = qimage2ndarray.array2qimage(corner_arr, normalize=False)
    qPixmapVar = QPixmap.fromImage(image)
    return qPixmapVar

def Hough(image):
    # grayscale
    image_arr = qimage2ndarray.rgb_view(image)
    gray_weights = [0.2989, 0.5870, 0.1140]
    grayscale = np.dot(image_arr[..., :3], gray_weights)

    # 가우시안 마스크 생성 및 적용
    scale = 5  # 로딩시간이 오래걸린다면 scale=3으로 변경
    gaussian_mask = gaussian(scale, 0.7)
    gaussian_arr = filter(gaussian_mask, scale, grayscale)

    # 라플라시안 마스크 및 적용
    laplacian_mask1 = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])
    laplacian_mask2 = np.array([[0, 0, -1, 0, 0],
                                [0, -1, -2, -1, 0],
                                [-1, -2, 16, -2, -1],
                                [0, -1, -2, -1, 0],
                                [0, 0, -1, 0, 0]])

    # 시간이 오래 걸린다면 laplacian_arr=filter(laplacian_mask1,3,gaussian_arr)으로 변경
    laplacian_arr = filter(laplacian_mask2, 5, gaussian_arr)
    image_arr2 = qimage2ndarray.rgb_view(image)
    line=image_arr2.copy()

    print(grayscale)
    #허프영역 생성
    x, y = grayscale.shape
    t = math.atan(y/x)
    p = x * math.cos(t) + y * math.sin(t)
    p=int(p)
    hough = np.zeros(shape=(p,90))

    #허프 변환
    for i in range(1, x):
        for j in range(1, y):
           #엣지픽셀 검출
            if laplacian_arr[i,j] > 170:
                 theta=math.atan(j/i)
                 r=i*math.cos(theta) + j*math.sin(theta)
                 theta=int(theta*(180/3.14))
                 r=int(r)
                 hough[r,theta]+=1

    #라인검출
    for i in range(1, p):
        for j in range(1, 90):
            if hough[i,j] >2:
                m=i
                n=j
                for k in range(1, x):
                    for s in range(1, y):
                        theta = math.atan(s / k)
                        r = k * math.cos(theta) + s * math.sin(theta)
                        theta = int(theta * (180 / 3.14))
                        r = int(r)
                        if (r==m) and (theta==n):
                            line[k,s]=[255,0,0]

    image = qimage2ndarray.array2qimage(line, normalize=False)
    qPixmapVar = QPixmap.fromImage(image)
    return qPixmapVar