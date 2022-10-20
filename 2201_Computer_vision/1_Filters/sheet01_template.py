import cv2 as cv
import numpy as np
import random
import time

# ********************TASK1***********************
def integral_image(img):
    # Your implementation of integral image
    pass


def sum_image(image):
    # Your implementation for summing up pixels
    pass


def task1():
    # Your implementation of Task1
    pass
# ************************************************
# ********************TASK2***********************
def equalize_hist_image(img):
    # Your implementation of histogram equalization
    pass


def task2():
    # Your implementation of Task2
    pass
# ************************************************
# ********************TASK4***********************
def get_kernel(sigma):
    # Your implementation of getGaussianKernel
    pass


def task4():
    # Your implementation of Task4
    pass
# ************************************************
# ********************TASK5***********************
def task5():
    # Your implementation of Task5
    pass
# ************************************************
# ********************TASK7***********************
def add_salt_n_pepper_noise(img):
    # Your implementation of adding noise to the image
    img_noise = np.copy(img)
    height, width = img_noise.shape
    for x in range(height):
        for y in range(width):
            p_noise = random.random()
            if p_noise < 0.3:
                p_white = random.random()
                if p_white < 0.5:
                    img_noise[x,y] = 255
                else:
                    img_noise[x,y] = 0
    return img_noise

def task7():
    # Your implementation of task 7
    img = cv.imread('bonn.png', cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_noise = add_salt_n_pepper_noise(img_gray)
    cv.imshow('Noised', img_noise)
    cv.waitKey(0)

    filter_sizes = [1,3,5,7,9]
    #a
    distance_min = 100000
    for size in filter_sizes:
        img_gaus_new = cv.GaussianBlur(img_noise,(size,size),0)
        distance = np.mean(np.abs(img_gray - img_gaus_new))
        if distance < distance_min:
            distance_min = distance
            img_gaus = img_gaus_new
    cv.imshow('Gaussian', img_gaus)
    cv.waitKey(0)

    #b
    distance_min = 100000
    for size in filter_sizes:
        img_median_new = cv.medianBlur(img_noise,size)
        distance = np.mean(np.abs(img_gray - img_median_new))
        if distance < distance_min:
            distance_min = distance
            img_median = img_median_new
    cv.imshow('Median', img_median)
    cv.waitKey(0)

    #c
    distance_min = 100000
    for size in filter_sizes:
        img_bilateral_new = cv.bilateralFilter(img_noise,size,80,80)
        distance = np.mean(np.abs(img_gray - img_bilateral_new))
        if distance <= distance_min:
            distance_min = distance
            img_bilateral = img_bilateral_new
    cv.imshow('Bilateral', img_bilateral)
    cv.waitKey(0)
    pass

task7()
# ************************************************
# ********************TASK8***********************
def task8():
    # Your implementation of task 8
    #a
    img = cv.imread('bonn.png', cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    K1 = np.array([[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]])
    K2 = np.array([[-0.8984, 0.1472, 1.1410], [-1.9075, 0.1566, 2.1359], [-0.8659, 0.0573, 1.0337]])
    img_k1 = cv.filter2D(img_gray,-1,K1)
    img_k2 = cv.filter2D(img_gray,-1,K2)
    cv.imshow('k1', img_k1)
    cv.waitKey(0)
    cv.imshow('k2', img_k2)
    cv.waitKey(0)

    #b
    w_k1, u_k1, vt_k1 = cv.SVDecomp(K1)
    w_k2, u_k2, vt_k2 = cv.SVDecomp(K2)
    img_k1_sep = cv.sepFilter2D(img_gray, -1, np.sqrt(w_k1[0,0])*vt_k1[0,:], np.sqrt(w_k1[0,0])*u_k1[:,0])
    cv.imshow('k1_sep', img_k1_sep)
    cv.waitKey(0)
    img_k2_sep = cv.sepFilter2D(img_gray, -1, np.sqrt(w_k2[0,0])*vt_k2[0,:], np.sqrt(w_k2[0,0])*u_k2[:,0])
    cv.imshow('k2_sep', img_k2_sep)
    cv.waitKey(0)

    #c
    error1 = np.max(cv.absdiff(img_k1,img_k1_sep))
    print(error1)
    error2 = np.max(cv.absdiff(img_k2,img_k2_sep))
    print(error2)
    pass
task8()
