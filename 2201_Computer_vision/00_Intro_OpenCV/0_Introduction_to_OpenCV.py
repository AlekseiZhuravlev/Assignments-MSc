# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 19:06:21 2022

@author: Алексей
"""
import random

import cv2 as cv
import numpy as np
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png'

    # 2a: read and display the image 
    img = cv.imread(img_path)
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    img_cpy = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(len(img[:, 0])):
        for j in range(len(img[0, :])):
            img_cpy[i, j] = [
                max(img[i, j][0] - 0.5 * img_gray[i, j], 0) / 255,
                max(img[i, j][1] - 0.5 * img_gray[i, j], 0) / 255,
                max(img[i, j][2] - 0.5 * img_gray[i, j], 0) / 255
            ]
    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perform the operation above
    img_cpy_2 = np.clip(img - 0.5 * np.dstack((img_gray, img_gray, img_gray)),
                        0, 255) / 255
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy_2)

    # 2e: Extract the center patch and place randomly in the image
    patch_size = 16
    img_patch = img[
                (img.shape[0] - patch_size) // 2: (img.shape[0] + patch_size) // 2,
                (img.shape[1] - patch_size) // 2: (img.shape[1] + patch_size) // 2,
                :
                ]
    display_image('2 - e - Center Patch', img_patch)

    # Random location of the patch for placement
    rand_coord = (
        np.random.randint(0, img.shape[0] - patch_size),
        np.random.randint(0, img.shape[1] - patch_size)
    )
    img_cpy = img.copy()
    img_cpy[
    rand_coord[0]:rand_coord[0] + patch_size,
    rand_coord[1]:rand_coord[1] + patch_size,
    ] = img_patch
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)

    # 2f: Draw random rectangles and ellipses
    img_cpy = img.copy()

    # rectangles
    for i in range(10):
        vertex_1 = [
            random.randint(0, img.shape[1] - 30),
            random.randint(0, img.shape[0] - 30),
        ]
        vertex_2 = [
            random.randint(vertex_1[0], img.shape[1]),
            random.randint(vertex_1[1], img.shape[0]),
        ]
        color = random.sample(range(256), 3)
        cv.rectangle(img_cpy, vertex_1, vertex_2, color, thickness=2)

    # ellipses
    for i in range(10):
        center = [
            random.randint(20, img.shape[1] - 30),
            random.randint(20, img.shape[0] - 30),
        ]
        axes_length = [
            random.randint(min(10, center[0]), min(img.shape[1] - center[0], center[0])),
            random.randint(min(10, center[1]), min(img.shape[0] - center[1], center[1])),
        ]
        color = random.sample(range(256), 3)
        cv.ellipse(img_cpy, center, axes_length, 0, 0, 360, color, thickness=2)

    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
