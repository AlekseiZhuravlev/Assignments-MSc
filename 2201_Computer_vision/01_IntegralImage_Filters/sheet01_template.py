import random
import time

import cv2 as cv
import numpy as np


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def read_image(grayscale=False):
    img_path = 'bonn.png'

    img = cv.imread(img_path)
    if grayscale:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        return img


# ********************TASK1***********************

def get_pixel_intensity(img, i, j):
    if 0 <= i < img.shape[0] and 0 <= j < img.shape[1]:
        return img[i, j]
    else:
        return 0


def integral_image(img):
    # Your implementation of integral image
    int_image = np.zeros(img.shape[0:2], dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            int_image[i, j] = \
                get_pixel_intensity(int_image, i - 1, j) + get_pixel_intensity(int_image, i, j - 1) - \
                get_pixel_intensity(int_image, i - 1, j - 1) + get_pixel_intensity(img, i, j)
    return int_image


def sum_image(image):
    # Your implementation for summing up pixels
    intensity = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity += image[i, j]
    return intensity


def task1():
    # Your implementation of Task1
    img_gray = read_image(grayscale=True)

    int_image = cv.integral(img_gray)
    print("Integral image using opencv:\n", int_image)

    int_image_own = integral_image(img_gray)
    print("Integral image using own implementation:\n", int_image_own)
    print("Checking equality:\n", int_image[1:, 1:] == int_image_own, '\n')

    print('Calculating average intensity for 10 random rectangles:')
    for _ in range(10):
        x_rand = random.randint(0, img_gray.shape[0] - 100)
        y_rand = random.randint(0, img_gray.shape[1] - 100)

        time_start = time.time()
        avg_summing = sum_image(img_gray[x_rand:x_rand + 100, y_rand:y_rand + 100]) / 10000
        print(f'By summing the pixels: {avg_summing}, {time.time() - time_start:.5f}s')

        time_start = time.time()
        avg_integral = cv.integral(img_gray[x_rand:x_rand + 100, y_rand:y_rand + 100])[-1, -1] / 10000
        print(f'By using integral function: {avg_integral}, {time.time() - time_start:.5f}s')

        time_start = time.time()
        avg_integral_own = integral_image(img_gray[x_rand:x_rand + 100, y_rand:y_rand + 100])[-1, -1] / 10000
        print(f'By using own integral function: {avg_integral_own}, {time.time() - time_start:.5f}s\n')


# ************************************************
# ********************TASK2***********************
def equalize_hist_image(img):
    # Your implementation of histogram equalization

    # image histogram
    hist_img = np.zeros(256, dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist_img[
                img[i, j]
            ] += 1

    # calculating cumulative distribution function
    cdf = hist_img.copy()
    for i in range(len(hist_img) - 1):
        cdf[i + 1] += cdf[i]
    cdf = (cdf / max(cdf) * 255).astype(np.uint8)

    # equalizing the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = cdf[img[i, j]]
    return img


def get_maximum_pixel_difference(img_1, img_2):
    max_abs = 0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            difference = abs(int(img_1[i, j]) - int(img_2[i, j]))
            if difference > max_abs:
                max_abs = difference
    return max_abs


def task2():
    # Your implementation of Task2
    img_gray = read_image(grayscale=True)

    equalized = cv.equalizeHist(img_gray.copy())
    equalized_own = equalize_hist_image(img_gray.copy())

    display_image('Source image', img_gray)
    display_image('Equalized Image, function', equalized)
    display_image('Equalized Image, own implementation', equalized_own)

    print('maximum pixel difference:',
          get_maximum_pixel_difference(equalized_own, equalized)
          )


# ************************************************
# ********************TASK4***********************
def get_kernel(sigma):
    # Your implementation of getGaussianKernel
    kernel_1d = get_kernel_1d(sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / np.sum(kernel_2d)


def get_kernel_1d(sigma):
    length = 2 * np.ceil(sigma * 3).astype(int)
    ax = np.linspace(
        -(length - 1) / 2.,
        (length - 1) / 2.,
        length
    )
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel / np.sum(kernel)


def task4():
    # Your implementation of Task4
    img_gray = read_image(grayscale=True)

    sigma = 2 * np.sqrt(2)
    img_blur = cv.GaussianBlur(img_gray.copy(), ksize=[0, 0], sigmaX=sigma)
    display_image('blurred GaussianBlur', img_blur)

    img_kernel = cv.filter2D(img_gray.copy(), -1, get_kernel(sigma))
    display_image('blurred filter2D', img_kernel)

    img_sep_kernel = cv.sepFilter2D(
        img_gray.copy(),
        -1,
        get_kernel_1d(sigma),
        get_kernel_1d(sigma),
    )
    display_image('blurred sepFilter2D', img_sep_kernel)

    print('Max difference:')
    print('GaussianBlur - filter2D', get_maximum_pixel_difference(img_blur, img_kernel))
    print('GaussianBlur - sepFilter2D', get_maximum_pixel_difference(img_blur, img_sep_kernel))
    print('filter2D - sepFilter2D', get_maximum_pixel_difference(img_kernel, img_sep_kernel))


# ************************************************
# ********************TASK5***********************
def task5():
    # Your implementation of Task5
    img_gray = read_image(grayscale=True)
    display_image('Original image', img_gray)

    sigma_1 = 2
    img_blur_1 = cv.GaussianBlur(img_gray.copy(), ksize=[0, 0], sigmaX=sigma_1)
    img_blur_2 = cv.GaussianBlur(img_blur_1.copy(), ksize=[0, 0], sigmaX=sigma_1)

    sigma_2 = 2 * np.sqrt(2)
    img_blur_3 = cv.GaussianBlur(img_gray.copy(), ksize=[0, 0], sigmaX=sigma_2)

    display_image('Gaussian blur sigma=2, 2 times', img_blur_2)
    display_image('Gaussian blur sigma=2*sqrt(2), 1 time', img_blur_3)

    print('Max difference:', get_maximum_pixel_difference(img_blur_2, img_blur_3))


# ************************************************
# ********************TASK7***********************
def get_mean_pixel_difference(img_1, img_2):
    diff_sum = 0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            diff_sum += abs(int(img_1[i, j]) - int(img_2[i, j]))
    return diff_sum / img_1.size


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
                    img_noise[x, y] = 255
                else:
                    img_noise[x, y] = 0
    return img_noise


def task7():
    # Your implementation of task 7
    img = cv.imread('bonn.png', cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_noise = add_salt_n_pepper_noise(img_gray)
    display_image('Noised', img_noise)

    filter_sizes = [1, 3, 5, 7, 9]
    # a
    distance_min = 100000
    for size in filter_sizes:
        img_gaus_new = cv.GaussianBlur(img_noise, (size, size), 0)
        distance = get_mean_pixel_difference(img_gray, img_gaus_new)
        # np.mean(np.abs(img_gray - img_gaus_new))
        if distance < distance_min:
            distance_min = distance
            img_gaus = img_gaus_new

    print(f'min difference, gaussian: {distance}')
    display_image('Gaussian', img_gaus)

    # b
    distance_min = 100000
    for size in filter_sizes:
        img_median_new = cv.medianBlur(img_noise, size)
        distance = get_mean_pixel_difference(img_gray, img_median_new)
        # np.mean(np.abs(img_gray - img_median_new))
        if distance < distance_min:
            distance_min = distance
            img_median = img_median_new

    print(f'min difference, median: {distance}')
    display_image('Median', img_median)

    # c
    distance_min = 100000
    for size in filter_sizes:
        img_bilateral_new = cv.bilateralFilter(img_noise, size, 80, 80)
        distance = get_mean_pixel_difference(img_gray, img_bilateral_new)
        # np.mean(np.abs(img_gray - img_bilateral_new))
        if distance <= distance_min:
            distance_min = distance
            img_bilateral = img_bilateral_new

    print(f'min difference, bilateral: {distance}')
    display_image('Bilateral', img_bilateral)

    pass


# ************************************************
# ********************TASK8***********************
def task8():
    # Your implementation of task 8
    # a
    img = cv.imread('bonn.png', cv.IMREAD_COLOR)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    K1 = np.array([[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]])
    K2 = np.array([[-0.8984, 0.1472, 1.1410], [-1.9075, 0.1566, 2.1359], [-0.8659, 0.0573, 1.0337]])
    img_k1 = cv.filter2D(img_gray, -1, K1)
    img_k2 = cv.filter2D(img_gray, -1, K2)
    display_image('k1', img_k1)

    display_image('k2', img_k2)

    # b
    w_k1, u_k1, vt_k1 = cv.SVDecomp(K1)
    w_k2, u_k2, vt_k2 = cv.SVDecomp(K2)
    img_k1_sep = cv.sepFilter2D(img_gray, -1, np.sqrt(w_k1[0, 0]) * vt_k1[0, :], np.sqrt(w_k1[0, 0]) * u_k1[:, 0])
    display_image('k1_sep', img_k1_sep)

    img_k2_sep = cv.sepFilter2D(img_gray, -1, np.sqrt(w_k2[0, 0]) * vt_k2[0, :], np.sqrt(w_k2[0, 0]) * u_k2[:, 0])
    display_image('k2_sep', img_k2_sep)

    # c
    error1 = np.max(cv.absdiff(img_k1, img_k1_sep))
    print(error1)
    error2 = np.max(cv.absdiff(img_k2, img_k2_sep))
    print(error2)
    pass


if __name__ == '__main__':
    task1()
    task2()
    task4()
    task5()
    task7()
    task8()
