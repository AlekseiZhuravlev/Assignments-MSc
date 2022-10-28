import os
import time

import cv2
import numpy as np

DATA_DIR = '..\data'
SIM_THRESHOLD = 0.5  # similarity threshold for template matching. Can be adapted.


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# implement the sum square difference (SQD) similarity
def calc_sum_square_difference(image, template):
    h = np.zeros(image.shape)
    for m in range(image.shape[0] - template.shape[0]):
        for n in range(image.shape[1] - template.shape[1]):
            # h[m, n] =
            diff_mat = (image[m: m + template.shape[0], n:n + template.shape[1]].astype(int) - template.astype(
                int)) / 255
            h[m, n] = np.sum(diff_mat ** 2)

    h[image.shape[0] - template.shape[0]:, :] = 1
    h[:, image.shape[1] - template.shape[1]:] = 1
    return h


# implement the normalized cross correlation (NCC) similarity
# def calc_normalized_cross_correlation(image, template):
#     image_mean = int(np.mean(image.astype(int)))
#     template_mean = int(np.mean(template.astype(int)))
#     h = np.zeros(image.shape)
#     for m in range(image.shape[0] - template.shape[0]):
#         for n in range(image.shape[1] - template.shape[1]):
#             g_norm = 0
#             h_norm = 0
#             up = 0
#             for k in range(template.shape[0]):
#                 for l in range(template.shape[1]):
#                     g_norm = g_norm + (int(template[k, l]) - template_mean) ** 2
#                     h_norm = h_norm + (int(image[m + k, n + l]) - image_mean) ** 2
#                     up = up + (int(template[k, l]) - template_mean) * (int(image[m + k, n + l]) - image_mean)
#             h[m, n] = up / ((g_norm * h_norm) ** 0.5)
#     return h

def calc_normalized_cross_correlation(image, template):
    template_mean = cv2.mean(template)[0]
    template_diff = (template.astype(int) - template_mean) / 255

    template_diff_norm = np.sum(template_diff ** 2)

    h = np.zeros(image.shape)
    for m in range(image.shape[0] - template.shape[0]):
        for n in range(image.shape[1] - template.shape[1]):
            g_norm = 0
            h_norm = 0
            up = 0

            image_patch = image[
                          m: m + template.shape[0],
                          n: n + template.shape[1]
                          ].astype(int)
            image_mean = cv2.mean(image_patch)[0]
            img_diff = (image_patch - image_mean) / 255
            img_diff_norm = np.sum(img_diff ** 2)

            upper_part = np.sum(img_diff * template_diff)
            lower_part = np.sqrt(img_diff_norm * template_diff_norm)
            h[m, n] = upper_part / lower_part
    return h


# draw rectanges on the input image in regions where the similarity is larger than SIM_THRESHOLD
# def draw_rectangles(input_im, similarity_im):
#     start_rect = False
#     start = []
#     end = []
#     new = input_im.copy()
#     for x in range(input_im.shape[0] - 1):
#         for y in range(input_im.shape[1] - 1):
#             if similarity_im[x, y] > SIM_THRESHOLD and not start_rect:
#                 start.append((x, y))
#                 start_rect = True
#             if similarity_im[x, y] > SIM_THRESHOLD >= similarity_im[x + 1, y] and similarity_im[
#                 x, y + 1] <= SIM_THRESHOLD and start_rect and x >= start[-1][0] and y >= start[-1][1]:
#                 end.append((x, y))
#                 start_rect = False
#     for i in range(np.shape(start)[0]):
#         new = cv2.rectangle(new, start[i], end[i], (255, 0, 0), 1)
#     return new

def draw_rectangles(input_im, similarity_im, template):
    new = input_im.copy()
    for x in range(input_im.shape[0]):
        for y in range(input_im.shape[1]):
            if similarity_im[x, y] < SIM_THRESHOLD:
                cv2.rectangle(new, (y, x), (y + template.shape[1], x + template.shape[0]), (255, 0, 0), 1)
    return new


# You can choose to resize the image using the new dimensions or the scaling factor
def pyramid_down(image, dstSize, scale_factor=None):
    pass


# create a pyramid of the image using the specified pyram function pyram_method.
# pyram_func can either be cv2.pyrDown or your own implementation
def create_gaussian_pyramid(image, pyram_func, num_levels):
    # in a loop, create a pyramid of downsampled blurred images using the Gaussian kernel
    pass


def calc_derivative_gaussian_kernel(size, sigma):
    # TODO: implement
    der_x = np.zeros((size, size))
    der_y = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            gaussian = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(
                -((i - size // 2) ** 2 + (j - size // 2) ** 2) / (2 * sigma ** 2))
            der_x[i, j] = gaussian * (-(i - size // 2)) / (sigma ** 2)
            der_y[i, j] = gaussian * (-(j - size // 2)) / (sigma ** 2)
    return der_x, der_y


def create_laplacian_pyramid(image, num_levels=5):
    # create the laplacian pyramid using the gaussian pyramid
    gaussian_pyramid = create_gaussian_pyramid(image, cv2.pyrdown, num_levels)
    # complete as described in the exercise sheet
    pass


# Given the final weighted pyramid, sum up the images at each level with the upscaled previous level
def collapse_pyramid(laplacian_pyramid):
    final_im = laplacian_pyramid[0]
    for l in range(1, len(laplacian_pyramid)):
        # TODO complete code
        pass
    return final_im


# Fourier Transform

# blur the image in the spatial domain using convolution
def blur_im_spatial(image, kernel):
    im_blurred = cv2.filter2D(image.copy(), -1, kernel)
    display_image('blurred_spacial', im_blurred)
    return im_blurred


def to_shape(a, shape):
    # TODO problem with padding

    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_ - y)
    x_pad = (x_ - x)
    return np.pad(a, ((0, y_pad),
                      (0, x_pad)
                      ))


# blur the image in the frequency domain
def blur_im_freq(image, kernel):
    img_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(
        to_shape(kernel, image.shape)
    )
    output_freq = np.multiply(img_fft, kernel_fft)
    output_spac = np.fft.ifft2(
        output_freq
    )
    output_real = np.abs(output_spac)
    output_scaled = (output_real / np.max(output_real) * 255).astype(np.uint8)
    display_image('blurred_fequency', output_scaled)
    return output_scaled


def get_gaussian_kernel(k_size, sigma):
    # Your implementation of getGaussianKernel
    bound = (k_size // 2)
    ax = list(range(-bound, bound + 1))
    kernel_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / np.sum(kernel_2d)


def get_root_mean_square_pixel_difference(img_1, img_2):
    diff_sum = 0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            diff_sum += np.square(int(img_1[i, j]) - int(img_2[i, j]))
    return np.sqrt(diff_sum / img_1.size)


def task1(input_im_file):
    full_path = os.path.join(DATA_DIR, input_im_file)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    display_image('original', image)

    kernel = get_gaussian_kernel(k_size=7, sigma=1)

    # time the blurring of the different methods
    start_time = time.time()
    conv_result = blur_im_spatial(image, kernel)
    time_conv = time.time() - start_time
    print('time taken to apply blur in the spatial domain', time_conv)

    # measure the timing here too
    start_time = time.time()
    fft_result = blur_im_freq(image, kernel)
    time_fft = time.time() - start_time
    print('time taken to apply blur in the frequency domain', time_fft)

    msd = get_root_mean_square_pixel_difference(conv_result, fft_result)
    print(f'time_fft - time_conv = {time_fft - time_conv}')
    print(f'root mean square pixel difference = {msd}')


# Template matching using single-scale
def task2(input_im_file, template_im_file):
    full_path_im = os.path.join(DATA_DIR, input_im_file)
    full_path_template = os.path.join(DATA_DIR, template_im_file)
    in_im = cv2.imread(full_path_im, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(full_path_template, cv2.IMREAD_GRAYSCALE)

    start_time = time.time()
    result_sqd = calc_sum_square_difference(in_im, template)
    print(f'ssd took {time.time() - start_time}')

    start_time = time.time()
    result_ncc = calc_normalized_cross_correlation(in_im, template)
    print(f'ncc took {time.time() - start_time}')

    display_image('sqd', result_sqd)
    display_image('ncc', result_ncc)
    # draw rectanges at matching regions
    vis_sqd = draw_rectangles(in_im, result_sqd, template)
    vis_ncc = draw_rectangles(in_im, 1 - result_ncc, template)

    display_image('sqd', vis_sqd)
    display_image('ncc', vis_ncc)


def task3(input_im_file, template_im_file):
    pass
    # TODO: calculate the time needed for template matching with the pyramid

    # TODO: show the template matching results using the pyramid


# Image blending
def task4(input_im_file1, input_im_file2, interest_region_file, num_pyr_levels=5):
    # TODO you can use the steps described in the exercise sheet to help guide you through the solution
    # import images
    im_1 = cv2.imread(os.path.join(DATA_DIR, input_im_file1))
    im_2 = cv2.imread(os.path.join(DATA_DIR, input_im_file2))

    # create Gaussian Pyramids
    GA = [im_1.copy()]
    GB = [im_2.copy()]
    for i in range(num_pyr_levels):
        A = cv2.pyrDown(GA[-1])
        B = cv2.pyrDown(GB[-1])
        GA.append(A)
        GB.append(B)

    # create Laplacian Pyramids
    LA = [GA[-1]]
    LB = [GB[-1]]
    for i in range(num_pyr_levels - 1, -1, -1):
        GA_exp = cv2.pyrUp(GA[i + 1], dstsize=(GA[i].shape[1], GA[i].shape[0]))
        GB_exp = cv2.pyrUp(GB[i + 1], dstsize=(GB[i].shape[1], GB[i].shape[0]))
        A = cv2.subtract(GA[i], GA_exp)
        B = cv2.subtract(GB[i], GB_exp)
        LA.insert(0, A)
        LB.insert(0, B)

    # import mask
    mask = cv2.imread(os.path.join(DATA_DIR, interest_region_file), cv2.IMREAD_GRAYSCALE)

    # create Gaussian Pyramid for mask
    GR = [mask.copy()]
    for i in range(num_pyr_levels):
        R = cv2.pyrDown(GR[-1])
        GR.append(R)

    # combine Laplacian pyramids
    LS = []
    for i in range(num_pyr_levels + 1):
        S = LA[i].copy()
        for x in range(LA[i].shape[0]):
            for y in range(LA[i].shape[1]):
                for p in range(LA[i].shape[2]):
                    alpha = GR[i][x, y] / 255
                    S[x, y, p] = int(min(255, alpha * LA[i][x, y, p] + (1 - alpha) * LB[i][x, y, p]))
        LS.append(S)

    # collapse
    for i in range(num_pyr_levels - 1, -1, -1):
        S_exp = cv2.pyrUp(LS[i + 1], dstsize=(LS[i].shape[1], LS[i].shape[0]))
        LS[i] = cv2.add(LS[i], S_exp)

    result = LS[0]
    cv2.imshow('result', result)
    cv2.waitKey(0)
    return result


def task5(input_im, kernel_size=5, sigma=0.5):
    image = cv2.imread("../data/einstein.jpeg", 0)

    kernel_x, kernel_y = calc_derivative_gaussian_kernel(kernel_size, sigma)

    edges_x = cv2.filter2D(image, -1, kernel_x)  # TODO: convolve with kernel_x
    edges_y = cv2.filter2D(image, -1, kernel_y)  # TODO: convolve with kernel_y

    magnitude = np.zeros((edges_x.shape[0], edges_x.shape[1]))  # TODO: compute edge magnitude
    direction = np.zeros((edges_x.shape[0], edges_x.shape[1]))  # TODO: compute edge direction

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            magnitude[i, j] = np.sqrt(edges_x[i, j] ** 2 + edges_y[i, j] ** 2)
            direction[i, j] = np.arctan2(edges_x[i, j], edges_y[i, j])
            if magnitude[i, j] > 0:
                dx = np.cos(direction[i, j]) * magnitude[i, j] / 100
                dy = np.sin(direction[i, j]) * magnitude[i, j] / 100
                image = cv2.arrowedLine(image, (j, i), (int(np.ceil(j + dy)), int(np.ceil(i + dx))), (0, 255, 0), 1)

    # TODO visualise the results
    cv2.imshow('edges', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # task1('orange.jpeg')
    # task1('celeb.jpeg')
    task2('RidingBike.jpeg', 'RidingBikeTemplate.jpeg')
    # task3('DogGray.jpeg', 'DogTemplate.jpeg')
    # task4('dog.jpeg', 'moon.jpeg', 'mask.jpeg')
    # just for fun, blend these these images as well
    # for i in [1, 2, 10]:
    #     ind = str(i).zfill(2)
    #     # blended_im = task4('task4_extra/source_%s.jpg'%ind, 'task4_extra/target_%s.jpg'%ind, 'task4_extra/mask_%s.jpg'%ind)
    #     # visualise the blended image
    #
    # task5('einstein.jpeg')
