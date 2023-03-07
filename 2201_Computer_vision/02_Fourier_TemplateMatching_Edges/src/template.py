import itertools
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


def calc_normalized_cross_correlation(image, template, coordinate_range=None):
    template_mean = cv2.mean(template)[0]
    template_diff = (template.astype(int) - template_mean) / 255

    template_diff_norm = np.sum(template_diff ** 2)

    h = np.zeros(image.shape)
    if not coordinate_range:
        x_range = list(range(0, image.shape[0] - template.shape[0]))
        y_range = list(range(0, image.shape[1] - template.shape[1]))
        coordinate_range = list(itertools.product(x_range, y_range))

    for m, n in coordinate_range:
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


def draw_rectangles(input_im, similarity_im, template):
    new = input_im.copy()
    for x in range(input_im.shape[0]):
        for y in range(input_im.shape[1]):
            if similarity_im[x, y] >= SIM_THRESHOLD:
                cv2.rectangle(new, (y, x), (y + template.shape[1], x + template.shape[0]), (255, 0, 0), 1)
    return new


def calc_derivative_gaussian_kernel(size, sigma):
    der_x = np.zeros((size, size))
    der_y = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            gaussian = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(
                -((i - size // 2) ** 2 + (j - size // 2) ** 2) / (2 * sigma ** 2))
            der_x[i, j] = gaussian * (-(i - size // 2)) / (sigma ** 2)
            der_y[i, j] = gaussian * (-(j - size // 2)) / (sigma ** 2)
    return der_x, der_y


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
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_ - y)
    x_pad = (x_ - x)
    padded = np.pad(a, ((0, y_pad),
                        (0, x_pad)
                        ))
    return padded


# blur the image in the frequency domain
def blur_im_freq(image, kernel):
    img_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(
        to_shape(kernel, image.shape)
    )
    kernel_fft = np.abs(kernel_fft)

    output_freq = np.multiply(img_fft, kernel_fft)
    output_spac = np.fft.ifft2(
        output_freq
    )
    output_real = np.abs(output_spac)
    output_scaled = (output_real / np.max(output_real) * 255).astype(np.uint8)
    display_image('blurred_frequency', output_scaled)
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
    vis_sqd = draw_rectangles(in_im, 1 - result_sqd, template)
    vis_ncc = draw_rectangles(in_im, result_ncc, template)

    display_image('recognized_sqd', vis_sqd)
    display_image('recognized_ncc', vis_ncc)


def blur_and_downsample(img, kernel):
    blurred_img = cv2.filter2D(img.copy(), -1, kernel)
    downsized_img = cv2.resize(
        blurred_img.copy(),
        dsize=((blurred_img.shape[1] + 1) // 2, (blurred_img.shape[0] + 1) // 2)
    )
    return downsized_img


def get_mean_pixel_difference(img_1, img_2):
    diff = np.abs(img_1.astype(int) - img_2.astype(int))
    return np.sum(diff) / img_1.size


def get_region_of_interest_ncc(ncc):
    reg_of_interest = []
    for x in range(ncc.shape[0]):
        for y in range(ncc.shape[1]):
            if ncc[x, y] >= SIM_THRESHOLD:
                reg_of_interest.append((x, y))

    # adding 3 adjacent pixels
    for i in reg_of_interest.copy():
        reg_of_interest.append((i[0] + 1, i[1] + 1))
        reg_of_interest.append((i[0], i[1] + 1))
        reg_of_interest.append((i[0] + 1, i[1]))

    # removing possible duplicates
    reg_of_interest = set(reg_of_interest)

    return list(reg_of_interest)


def task3(input_im_file, template_im_file):
    img = cv2.imread(os.path.join(DATA_DIR, input_im_file), cv2.IMREAD_GRAYSCALE)
    templ = cv2.imread(os.path.join(DATA_DIR, template_im_file), cv2.IMREAD_GRAYSCALE)
    gauss_kernel = get_gaussian_kernel(k_size=5, sigma=1)

    # own implementation
    pyramid_img = [img.copy()]
    pyramid_templ = [templ.copy()]
    for i in range(4):
        downs_img = blur_and_downsample(pyramid_img[-1], gauss_kernel)
        downs_templ = blur_and_downsample(pyramid_templ[-1], gauss_kernel)
        pyramid_img.append(downs_img)
        pyramid_templ.append(downs_templ)

    # built-in
    pyramid_img_bin = [img.copy()]
    pyramid_templ_bin = [templ.copy()]
    for i in range(4):
        downs_img = cv2.pyrDown(pyramid_img_bin[-1])
        downs_templ = cv2.pyrDown(pyramid_templ_bin[-1])
        pyramid_img_bin.append(downs_img)
        pyramid_templ_bin.append(downs_templ)

    # showing median error
    for i, (img_own, img_bin) in enumerate(zip(pyramid_img, pyramid_img_bin)):
        print(
            f'median error at {i} level: {get_mean_pixel_difference(img_own, img_bin):.2f}'
        )

    # pattern recognition at different levels of the pyramid
    for i, (img_pyr, templ_pyr) in enumerate(zip(pyramid_img_bin, pyramid_templ_bin)):
        start_time = time.time()
        result_ncc = calc_normalized_cross_correlation(img_pyr, templ_pyr)
        print(f'ncc at {i} level took {time.time() - start_time:.2f}')
        display_image('ncc', result_ncc)

        # draw rectanges at matching regions
        vis_ncc = draw_rectangles(img_pyr, result_ncc, templ_pyr)
        display_image('recognized_ncc', vis_ncc)

    print('pattern recognition from the bottom of pyramid')
    # pattern recognition from the bottom of pyramid
    region_of_interest = None
    for i, (img_pyr, templ_pyr) in enumerate(zip(pyramid_img_bin[::-1], pyramid_templ_bin[::-1])):
        start_time = time.time()
        result_ncc = calc_normalized_cross_correlation(img_pyr, templ_pyr, region_of_interest)
        print(f'ncc at {i} level took {time.time() - start_time:.3f}')
        display_image('ncc', result_ncc)

        # draw rectanges at matching regions
        vis_ncc = draw_rectangles(img_pyr, result_ncc, templ_pyr)
        display_image('recognized_ncc', vis_ncc)

        # get region of interest
        region_of_interest = get_region_of_interest_ncc(result_ncc)
        # rescale coordinates
        for i in range(len(region_of_interest)):
            region_of_interest[i] = [
                region_of_interest[i][0] * 2,
                region_of_interest[i][1] * 2
            ]


# Image blending
def task4(input_im_file1, input_im_file2, interest_region_file, num_pyr_levels=5):
    # import images
    im_1 = cv2.imread(os.path.join(DATA_DIR, input_im_file1))
    im_2 = cv2.imread(os.path.join(DATA_DIR, input_im_file2))

    # we use padding to properly position image 1 with respect to 2
    # it requires manual setup for each image
    im_1 = cv2.copyMakeBorder(im_1.copy(),
                              im_2.shape[0] - im_1.shape[0],
                              0,
                              0,
                              im_2.shape[1] - im_1.shape[1],
                              cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])

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

    mask = cv2.copyMakeBorder(mask.copy(),
                              im_2.shape[0] - mask.shape[0],
                              0,
                              0,
                              im_2.shape[1] - mask.shape[1],
                              cv2.BORDER_CONSTANT,
                              value=0)

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
    display_image('result', result)
    return result


def task5(input_im, kernel_size=5, sigma=0.5):
    image = cv2.imread("../data/einstein.jpeg", 0)

    display_image('original', image)

    kernel_x, kernel_y = calc_derivative_gaussian_kernel(kernel_size, sigma)

    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)

    magnitude = np.zeros((edges_x.shape[0], edges_x.shape[1]))
    direction = np.zeros((edges_x.shape[0], edges_x.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            magnitude[i, j] = np.sqrt(edges_x[i, j] ** 2 + edges_y[i, j] ** 2)
            direction[i, j] = np.arctan2(edges_x[i, j], edges_y[i, j])
            if magnitude[i, j] > 0:
                dx = np.cos(direction[i, j]) * magnitude[i, j] / 100
                dy = np.sin(direction[i, j]) * magnitude[i, j] / 100
                image = cv2.arrowedLine(image, (j, i), (int(np.ceil(j + dy)), int(np.ceil(i + dx))), (0, 255, 0), 1)

    display_image('edges', image)


if __name__ == "__main__":
    task1('orange.jpeg')
    task1('celeb.jpeg')
    task2('RidingBike.jpeg', 'RidingBikeTemplate.jpeg')
    task3('DogGray.jpeg', 'DogTemplate.jpeg')
    task4('dog.jpeg', 'moon.jpeg', 'mask.jpeg')
    # just for fun, blend these these images as well
    for i in [1, 2, 10]:
        ind = str(i).zfill(2)
        blended_im = task4('task4_extra/source_%s.jpg' % ind, 'task4_extra/target_%s.jpg' % ind,
                           'task4_extra/mask_%s.jpg' % ind)

    # visualise the blended image

    task5('einstein.jpeg')
