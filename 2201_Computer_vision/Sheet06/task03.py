#!/usr/bin/python3.5

import numpy as np
import cv2 as cv
from scipy.stats import multivariate_normal


def read_image(filename):
    """
        load the image and foreground/background parts
        image: the original image
        background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
    """
    image = cv.imread(filename).astype(int) / 255
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[150:250, 130:250, :] = 1
    foreground = image[bounding_box == 1].reshape((-1, 3))
    background = image[bounding_box == 0].reshape((-1, 3))
    return image, foreground, background


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


class GaussianMixture:
    def __init__(self, data):
        self.data = data
        self.weights = []
        self.means = []
        self.c_matrices = []

        self.responsibilities = None

    def fit_single_gaussian(self):
        self.weights.append(1)
        self.means.append(self.data.mean(axis=0))
        self.c_matrices.append(
            np.diag(self.data.std(axis=0))
        )

    def split(self):
        new_means = []
        new_weights = []
        new_c_matrices = []
        epsilon = 1

        for i in range(len(self.means)):
            # weights
            new_weights.append(self.weights[i])
            new_weights.append(self.weights[i])

            # means
            mean_1 = self.means[i] + epsilon * np.diag(self.c_matrices[i])
            mean_2 = self.means[i] - epsilon * np.diag(self.c_matrices[i])
            new_means.append(
                np.clip(mean_1, 0, 1)
            )
            new_means.append(
                np.clip(mean_2, 0, 1)
            )

            # covariance matrices
            new_c_matrices.append(self.c_matrices[i])
            new_c_matrices.append(self.c_matrices[i])

        self.weights = np.array(new_weights) / 2
        self.means = new_means
        self.c_matrices = new_c_matrices

    def print_gaussians(self):
        print('weights')
        print(self.weights)
        print('means')
        print(self.means)
        print('c_matrices')
        print(self.c_matrices)

    def e_step(self):
        self.responsibilities = np.zeros((len(self.data), len(self.weights)))

        for i, data_entry in enumerate(self.data):
            upper_terms = np.zeros(len(self.weights))

            for j in range(len(self.weights)):
                gauss_func = multivariate_normal(mean=self.means[j], cov=self.c_matrices[j])
                upper_terms[j] = self.weights[j] * gauss_func.pdf(data_entry)

            responsibilities_row = upper_terms / np.sum(upper_terms)
            self.responsibilities[i] = responsibilities_row

    def m_step(self):
        # weights
        new_weights = self.responsibilities.sum(axis=0) / self.responsibilities.sum()

        # means
        new_means = [0 for _ in range(len(self.means))]
        for i in range(len(self.weights)):
            temp = [0 for _ in range(len(self.data))]

            for j in range(len(self.data)):
                temp[j] = self.responsibilities[j, i] * self.data[j]
            new_means[i] = np.sum(temp, axis=0) / self.responsibilities.sum(axis=0)[i]

        # covariance matrices
        new_c_matrices = [0 for _ in range(len(self.c_matrices))]
        for i in range(len(self.weights)):
            temp = [0 for _ in range(len(self.data))]

            for j in range(len(self.data)):
                temp[j] = self.responsibilities[j, i] * \
                          np.outer(self.data[j] - new_means[i], self.data[j] - new_means[i])
            new_c_matrices[i] = np.sum(temp, axis=0) / self.responsibilities.sum(axis=0)[i]

        self.weights = new_weights
        self.means = new_means
        self.c_matrices = new_c_matrices

    def get_probability(self, pixel):
        probty = 0
        for j in range(len(self.weights)):
            gauss_func = multivariate_normal(mean=self.means[j], cov=self.c_matrices[j])
            probty += self.weights[j] * gauss_func.pdf(pixel)
        return probty

    def set_background_black(self, image, threshold):
        for i in range(image.shape[0]):
            for j in range(75, 490):  # range(image.shape[1]):
                # image[i, j] = self.get_probability(image[i, j])
                if self.get_probability(image[i, j]) < threshold:
                    image[i, j] = np.zeros(3)
        return image


if __name__ == '__main__':

    image, foreground, background = read_image('data/cv_is_great.png')
    print(background)

    # area for training gaussian mixture
    cv.rectangle(image, (150, 130), (250, 250), (255, 0, 0), 2)

    # test area for subtracting the background
    cv.rectangle(image, (75, 0), (490, image.shape[0]), (0, 0, 255), 2)

    display_image('original', image)

    gmm = GaussianMixture(data=foreground)
    gmm.fit_single_gaussian()
    gmm.print_gaussians()

    gmm.split()
    gmm.print_gaussians()
    gmm.split()
    gmm.print_gaussians()

    gmm.e_step()
    gmm.m_step()
    gmm.print_gaussians()

    for threshold in [0.75]:
        img = gmm.set_background_black(image.copy(), threshold)
        display_image('gmm_adjusted', img)

    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
