import cv2 as cv
import numpy as np


def display_image(img, window_name='image'):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    matrix = [
        [3, 1, -9, -2, 0],
        [5, 2, 2, 3, -1],
        [9, 4, -9, -8, 1],
        [2, 10, -20, -20, 0],
        [4, 8, 4, -6, 0]
    ]

    matrix = [
        [-21, 6, 3, 12, 9],
        [7, -2, -1, -4, -3],
        [0, 0, 0, 0, 0],
        [35, -10, -5, -20, -15],
        [-14, 4, 2, 8, 6]
    ]
    a, b, c = np.linalg.svd(matrix)

    value = b[0]
    v1 = a[:, 0] * np.sqrt(value)
    v2 = c[0] * np.sqrt(value)

    print('v1', v1)
    print('v2', v2)
    print('v1 x v2\n', np.round(np.outer(v1, v2), 2))
