import cv2 as cv
import matplotlib.pyplot as plt

# --- this are all the imports you are supposed to use!
# --- please do not add more imports!


if __name__ == '__main__':
    n_views = 101
    n_features = 215

    with open('data/data_matrix.txt', 'r') as f:
        lines = f.readlines()

    lines = [[float(coordinate) for coordinate in line.split(' ')] for line in lines]

    plt.ion()
    plt.show()

    for i in range(n_views):
        plt.clf()

        img = cv.imread(f'data/frame{i + 1:08}.jpg')
        plt.imshow(img)
        plt.scatter(lines[2 * i], lines[2 * i + 1], marker='.')

        plt.draw()
        plt.pause(0.001)
