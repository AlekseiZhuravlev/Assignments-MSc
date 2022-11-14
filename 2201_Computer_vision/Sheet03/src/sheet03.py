import numpy as np
import cv2 as cv


# import random
# from collections import defaultdict

##############################################
#     Task 1        ##########################
##############################################

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def draw_circles(img, circles):
    for i in circles[0]:
        # draw the outer circle
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/billiards.png')
    display_image('billiards', img)

    img = cv.medianBlur(img, 7)
    cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimg, cv.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))

    draw_circles(img, circles)

    display_image('cv.HoughCircles', img)


def check_distance(circle, detected_circles, min_dist):
    for i in detected_circles:
        if np.sqrt(
                (circle[0] - i[0]) ** 2 + (circle[1] - i[1]) ** 2
        ) < min_dist:
            return False
    return True


def myHoughCircles(edges, minRadius, maxRadius, threshold, minDist):
    """
    Your implementation of HoughCircles
    :edges: single-channel binary source image (e.g: edges)
    :minRadius: minimum circle radius
    :maxRadius: maximum circle radius
    :param threshold: minimum number of votes to consider a detection
    :minDist: minimum distance between two centers of the detected circles. 
    :return: list of detected circles as (a, b, r) triplet
    """

    print(
        """
        Commentary:
        min-max radii - if we set greater radius, we will get more potential circles
        threshold - smaller threshold -> more circles in which the algorithm is not confident
        min distance - smaller distance -> more adjacent circles which may overlap and make output less clear
        """
    )

    accumulator = np.zeros((
        edges.shape[0],
        edges.shape[1],
        maxRadius + 1
    ), dtype=int)
    detected_circles = []

    edges_points = np.nonzero(edges)

    for y, x in zip(*edges_points):
        for r in np.linspace(minRadius, maxRadius, 25, dtype=int):
            for theta in np.linspace(0, np.pi * 2, 60):
                a = x - r * np.cos(theta)
                b = y - r * np.sin(theta)

                # keeping circle candidates with center inside the image
                if 0 <= a <= edges.shape[0] - 1 and 0 <= b <= edges.shape[1] - 1:
                    accumulator[
                        int(a),
                        int(b),
                        r
                    ] += 1

    accumulator[accumulator < threshold] = 0
    circle_candidates = np.nonzero(accumulator)

    for i in zip(*circle_candidates):
        if check_distance(i, detected_circles, minDist):
            detected_circles.append(i)

    return [detected_circles]


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/billiards.png')
    minRadius = 15
    maxRadius = 100
    minDist = img.shape[0] / 16
    threshold = 20

    # convert the image into grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # detect the edges
    edges = cv.Canny(img_gray, 100, 200)
    display_image('edges', edges)

    detected_circles = myHoughCircles(edges, minRadius, maxRadius, threshold, minDist)

    draw_circles(img, detected_circles)

    display_image('myHoughCircles', img)


##############################################
#     Task 2        ##########################
##############################################

def houghLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    edges_points = np.array(np.nonzero(img_edges))

    for i in range(edges_points.shape[1]):
        for theta in range(0, 180, theta_step_sz):
            d = int((edges_points[1][i] * np.cos(theta * np.pi / 180.) + edges_points[0][i] * np.sin(
                theta * np.pi / 180.)) / d_resolution)
            accumulator[int(theta / theta_step_sz), d] += 1

    accumulator_copy = accumulator
    detected_lines = []
    finished = False
    while not finished:
        idx = np.argmax(accumulator_copy)
        theta, d = np.unravel_index(idx, accumulator_copy.shape)

        if accumulator_copy[theta, d] > threshold:
            detected_lines.append([d * d_resolution, theta * theta_step_sz * np.pi / 180.])
        else:
            finished = True

        accumulator_copy[theta, d] = 0

    return detected_lines, accumulator


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None  # convert the image into grayscale
    edges = None  # detect the edges
    theta_res = None  # set the resolution of theta
    d_res = None  # set the distance resolution
    # _, accumulator = houghLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 3        ##########################
##############################################


def myKmeans_intensity(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """

    cluster_map = np.zeros((
        data.shape[0], data.shape[1]
    ), dtype=int)

    # initialize centers using some random points from data
    x_rand = np.random.choice(range(data.shape[0]), k)
    y_rand = np.random.choice(range(data.shape[1]), k)

    # TODO generalize
    centers = [data[i[0], i[1]] for i in zip(x_rand, y_rand)]

    convergence = False
    iterationNo = 0
    while not convergence:
        new_centers = np.zeros(k)
        new_centers_counts = np.zeros(k)

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                distances = [(int(data[x, y]) - center) ** 2 for center in centers]

                nearest_center = np.argmin(distances)

                cluster_map[x, y] = nearest_center

                new_centers[nearest_center] += int(data[x, y])
                new_centers_counts[nearest_center] += 1

        for i in range(k):
            new_centers[i] /= new_centers_counts[i]

        center_change = np.sqrt(np.sum((centers - new_centers) ** 2))

        if center_change < 1:
            convergence = True
        else:
            centers = new_centers.copy()

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return cluster_map


def task_3_a():
    # print("Task 3 (a) ...")
    # img = cv.imread('../images/flower.png', cv.IMREAD_GRAYSCALE)
    #
    # for k in [2, 4, 6]:
    #     cluster_map = myKmeans_intensity(img, k)
    #     cluster_map *= 255//k
    #
    #     display_image('clusters', cluster_map.astype(np.uint8))

    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')

    for k in [2, 4, 6]:
        cluster_map = myKmeans_color(img, k)
        cluster_map *= 255 // k

        display_image('clusters', cluster_map.astype(np.uint8))


def myKmeans_color(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """

    cluster_map = np.zeros((
        data.shape[0], data.shape[1]
    ), dtype=int)

    # initialize centers using some random points from data
    x_rand = np.random.choice(range(data.shape[0]), k)
    y_rand = np.random.choice(range(data.shape[1]), k)

    # TODO generalize
    centers = [data[i[0], i[1]].astype(int) for i in zip(x_rand, y_rand)]

    convergence = False
    iterationNo = 0
    while not convergence:
        new_centers = [np.zeros(3) for _ in range(k)]
        new_centers_counts = np.zeros(k)

        print(centers)

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                distances = [np.sum((data[x, y].astype(int) - center) ** 2) for center in centers]

                nearest_center = np.argmin(distances)

                cluster_map[x, y] = nearest_center

                new_centers[nearest_center] += data[x, y].astype(int)
                new_centers_counts[nearest_center] += 1

        for i in range(k):
            new_centers[i] /= new_centers_counts[i]

        center_change = [np.zeros(3) for _ in range(k)]

        # print(centers[0] - new_centers[0])
        # print(new_centers[0])
        # print(center_change[0])

        # print(centers[k] - new_centers[k])

        for i in range(k):
            print(i)
            center_change[i] = np.sqrt(np.sum((centers[i] - new_centers[i]) ** 2))

        if np.sqrt(np.sum(center_change) ** 2) < 1:
            convergence = True
        else:
            centers = new_centers.copy()

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return cluster_map


# TODO return center VALUES
def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################

if __name__ == "__main__":
    # task_1_a()
    # task_1_b()
    # task_2()
    task_3_a()
    # task_3_b()
    # task_3_c()
    # task_4_a()
