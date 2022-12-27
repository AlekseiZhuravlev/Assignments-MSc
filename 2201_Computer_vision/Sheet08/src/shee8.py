import os

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = '../data/task%s'


def decomposePCA(dataPoints, k=None, preservRatio=None):
    # implement PCA for task1 yourself and return the first k 
    # components that preserve preservRatio of the energy and their eigen valuess
    pass


def loadData(filename):
    # return the data points
    df = pd.read_csv(filename, header=0)
    pass


def visualiseHands(kpts, title):
    # use matplotlib for that
    pass


# task 1: training the statistical shape model
# return the trained model so you can use it in task 2
# def task1(train_file='hands_aligned_train.txt'):
#
#     trainfilePath = os.path.join(DATA_DIR%'1', train_file)
#


def read_training_data(path):
    with open(path, 'r') as file:
        data = file.read().splitlines()

    processed_data = []
    for entry in data[1:]:
        splitted = entry.split('\t')
        if not splitted[-1]:
            splitted = splitted[:-1]
        row = np.array(splitted)
        processed_data.append(row.astype(int))

    processed_data = np.array(processed_data)
    return processed_data


def convert_training_data_to_xy(data):
    n_samples = 56

    if len(data) > 1:
        data_t = data.T
    else:
        data_t = data

    x_list = []
    y_list = []
    for entry in data_t:
        x_list.append(entry[:n_samples])
        y_list.append(entry[n_samples:])

    return x_list, y_list


def visualize_hand(x_list, y_list, title):
    plt.plot(x_list, y_list)
    plt.title(title)
    plt.show()


def calculate_mean_hand(data):
    return np.mean(data, axis=1)


def pca(data, mean_hand):
    n_hands = len(data[0])
    mean_hand_repeated = np.tile(np.array([mean_hand]).transpose(), (1, n_hands))

    data_centered = data - mean_hand_repeated

    covariance = np.dot(data_centered, data_centered.transpose())

    _, eigenvalues, eigenvectors = np.linalg.svd(covariance)

    combined = zip(eigenvalues, eigenvectors)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # selecting eigenvalues to keep variance at 0.9
    eigenvalues_sum = sum(eigenvalues)
    current_sum = eigenvalues_sum
    vals_keep = []
    vectors_keep = []
    for eigenvalue, eigenvector in combined:
        current_sum -= eigenvalue
        if current_sum / eigenvalues_sum < 0.1:
            break

        vals_keep.append(eigenvalue)
        vectors_keep.append(eigenvector)

    return vals_keep, vectors_keep


def task1():
    # TASK 1

    print("""

    TASK 1

    """)

    data = read_training_data('../data/task1/hands_aligned_train.txt')
    x_list, y_list = convert_training_data_to_xy(data)
    visualize_hand(x_list[0], y_list[0], 'example_hand')

    mean_hand = calculate_mean_hand(data)

    x_list, y_list = convert_training_data_to_xy(np.array([mean_hand]))
    visualize_hand(x_list[0], y_list[0], 'mean_hand')

    eigenvalues, eigenvectors = pca(data, mean_hand)

    weights = [-0.4, -0.2, 0.0, 0.2, 0.4]
    components = [weights[i] * np.sqrt(eigenvalues[i]) * eigenvectors[i] for i in range(len(eigenvalues))]
    hand = mean_hand + np.sum(components, axis=0)

    x_list, y_list = convert_training_data_to_xy(np.array([hand]))
    visualize_hand(x_list[0], y_list[0], 'reconstructed_hand')

    return eigenvalues, eigenvectors, mean_hand


# task2: performing inference on the test hand data using the trained shape model
def task2(eigenvalues, eigenvectors, mean_hand):
    print("""
    
    TASK 2
    
    """)
    # TASK 2
    # the code should be correct, but the eigenvectors are probably skewed...

    data = read_training_data('../data/task1/hands_aligned_test.txt')
    y = data.flatten()

    weights = np.zeros(5)
    components = np.array([np.sqrt(eigenvalues[i]) * eigenvectors[i] for i in range(len(eigenvalues))])

    for i in range(5):
        x = mean_hand + np.sum(np.dot(weights, eigenvectors), axis=0)

        x_list, y_list = convert_training_data_to_xy(np.array([x]))
        visualize_hand(x_list[0], y_list[0], f'reconstructed_hand_{i}')

        rmse = np.sqrt(np.mean((y - x) ** 2))
        print("RMSE", rmse)

        x_list, y_list = convert_training_data_to_xy(np.array([x]))
        x_joined = np.vstack([x_list, y_list])

        x_list, y_list = convert_training_data_to_xy(np.array([y]))
        y_joined = np.vstack([x_list, y_list]).T

        A = np.vstack(
            [x_joined, np.ones(len(x_joined[0]))]
        ).T
        sol = np.linalg.lstsq(A, y_joined, rcond=None)[0]

        multiplier = sol[0:2, :].T
        addition = sol[2, :]

        multiplier_inv = np.linalg.inv(multiplier)

        y_upd = np.dot(y_joined - addition, multiplier_inv).reshape(-1)

        weights = np.dot(components, y_upd - mean_hand)

        print(weights)


# eigen faces
# (a)you can use scikit PCA for the decomposition and the dataset class for loading LFW dataset directly
# train over LFW training split that you created
# (b) evaluate over the samples in data/task3/detect/face and data/task3/detect/notFace
def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_images(width, height):
    faces_path = '../data/task3/detect/face'
    no_faces_path = '../data/task3/detect/notFace'

    face_list = []
    for image_path in os.listdir(faces_path):
        image = cv2.imread(os.path.join(faces_path, image_path), 0).astype(int) / 255
        image = cv2.resize(image, (width, height))
        face_list.append(image.reshape(-1))

    no_face_list = []
    for image_path in os.listdir(no_faces_path):
        image = cv2.imread(os.path.join(no_faces_path, image_path), 0).astype(int) / 255
        image = cv2.resize(image, (width, height))
        no_face_list.append(image.reshape(-1))

    return np.array(face_list), np.array(no_face_list)


def find_reconstruction_error(pca, img_list):
    img_pca = pca.transform(img_list)
    img_reconstruct = pca.inverse_transform(img_pca)

    # for i in range(4):
    #     fig, ax = plt.subplots(1, 2)
    #
    #     ax[0].imshow(img_list[i].reshape(50, 37), cmap='gray')
    #     ax[0].set_title('original')
    #
    #     ax[1].imshow(img_reconstruct[i].reshape(50, 37), cmap='gray')
    #     ax[1].set_title('reconstructed')
    #     plt.show()

    return np.apply_along_axis(lambda x: np.linalg.norm(x), 1, img_list - img_reconstruct)


def task3():
    print("""

    TASK 3

    """)

    # fetch the dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    scaler = StandardScaler()

    # fit pca
    pca = PCA(n_components=0.95)
    pca.fit(scaler.fit_transform(X_train))

    # reshape the images and display them
    _, height, width = lfw_people.images.shape
    print(height, width)
    eigenfaces = pca.components_.reshape((pca.n_components_, height, width))

    print('number of components:', pca.n_components_)

    fig, axs = plt.subplots(10)
    for i in range(10):
        axs[i].imshow(eigenfaces[i], cmap='gray')
    plt.show()

    # read sample data
    faces_list, no_face_list = read_images(width, height)

    # average face image reconstruction error
    face_errors = find_reconstruction_error(pca, faces_list)
    print('mean reconstruction error, face:', face_errors.mean())

    no_face_errors = find_reconstruction_error(pca, no_face_list)

    threshold = 6
    correct_predictions = 0

    for i in face_errors:
        if i < threshold:
            correct_predictions += 1
    for i in no_face_errors:
        if i > threshold:
            correct_predictions += 1
    print('accuracy on our data:', correct_predictions / (len(face_errors) + len(no_face_errors)))

    knn = KNeighborsClassifier(5)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', pca)
    ])

    # faces_prepared = pipeline.transform(X_train)
    knn.fit(pipeline.fit_transform(X_train), y_train)
    y_pred = knn.predict(pipeline.transform(X_test))

    print('KNN accuracy:', accuracy_score(y_test, y_pred))


# compute the structural tensor M, you can apply an opencv filters to calculate the gradients
def computeStructural(image):
    M = None
    # todo
    return M


def detectorCornerHarris(image, M, responseThresh):
    pass


def detectorCornerFoerstner(image, M, responseThresh):
    pass


# corner detectors: implement Harris corner detector and Förstner corner
def task4(imFile='palace.jpeg'):
    image = cv2.imread(DATA_DIR % '4', imFile)

    # (a)
    # todo
    # (b) apply Harris corner detector and visualise
    # todo
    # (c) apply Foerstner corner detector and visualise
    # todo
    pass


# perform the matching using sift
def match(sift, image1, image2):
    pass


def rotateImage(image, rotAngle):
    # to get the rotation matrix, you can use cv2.getRotationMatrix2D
    rotMat = None
    rotatedIm = cv2.warpAffine(image, rotMat)  # fill in the missing parameters
    return rotatedIm


# keypoint matching
def task5(imFile='castle.jpeg'):
    image = cv2.imread(DATA_DIR % '5' % imFile)
    # you can use cv2's implementation of SIFT
    sift = cv2.SIFT_create()
    # apply each of the following transformations and then perform matching
    # Please choose two values of your choice for each one
    rotAngles = [0, 0]  # todo choose two values
    for rotAngle in rotAngles:
        rotatedIm = rotateImage(image, rotAngle)

        match(sift, image, rotatedIm)
        # todo visualise using cv2.drawMatches

    # do the same with a translation transformation

    # do the same with a scaling transformation


# Image homography: implement the RANSAC algorithm then apply it to stitch the two images of Bonn's
# Poppelsdorfer Schloss and then visualise the stitched image
def task6():
    pass


if __name__ == "__main__":
    eigenvalues, eigenvectors, mean_hand = task1()
    task2(eigenvalues, eigenvectors, mean_hand)
    task3()
    task4()
    task5()
    task6()
