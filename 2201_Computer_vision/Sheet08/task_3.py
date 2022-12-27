import pathlib

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def read_images(width, height):
    faces_path = pathlib.Path('data/task3/detect/face')
    no_faces_path = pathlib.Path('data/task3/detect/notFace')

    face_list = []
    for image_path in faces_path.iterdir():
        image = cv.imread(str(image_path), 0).astype(int) / 255
        image = cv.resize(image, (width, height))
        face_list.append(image.reshape(-1))

    no_face_list = []
    for image_path in no_faces_path.iterdir():
        image = cv.imread(str(image_path), 0).astype(int) / 255
        image = cv.resize(image, (width, height))
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


def task_3():
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


if __name__ == '__main__':
    task_3()
