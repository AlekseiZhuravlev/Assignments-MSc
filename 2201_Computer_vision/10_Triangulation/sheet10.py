import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def display_image(title, image):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extract_sift_keypoints(img):
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img_grey, None)

    img_keypoints = cv.drawKeypoints(img_grey, kp, img)
    display_image('sift_keypoints', img_keypoints)

    return kp, des


if __name__ == '__main__':
    # Load data
    query_img = cv.imread('data/1.jpg')
    train_img = cv.imread('data/2.jpg')

    # Extract SIFT key points and features
    query_kp, query_des = extract_sift_keypoints(query_img)
    train_kp, train_des = extract_sift_keypoints(train_img)

    # Compute matches

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L2)

    # Match descriptors.
    matches = bf.match(query_des, train_des)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first matches.
    img_matches = cv.drawMatches(
        query_img, query_kp, train_img, train_kp, matches[:300],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(img_matches)
    plt.show()

    # Projection matrix for query_img and train_img
    P_q = np.array([[1.0, 0, 0, 0],
                    [0, 1.0, 0, 0],
                    [0, 0, 1.0, 0]])

    P_t = np.array([[1.0, 0, 0, 1],
                    [0, 1.0, 0, 1],
                    [0, 0, 1.0, 0]])

    # Compute 3D points
    p_0 = np.array([query_kp[match.queryIdx].pt for match in matches])
    p_1 = np.array([train_kp[match.trainIdx].pt for match in matches])
    points_3d = cv.triangulatePoints(P_q, P_t, p_0.T, p_1.T)

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_3d[0], points_3d[1], points_3d[2])
    plt.show()
