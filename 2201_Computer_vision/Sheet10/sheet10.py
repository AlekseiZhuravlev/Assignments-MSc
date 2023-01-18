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

    img_keypoints = cv.drawKeypoints(img, kp, query_img)
    # display_image('sift_keypoints', img_keypoints)

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
        query_img, query_kp, train_img, train_kp, matches[:200],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Projection matrix for query_img and train_img
    P_q = np.array([[1.0, 0, 0, 0],
                    [0, 1.0, 0, 0],
                    [0, 0, 1.0, 0]])

    P_t = np.array([[1.0, 0, 0, 1],
                    [0, 1.0, 0, 1],
                    [0, 0, 1.0, 0]])

    # Compute 3D points
    print('Triangulation')

    for i, match in enumerate(matches[:5]):
        p0 = query_kp[match.queryIdx].pt
        p1 = train_kp[match.trainIdx].pt

        mat_0 = np.array([
            [0, -1, p0[1]],
            [1, 0, -p0[0]],
            [-p0[1], p0[0], 0],
        ])
        mat_1 = np.array([
            [0, -1, p1[1]],
            [1, 0, -p1[0]],
            [-p1[1], p1[0], 0]
        ])

        combined = np.concatenate([mat_0 @ P_q, mat_1 @ P_t])
        solution = np.linalg.lstsq(combined[:, :-1], -combined[:, -1], rcond=None)[0]

        print(
            f"""
            {i}) 2d points:
            {p0}
            {p1}
            3d point:
            {solution}"""
        )

    # Visualization
    plt.imshow(img_matches)
    plt.show()
