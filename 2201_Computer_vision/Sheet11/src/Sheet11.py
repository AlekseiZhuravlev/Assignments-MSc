import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawEpipolar(im1, im2, corr1, corr2, fundMat):
    ## Insert epipolar lines
    print("Drawing epipolar lines")
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), cv2.waitKey(0), cv2.destroyAllWindows()
    return


def display_correspondences(im1, im2, corr1, corr2):
    ## Insert correspondences
    print("Display correspondences")
    cv2.imshow('Image 1', im1), \
    cv2.imshow('Image 2', im2), cv2.waitKey(0), cv2.destroyAllWindows()
    return


def computeFundMat(im1, im2, corr1, corr2):
    fundMat = np.zeros((3, 3))

    return fundMat


def question_q1_q2(im1, im2, correspondences):
    ## Compute and print Fundamental Matrix using the normalized corresponding points method.
    ## Display corresponding points and Epipolar lines
    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]

    print("Compute Fundamental Matrix")
    fundMat = computeFundMat(im1.copy(), im2.copy(), corr1, corr2)
    display_correspondences(im1.copy(), im2.copy(), corr1, corr2)
    drawEpipolar(im1.copy(), im2.copy(), corr1, corr2, fundMat)
    return


def question_q3(im1, im2):
    dispar = np.zeros((im1.shape[0], im1.shape[1]))
    ## compute disparity map
    print("Compute Disparity Map")

    img1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    window_size = 30
    for i in range(0, img2_gray.shape[0] - window_size, 3):
        for j in range(0, img2_gray.shape[1] - window_size, 3):
            patch = img2_gray[i: i + window_size, j: j + window_size]
            result = cv2.matchTemplate(img1_gray, patch, cv2.TM_SQDIFF)
            cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
            #
            best_match = np.argmin(result[:, j])

            dispar[i: i + 3, j: j + 3] = best_match - i

            # im_1_draw = im1.copy()
            # im_2_draw = im2.copy()

            # matchLoc = (best_match, j)
            # cv2.rectangle(im_1_draw, matchLoc, (matchLoc[0] + patch.shape[0], matchLoc[1] + patch.shape[1]), (255, 0, 0), 2)
            # cv2.rectangle(im_2_draw, (i, j), (i + window_size, j + window_size), (255, 0, 0), 2)
            # cv2.rectangle(result, matchLoc, (matchLoc[0] + patch.shape[0], matchLoc[1] + patch.shape[1]), (255, 0, 0), 2)
            # cv2.imshow('image_where_searching', im_1_draw)
            # cv2.imshow('image_with_patch', im_2_draw)
            # cv2.imshow('result_window', result)
            # cv2.waitKey(0), cv2.destroyAllWindows()

    # cv2.normalize(dispar, dispar, 0, 1, cv2.NORM_MINMAX, -1)

    # Display disparity Map
    cv2.imshow('Image 1', im1)
    cv2.imshow('Image 2', im2)
    # cv2.imshow('Disparity Map', dispar), cv2.waitKey(0), cv2.destroyAllWindows()

    plt.imshow(dispar, 'gray')
    plt.show()

    return dispar


def calculate_z(A, B):
    U, eigens_A, _ = np.linalg.svd(A)
    D = (U @ np.diag(np.sqrt(eigens_A))).T

    D_inv = np.linalg.inv(D)
    eigenvalues, eigenvectors = np.linalg.eig(D_inv.T @ B @ D_inv)

    # eigenvector corresponding to largest eigenvalue
    y = eigenvectors[:, 0]

    return D_inv @ y


def question_q4(img1, img2, correspondences):
    corr1 = correspondences[:, :2]
    corr2 = correspondences[:, 2:]
    ## Perform Image rectification

    ### usage of either one is permitted
    print("Fundamental Matrix")

    fund_mat = np.asmatrix([[-1.78999e-7, 5.70878e-6, -0.00260653],
                            [-5.71422e-6, 1.63569e-7, -0.0068799],
                            [0.00253316, 0.00674493, 0.191989]])

    ## Compute Rectification or Homography
    print("Compute Rectification")

    w = img1.shape[1]
    h = img1.shape[0]

    pc = np.array([(w - 1) / 2, (h - 1) / 2, 1])

    P_PT = w * h / 12 * np.array([
        [w ** 2 - 1, 0, 0],
        [0, h ** 2 - 1, 0],
        [0, 0, 0]
    ])

    pc_pcT = np.outer(pc, pc)

    _, _, V = np.linalg.svd(fund_mat)

    epipole_l = (V[2, :] / V[2, 0]).A1

    ex_vecprod = np.array([
        [0, -epipole_l[2], epipole_l[1]],
        [epipole_l[2], 0, -epipole_l[0]],
        [-epipole_l[1], epipole_l[0], 0]
    ])

    A = (ex_vecprod.T @ P_PT @ ex_vecprod)[:2, :2]
    A_ = (fund_mat.T @ P_PT @ fund_mat)[:2, :2]

    B = (ex_vecprod.T @ pc_pcT @ ex_vecprod)[:2, :2]
    B_ = (fund_mat.T @ pc_pcT @ fund_mat)[:2, :2]

    z_1 = calculate_z(A, B)
    z_2 = calculate_z(A_, B_)
    z_solution = (z_1 / np.sqrt(sum(z_1 ** 2)) + z_2 / np.sqrt(sum(z_2 ** 2))) / 2

    ## Apply Homography

    print("Display Warped Images")
    # cv2.imshow('Warped Image 1', im1), \
    # cv2.imshow('Warped Image 2', im2), cv2.waitKey(0), cv2.destroyAllWindows()
    return


def main():
    apt1 = cv2.imread('../images/apt1.jpg')
    apt2 = cv2.imread('../images/apt1.jpg')
    aloe1 = cv2.imread('../images/aloe1.png')
    aloe2 = cv2.imread('../images/aloe2.png')
    correspondences = np.genfromtxt('../images/corresp.txt', dtype=float, skip_header=1)

    # question_q1_q2(apt1,apt2,correspondences)
    # question_q3(aloe1,aloe2)
    question_q4(apt1, apt2, correspondences)


if __name__ == '__main__':
    main()
