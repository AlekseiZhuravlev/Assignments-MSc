import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt


# --- this are all the imports you are supposed to use!
# --- please do not add more imports!

def plot_M_S(M, S, title):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(M[:, 0], M[:, 1], M[:, 2])
    ax.set_title(f'Motion - {title}')

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(S.T[:, 0], S.T[:, 1], S.T[:, 2])
    ax.set_title(f'Shape - {title}')

    plt.show()


if __name__ == '__main__':
    n_views = 101
    n_features = 215

    with open('data/data_matrix.txt', 'r') as f:
        lines = f.readlines()

    data = [[float(coordinate) for coordinate in line.split(' ')] for line in lines]
    data = np.matrix(data)

    # Structure-from-Motion
    u, s, vh = np.linalg.svd(data)

    u_3 = u[:, :3]
    w_3 = np.diag(s[:3])
    v_3 = vh.T[:, :3]

    M = u_3
    S = w_3 @ v_3.T

    plot_M_S(M, S, 'ambiguity')

    # affine ambiguity
    # see slide 14
    # http://cg.elte.hu/~hajder/vision/slides/lec03_multicamera.pdf

    A = np.zeros((3 * n_views, 6))
    B = np.zeros((3 * n_views, 1))

    for i in range(n_views):
        m_1 = M[i * 2].A1
        m_2 = M[i * 2 + 1].A1

        A[i * 3] = np.array([m_1[0] ** 2, 2 * m_1[0] * m_1[1], 2 * m_1[0] * m_1[2],
                             m_1[1] ** 2, 2 * m_1[1] * m_1[2], m_1[2] ** 2])
        A[i * 3 + 1] = np.array([m_2[0] ** 2, 2 * m_2[0] * m_2[1], 2 * m_2[0] * m_2[2],
                                 m_2[1] ** 2, 2 * m_2[1] * m_2[2], m_2[1] ** 2])
        A[i * 3 + 2] = np.array([m_1[0] * m_2[0], m_1[1] * m_2[0] + m_1[0] * m_2[1], m_1[2] * m_2[0] + m_1[0] * m_2[2],
                                 m_1[1] * m_2[1], m_1[2] * m_2[1] + m_1[1] * m_2[2], m_1[2] * m_2[2]])

        B[i * 3] = 1
        B[i * 3 + 1] = 1
        B[i * 3 + 2] = 0

    X = la.lstsq(A, B)[0].T[0]

    L = np.matrix([
        [X[0], X[1], X[2]],
        [X[1], X[3], X[4]],
        [X[2], X[4], X[5]]
    ])

    U_, S_, Vt_ = la.svd(L)
    C = U_ @ np.sqrt(np.diag(S_))
    # print(U @ np.sqrt(np.diag(S)), '\n', (np.sqrt(np.diag(S)) @ Vt).T)

    M_adj = M @ C
    S_adj = la.inv(C) @ S

    plot_M_S(M_adj, S_adj, 'no ambiguity')

    # reprojection error
    data = data.A

    plt.ion()
    plt.show()

    for i in range(n_views):
        plt.clf()

        img = cv2.imread(f'data/frame{i + 1:08}.jpg')
        plt.imshow(img)
        plt.scatter(data[2 * i], data[2 * i + 1], marker='.')

        points_2d = np.zeros((n_features, 2))
        # print(M_adj.shape)
        for j in range(n_features):
            pts = np.array([M_adj.A[2 * i], M_adj.A[2 * i + 1]])
            shape = S_adj.T[j].A1
            points_2d[j] = shape @ pts.T

        plt.scatter(points_2d[:, 0], points_2d[:, 1], facecolors='none', edgecolors='r')

        plt.draw()
        plt.pause(0.01)
