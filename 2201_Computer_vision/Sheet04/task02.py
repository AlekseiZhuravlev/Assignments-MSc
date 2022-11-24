import numpy as np
import cv2
import matplotlib.pyplot as plt


# rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75


def load_data(downsize_ratio=0.5):
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = int(70 * downsize_ratio / 0.5)
    Im = cv2.imread(fpath, 0).astype('float32') / 255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=downsize_ratio, fy=downsize_ratio)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]

    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1

    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# derivatives

def d_x(arr, i, j):
    if j - 1 >= 0 and j + 1 < arr.shape[1]:
        return (arr[i, j + 1] - arr[i, j - 1]) / 2
    else:
        return 0


def d2_x(arr, i, j):
    if j - 1 >= 0 and j + 1 < arr.shape[1]:
        return arr[i, j + 1] - 2 * arr[i, j] + arr[i, j - 1]
    else:
        return 0


def d_y(arr, i, j):
    if i - 1 >= 0 and i + 1 < arr.shape[0]:
        return (arr[i + 1, j] - arr[i - 1, j]) / 2
    else:
        return 0


def d2_y(arr, i, j):
    if i - 1 >= 0 and i + 1 < arr.shape[0]:
        return arr[i + 1, j] - 2 * arr[i, j] + arr[i - 1, j]
    else:
        return 0


def d_xy(arr, i, j):
    if i - 1 >= 0 and i + 1 < arr.shape[0] and j - 1 >= 0 and j + 1 < arr.shape[1]:
        return (arr[i + 1, j + 1] - arr[i + 1, j - 1] - arr[i - 1, j + 1] + arr[i - 1, j - 1]) / 4
    else:
        return 0


# ------------------------

class ActiveContours:
    def __init__(self, img, phi):
        self.img = img
        self.phi = phi

        self.omega = self.calculate_omega()

    def calculate_omega(self):
        omega = np.zeros(self.img.shape)

        for i in range(omega.shape[0]):
            for j in range(omega.shape[1]):
                gradient = np.array([
                    d_x(self.img, i, j),
                    d_y(self.img, i, j)
                ])
                epsilon = 1
                omega[i, j] = 1 / np.sqrt(np.sum(gradient ** 2) + epsilon)

        # plt.imshow(omega)
        # plt.show()
        return omega

    def update_phi(self):
        phi_upd = self.phi.copy()

        for i in range(self.phi.shape[0]):
            for j in range(self.phi.shape[1]):
                curvature_motion = self.get_curvature_motion(i, j)
                front_propagation = self.get_front_propagation(i, j)
                # print(curvature_motion, front_propagation)
                phi_upd[i, j] += curvature_motion + front_propagation
        self.phi = phi_upd

    def get_curvature_motion(self, i, j):
        upper_term = d2_x(self.phi, i, j) * d_y(self.phi, i, j) ** 2 - \
                     2 * d_x(self.phi, i, j) * d_y(self.phi, i, j) * d_xy(self.phi, i, j) + \
                     d2_y(self.phi, i, j) * d_x(self.phi, i, j) ** 2
        lower_term = d_x(self.phi, i, j) ** 2 + d_y(self.phi, i, j) ** 2 + 1
        return 0.1 * self.omega[i, j] * upper_term / lower_term

    def get_front_propagation(self, i, j):
        result = 0
        if j + 1 < self.phi.shape[1]:
            result += max(d_x(self.omega, i, j), 0) * (self.phi[i, j + 1] - self.phi[i, j])
        if j - 1 >= 0:
            result += min(d_x(self.omega, i, j), 0) * (self.phi[i, j] - self.phi[i, j - 1])
        if i + 1 < self.phi.shape[0]:
            result += max(d_y(self.omega, i, j), 0) * (self.phi[i + 1, j] - self.phi[i, j])
        if i - 1 >= 0:
            result += min(d_y(self.omega, i, j), 0) * (self.phi[i, j] - self.phi[i - 1, j])
        return result


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi_start = load_data(0.1)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    active_contours = ActiveContours(Im, phi_start)
    # exit(0)
    # ------------------------
    # your implementation here

    # ------------------------

    for t in range(n_steps):
        # print(f'iteration {t}')
        active_contours.update_phi()

        # ------------------------
        # your implementation here

        # ------------------------

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))

            contour = get_contour(active_contours.phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

            ax2.clear()
            ax2.imshow(active_contours.phi)
            ax2.set_title(r'phi', fontsize=22)
            plt.pause(0.01)

    plt.show()
