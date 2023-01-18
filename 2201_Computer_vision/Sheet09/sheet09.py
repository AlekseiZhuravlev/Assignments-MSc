import os

import numpy as np
import cv2 as cv


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename
    flo_file = open(filename, 'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25, 'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2 * w[0] * h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow


class OpticalFlow:
    def __init__(self):
        # Parameters for Lucas_Kanade_flow()
        self.EIGEN_THRESHOLD = 0.01  # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade
        self.WINDOW_SIZE = [25, 25]  # the number of points taken in the neighborhood of each pixel

        # Parameters for Horn_Schunck_flow()
        self.EPSILON = 0.002  # the stopping criterion for the difference when performing the Horn-Schuck algorithm
        self.MAX_ITERS = 1000  # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
        self.ALPHA = 1.0  # smoothness term

        # Parameter for flow_map_to_bgr()
        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        # calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1] - frames[0]

        return True

        # ***********************************************************************************

    # implement Lucas-Kanade Optical Flow
    # returns the Optical flow based on the Lucas-Kanade algorithm and visualisation result
    def Lucas_Kanade_flow(self):
        image = self.prev
        w1 = self.WINDOW_SIZE[0] // 2
        w2 = self.WINDOW_SIZE[1] // 2
        flow = np.zeros((image.shape[0], image.shape[1], 2))
        for i in range(w1, image.shape[0] - w1):
            for j in range(w2, image.shape[1] - w2):
                ix = self.Ix[i - w1:i + w1 + 1, j - w2:j + w2 + 1].flatten()
                iy = self.Iy[i - w1:i + w1 + 1, j - w2:j + w2 + 1].flatten()
                it = self.It[i - w1:i + w1 + 1, j - w2:j + w2 + 1].flatten()
                A = np.vstack((ix, iy)).T
                M = A.T @ A
                if np.min(abs(np.linalg.eigvals(M))) >= self.EIGEN_THRESHOLD:
                    b = np.reshape(it, (it.shape[0], 1))
                    d = np.linalg.inv(M) @ A.T @ b
                    flow[i, j, 0] = d[0]
                    flow[i, j, 1] = d[1]

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    # ***********************************************************************************
    # implement Horn-Schunck Optical Flow 
    # returns the Optical flow based on the Horn-Schunck algorithm and visualisation result
    def Horn_Schunck_flow(self):
        image = self.prev
        w1 = self.WINDOW_SIZE[0] // 2
        w2 = self.WINDOW_SIZE[1] // 2
        flow = np.zeros((image.shape[0], image.shape[1], 2))

        n_iterations = 0
        while True:
            flow_prev = flow.copy()

            u_bar = flow[:, :, 0] + cv.Laplacian(flow[:, :, 0], cv.CV_64F) / 4
            v_bar = flow[:, :, 1] + cv.Laplacian(flow[:, :, 1], cv.CV_64F) / 4

            flow[:, :, 0] = u_bar - self.Ix * (self.Ix * u_bar + self.Iy * v_bar + self.It) / \
                            (1 + self.Ix ** 2 + self.Iy ** 2)
            flow[:, :, 1] = v_bar - self.Iy * (self.Ix * u_bar + self.Iy * v_bar + self.It) / \
                            (1 + self.Ix ** 2 + self.Iy ** 2)

            # termination condition
            flow_difference = flow - flow_prev
            metric = np.abs(np.sum(flow_difference[:, :, 0])) + np.abs(np.sum(flow_difference[:, :, 1]))

            n_iterations += 1

            # metric is set to 2 instead of 0.002 to speed up the computation
            if metric < 2:
                break

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    # ***********************************************************************************
    # calculate the angular error here
    # return average angular error and per point error map
    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        aae_per_point = np.zeros(self.prev.shape)
        for i in range(self.prev.shape[0]):
            for j in range(self.prev.shape[1]):
                aae_per_point[i, j] = np.arccos((groundtruth_flow[i, j, 0] * estimated_flow[i, j, 0] + groundtruth_flow[
                    i, j, 1] * estimated_flow[i, j, 1] + 1) /
                                                np.sqrt((groundtruth_flow[i, j, 0] ** 2 + groundtruth_flow[
                                                    i, j, 1] ** 2 + 1) * (estimated_flow[i, j, 0] ** 2 + estimated_flow[
                                                    i, j, 1] ** 2 + 1)))
        aae = np.average(aae_per_point)
        return aae, aae_per_point

    # ***********************************************************************************
    # calculate the endpoint error here
    # return average endpoint error and per point error map
    def calculate_endpoint_error(self, estimated_flow, groundtruth_flow):
        aee_per_point = np.zeros(self.prev.shape)
        for i in range(self.prev.shape[0]):
            for j in range(self.prev.shape[1]):
                aee_per_point[i, j] = (groundtruth_flow[i, j, 0] - estimated_flow[i, j, 0]) ** 2 + (
                        groundtruth_flow[i, j, 1] - estimated_flow[i, j, 1]) ** 2
        aee = np.average(aee_per_point)
        return aee, aee_per_point

    # ***********************************************************************************
    # function for converting flow map to to BGR image for visualisation
    # return bgr image
    def flow_map_to_bgr(self, flow):
        hsv = np.zeros([self.prev.shape[0], self.prev.shape[1], 3])
        hsv[..., 1] = 255

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        flow_bgr = cv.cvtColor(hsv.astype('uint8'), cv.COLOR_HSV2BGR)

        return flow_bgr


def visualize_flow(title, flow):
    cv.imshow(title, flow)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":

    data_list = [
        'data/frame_0001.png',
        'data/frame_0002.png',
        'data/frame_0007.png',
    ]

    gt_list = [
        './data/frame_0001.flo',
        './data/frame_0002.flo',
        './data/frame_0007.flo',
    ]

    Op = OpticalFlow()

    for (i, (frame_filename, gt_filemane)) in enumerate(zip(data_list, gt_list)):
        groundtruth_flow = load_FLO_file(gt_filemane)
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        aee_lucas_kanade, aee_lucas_kanade_per_point = Op.calculate_endpoint_error(flow_lucas_kanade, groundtruth_flow)

        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow)
        aee_horn_schunk, aee_horn_schunk_per_point = Op.calculate_endpoint_error(flow_horn_schunck, groundtruth_flow)

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        # Implement vizualization below  
        # Your functions here

        visualize_flow('ground_truth', flow_bgr_gt)
        visualize_flow('lucas_kanade', flow_lucas_kanade_bgr)
        visualize_flow('horn_schunck', flow_horn_schunck_bgr)

        # Collect and display all the numerical results from all the runs in tabular form (the exact formating is up to your choice)
        print(f"""
        {"*" * 20}
        frame {i}
        
        Lukas_Kanade:
        average angular error: {aae_lucas_kanade}
        average endpoint error: {aee_lucas_kanade}
        
        Horn-Schunck:
        average angular error: {aae_horn_schunk}
        average endpoint error: {aee_horn_schunk}
        
        """)
