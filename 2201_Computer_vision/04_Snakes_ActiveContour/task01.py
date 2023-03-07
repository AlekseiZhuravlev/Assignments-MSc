import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0, :]).reshape((-1, 2))
    ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
    ax.scatter(V[:, 0], V[:, 1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 10  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------

class SnakeContours:

    def __init__(self, img, snakes):
        self.img = img
        self.snakes = snakes

        self.grad_img_x, self.grad_img_y = self.calculate_gradient_image()

    def calculate_gradient_image(self):
        sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3).astype(int)
        sobel_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3).astype(int)
        return sobel_x, sobel_y

    def get_external_energy(self, point):
        return -(self.grad_img_x[point[0], point[1]] ** 2 + self.grad_img_y[point[0], point[1]] ** 2)

    def get_internal_energy(self, point, prev_point, next_point, avg_dist):
        alpha = 1
        beta = 1
        elasticity = alpha * np.sum((avg_dist - np.linalg.norm(next_point - prev_point)) ** 2)
        curvature = beta * np.sum((next_point - 2 * point + prev_point) ** 2)
        return elasticity + curvature

    def make_list_of_possible_vertices(self):
        possible_vertices = []
        for vertex in self.snakes:
            directions = [vertex.copy()]
            if vertex[0] - 1 > 0:
                directions.append(
                    np.array([vertex[0] - 1, vertex[1]])
                )
            if vertex[1] - 1 > 0:
                directions.append(
                    np.array([vertex[0], vertex[1] - 1])
                )
            if vertex[0] + 1 < self.img.shape[0]:
                directions.append(
                    np.array([vertex[0] + 1, vertex[1]])
                )
            if vertex[1] + 1 < self.img.shape[1]:
                directions.append(
                    np.array([vertex[0], vertex[1] + 1])
                )

            if not directions:
                raise ValueError('A snake was empty')
            possible_vertices.append(directions)
        return possible_vertices

    def get_avg_snakes_distance(self):
        dist = 0
        for i in range(len(self.snakes) - 1):
            dist += np.sqrt(np.sum((self.snakes[i] - self.snakes[i + 1]) ** 2))
        dist += np.sqrt(np.sum((self.snakes[0] - self.snakes[-1]) ** 2))
        return dist / len(self.snakes)

    def update_snakes(self):
        possible_vertices = self.make_list_of_possible_vertices()
        avg_dist = self.get_avg_snakes_distance()
        sequences = [{'sequence': [], 'cost': 0} for _ in range(len(self.snakes))]

        for i, vertex in enumerate(self.snakes):
            vertex_options = possible_vertices[i]
            # print(vertex_options)

            new_sequences = []
            for sequence in sequences:
                for option in vertex_options:
                    # + first node case
                    prev_node = sequence['sequence'][-1] if sequence['sequence'] else self.snakes[-1]
                    # + last node case
                    next_node = self.snakes[i + 1].copy() if i + 1 < len(self.snakes) else self.snakes[0]

                    new_sequence = sequence.copy()
                    new_sequence['cost'] += self.get_internal_energy(option, prev_node, next_node, avg_dist)
                    new_sequence['cost'] += self.get_external_energy(option)

                    # print(self.get_internal_energy(option, prev_node, next_node, avg_dist), self.get_external_energy(option))

                    new_sequence['sequence'].append(option.copy())

                    new_sequences.append(new_sequence)

            # keep only 5 best sequences
            # print(new_sequences)
            new_sequences = sorted(new_sequences.copy(), key=lambda d: d['cost'], reverse=True)
            sequences = new_sequences[:5].copy()
            print(len(sequences[0]))
        print(len(sequences))
        sequences = sorted(sequences.copy(), key=lambda d: d['cost'], reverse=True)
        self.snakes = np.array(sequences[0]['sequence'].copy())
        print(sequences[0]['cost'], len(self.snakes))


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 205

    # ------------------------
    # your implementation here

    # ------------------------
    snake_kernel = SnakeContours(Im, V)

    for t in range(n_steps):
        # ------------------------
        # your implementation here

        # ------------------------
        snake_kernel.update_snakes()

        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, snake_kernel.snakes)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
