import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    # TASK 1
    data = read_training_data('data/task1/hands_aligned_train.txt')
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

    # TASK 2
    # the code should be correct, but the eigenvectors are probably skewed...

    data = read_training_data('data/task1/hands_aligned_test.txt')
    y = data.flatten()

    weights = np.zeros(5)
    components = np.array([np.sqrt(eigenvalues[i]) * eigenvectors[i] for i in range(len(eigenvalues))])

    for i in range(5):
        x = mean_hand + np.sum(np.dot(weights, eigenvectors), axis=0)

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
