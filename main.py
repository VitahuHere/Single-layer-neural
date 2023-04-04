import csv
import string
import numpy as np
import re


def clean(text: str):
    text = text.lower()
    return re.sub(r"[^a-z]", "", text)


def normalize_vector(vector: np.array) -> np.array:
    return vector / np.linalg.norm(vector)


def create_char_count_list(text) -> np.array:
    letter_map = {}
    for letter in string.ascii_lowercase:
        letter_map[letter] = 0
    for letter in text:
        letter_map[letter] += 1
    return np.array([letter_map[letter] for letter in string.ascii_lowercase])


def modify_weight_vector(
    weight_vector: np.array,
    alpha: float,
    d: float,
    y: float,
    input_vector: np.array,
) -> list:
    multiplier = alpha * (d - y)
    new_input = input_vector * multiplier
    return weight_vector + new_input


def modify_theta(old_theta: float, alpha: float, d: float, y: float) -> float:
    return old_theta - alpha * (d - y)


def perceptron(
    weight_vector: np.array,
    theta: float,
    input_vector: np.array,
) -> float:
    return 1 if np.sum(weight_vector * input_vector) >= theta else 0


def train_layer(row, classes, theta_mapper, theta):
    input_vector = create_char_count_list(row[1])
    input_vector = normalize_vector(input_vector)
    d = np.array([1 if row[0] == key else 0 for key in classes.keys()])
    y = np.array(
        [
            perceptron(classes[key], theta[theta_mapper[key]], input_vector)
            for key in classes.keys()
        ]
    )
    return d, y, input_vector


def modify_params(classes, theta_mapper, theta, alpha, d, y, input_vector):
    for key in classes.keys():
        classes[key] = modify_weight_vector(
            classes[key],
            alpha,
            d[theta_mapper[key]],
            y[theta_mapper[key]],
            input_vector,
        )
    theta = np.array(
        [modify_theta(theta[i], alpha, d[i], y[i]) for i in range(len(classes))]
    )
    return classes, theta


def test_layer(i, classes, theta_mapper, theta, test_data, data, alpha):
    correct = 0
    for r in test_data:
        input_vector = create_char_count_list(r[1])
        input_vector = normalize_vector(input_vector)
        y = np.array(
            [
                perceptron(classes[key], theta[theta_mapper[key]], input_vector)
                for key in classes.keys()
            ]
        )
        if r[0] == list(classes.keys())[np.argmax(y)]:
            correct += 1
    with open(f"results_{alpha}.csv", "a") as f:
        f.write(f"{i};{correct / len(test_data) * 100}\n")


def main():
    reader = csv.reader(open("lang.train.csv", "r", encoding="utf-8"))
    data = np.array([[row[0], clean(row[1])] for row in reader])
    test_reader = csv.reader(open("lang.test.csv", "r", encoding="utf-8"))
    test_data = np.array([[row[0], clean(row[1])] for row in test_reader])
    alpha = 0.1
    for _ in range(3):
        classes = {row: np.random.random_sample(26) for row in np.unique(data[:, 0])}
        theta_mapper = {list(classes.keys())[i]: i for i in range(len(classes))}
        theta = np.random.random_sample(len(classes))
        for i in range(1, 5001):
            for row in data:
                d, y, input_vector = train_layer(row, classes, theta_mapper, theta)
                classes, theta = modify_params(classes, theta_mapper, theta, alpha, d, y, input_vector)
            test_layer(i, classes, theta_mapper, theta, test_data, data, alpha)
        alpha /= 10


if __name__ == "__main__":
    main()
