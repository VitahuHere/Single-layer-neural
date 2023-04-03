import csv
import string
import numpy as np
import re


def clean(text: str):
    text = text.lower()
    return re.sub(r'[^a-z]', '', text)


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
    return (
        1 if np.sum(weight_vector * input_vector) >= theta else 0
    )


def main():
    reader = csv.reader(open('lang.train.csv', 'r', encoding='utf-8'))
    data = np.array([[row[0], clean(row[1])] for row in reader])
    classes = {row: np.random.random_sample(26) for row in np.unique(data[:, 0])}
    theta_mapper = {list(classes.keys())[i]: i for i in range(len(classes))}
    alpha = 0.1
    theta = np.random.random_sample(len(classes))
    for row in data:
        input_vector = create_char_count_list(row[1])
        input_vector = normalize_vector(input_vector)
        d = row[0]
        y = perceptron(classes[d], theta[theta_mapper[d]], input_vector)


if __name__ == '__main__':
    main()
