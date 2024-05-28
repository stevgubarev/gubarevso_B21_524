import csv
import math
import numpy as np
from PIL import Image

RUSSIAN_LETTERS_UNICODE = ["0410", "0411", "0412", "0413", "0414", "0415", "0416", "0417", "0418", "0419", "041A",
                           "041B", "041C", "041D", "041E", "041F", "0420", "0421", "0423", "0424", "0425", "0426",
                           "0427", "0428", "0429", "042A", "042B", "042C", "042D", "042E", "042F", "0422"]
RUSSIAN_LETTERS = [chr(int(letter, 16)) for letter in RUSSIAN_LETTERS_UNICODE]
WHITE = 255
PHRASE = "ПРИВЕТ ВИДЕЛ ТЕБЯ НА ФИЗРЕ ДАВАЙ ПОЗНАКОМИМСЯ".replace(" ", "")


def calculate_features(img: np.array):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != WHITE] = 1  # Assuming white pixel value is 255

    # Calculate weight
    weight = np.sum(img_b)

    # Calculate center of mass
    y_indices, x_indices = np.indices(img_b.shape)
    y_center_of_mass = np.sum(y_indices * img_b) / weight
    x_center_of_mass = np.sum(x_indices * img_b) / weight

    # Calculate inertia
    inertia_x = np.sum((y_indices - y_center_of_mass) ** 2 * img_b) / weight
    inertia_y = np.sum((x_indices - x_center_of_mass) ** 2 * img_b) / weight

    return weight, x_center_of_mass, y_center_of_mass, inertia_x, inertia_y


def segment_letters(img):
    """ Находит начало и конец каждой буквы. """
    profile = np.sum(img == 0, axis=0)

    in_letter = False
    letter_bounds = []

    for i in range(len(profile)):
        if profile[i] > 0:
            if not in_letter:
                in_letter = True
                start = i
        else:
            if in_letter:
                in_letter = False
                end = i
                letter_bounds.append((start - 1, end))

    if in_letter:
        letter_bounds.append((start, len(profile)))

    return letter_bounds


def get_alphabet_info() -> dict[chr, tuple]:
    def parse_tuple(string):
        """Преобразует строку вида '(x, y)' в кортеж (x, y)."""
        return tuple(map(float, string.strip('()').split(',')))

    tuples_list = dict()
    with open('input/data1.csv', 'r') as file:
        reader = csv.DictReader(file)
        i = 0
        for row in reader:
            weight = int(row['weight'])
            center_of_mass = parse_tuple(row['center_of_mass'])
            inertia = parse_tuple(row['inertia'])
            tuples_list[RUSSIAN_LETTERS[i]] = weight, *center_of_mass, *inertia
            i += 1
    return tuples_list


def create_hypothesis(alphabet_info: dict[chr, tuple], target_features):
    def euclidean_distance(feature1, feature2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(feature1, feature2)))

    """ Возвращает список кортежей (буква, мера близости) в порядке убывания близости. """
    distances = dict()
    for letter, features in alphabet_info.items():
        distance = euclidean_distance(target_features, features)
        distances[letter] = distance

    max_distance = max(distances.values())

    similarities = [(letter, round(1 - distance / max_distance, 2)) for letter, distance in distances.items()]

    return sorted(similarities, key=lambda x: x[1])


def get_phrase_from_hypothesis(img: np.array, bounds) -> str:
    alphabet_info = get_alphabet_info()
    res = []
    for start, end in bounds:
        letter_features = calculate_features(img[:, start: end])
        hypothesis = create_hypothesis(alphabet_info, letter_features)
        best_hypotheses = hypothesis[-1][0]
        res.append(best_hypotheses)
    return "".join(res)


def write_res(recognized_phrase: str):
    max_len = max(len(PHRASE), len(recognized_phrase))
    orig = PHRASE.ljust(max_len)
    detected = recognized_phrase.ljust(max_len)

    with open("output/result.txt", 'w') as f:
        correct_letters = 0
        by_letter = ["has | got | correct"]
        for i in range(max_len):
            is_correct = orig[i] == detected[i]
            by_letter.append(f"{orig[i]}\t{detected[i]}\t{is_correct}")
            correct_letters += int(is_correct)
        f.write("\n".join([
            f"phrase:      {orig}",
            f"detected:    {detected}",
            f"correct:     {math.ceil(correct_letters / max_len * 100)}%\n\n"
        ]))
        f.write("\n".join(by_letter))


if __name__ == "__main__":
    """ 7 Лабораторная работа """
    img = np.array(Image.open(f'input/original_phrase.bmp').convert('L'))
    bounds = segment_letters(img)
    recognized_phrase = get_phrase_from_hypothesis(img, bounds)
    write_res(recognized_phrase)