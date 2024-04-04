import os

def load_class_dict(classes_path):
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj, classes_path):
    folder = os.path.dirname(classes_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(classes_path, 'w') as file:
        json.dump(obj, file, indent=2)