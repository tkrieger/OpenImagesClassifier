""" Configuration for setup and classification
"""

DATA_DIRECTORY = './data/'

SUMMARY_DIR = './log'
MODEL_SAVE_DIR = "./models"

THREAD_COUNT = 10

TRAINED_MODEL = {'name': 'ResNet_v1_50',
                 'url': 'https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1'}

DATABASE = {
    'filename': DATA_DIRECTORY + 'open_images.db'
}

CSV_FILES = {
    'Images': [
        {'file_path': DATA_DIRECTORY + '2017_11/train/images.csv'},
        {'file_path': DATA_DIRECTORY + '2017_11/test/images.csv'},
        {'file_path': DATA_DIRECTORY + '2017_11/validation/images.csv'}
    ],
    'Labels': [
        {'file_path': DATA_DIRECTORY + '2017_11/train/annotations-human.csv'},
        {'file_path': DATA_DIRECTORY + '2017_11/test/annotations-human.csv'},
        {'file_path': DATA_DIRECTORY + '2017_11/validation/annotations-human.csv'}
    ],
    'Dict': [
        {'file_path': DATA_DIRECTORY + '2017_11/class-descriptions.csv'}
    ]
}

DOWNLOAD_PATHS = {
    'https://storage.googleapis.com/openimages/2017_11/images_2017_11.tar.gz',
    'https://storage.googleapis.com/openimages/2017_11/annotations_human_2017_11.tar.gz',
    'https://storage.googleapis.com/openimages/2017_11/classes_2017_11.tar.gz'
}

CATEGORIES = [
    'Person',
    'Plant',
    'Animal',
    'Car',
    'Building',
    'Laptop',
    'Book',
    'Furniture',
    'Drink',
    'Snack'
]

TRAIN_DATASET = {
    'images_per_class': 8000
}

TEST_DATASET = {
    'images_per_class': 2000
}

VALID_DATASET = {
    'images_per_class': 2000
}



