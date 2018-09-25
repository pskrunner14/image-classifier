import os
import click
import logging

import h5py
import keras
import sklearn
import numpy as np
import keras_applications

from tqdm import tqdm
from sklearn.model_selection import train_test_split

resolution = 256

def preprocess_image(image):
    x = keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    return keras_applications.imagenet_utils.preprocess_input(x)[0]

def process_image_dataset(dataset_path):
    if dataset_path is None:
        raise UserWarning('Dataset path should not be None!')

    X = []
    images = os.listdir(dataset_path)

    for image_path in tqdm(os.listdir(dataset_path), total=len(images), desc='Processing Images'):
        image = keras.preprocessing.image.load_img('{}/{}'
            .format(dataset_path, image_path), 
            target_size=(resolution, resolution))
        X.append(preprocess_image(image))

    # convert to desired format
    X = np.array(X)
    y = np.ones((len(images), 1))
    
    logging.info('Features shape: {}'.format(X.shape))
    logging.info('Targets shape: {}'.format(y.shape))

    # randomly shuffle both arrays but in same order
    logging.info('Randomly shuffling arrays')
    X, y = sklearn.utils.shuffle(X, y, random_state=0)

    #divide into sets
    logging.info('Splitting into train, val and test datasets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    #write final
    logging.info('Writing preprocessed data to files')
    train_file = h5py.File('datasets/training_data.h5', "w")
    train_file.create_dataset('X_train', data=X_train)
    train_file.create_dataset('y_train', data=y_train)

    train_file = h5py.File('datasets/validation_data.h5', "w")
    train_file.create_dataset('X_val', data=X_val)
    train_file.create_dataset('y_val', data=y_val)

    test_file = h5py.File('datasets/testing_data.h5', "w")
    test_file.create_dataset('X_test', data=X_test)
    test_file.create_dataset('y_test', data=y_test)

@click.command()
@click.option('-ds', '--dataset-path', default='datasets/images', help='Path for your Image Dataset')
def main(dataset_path):
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO')
    process_image_dataset(dataset_path)
    logging.info('Done preprocessing!')

if __name__ == '__main__':
    main()