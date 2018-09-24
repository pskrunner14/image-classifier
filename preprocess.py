import os
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import h5py

resolution = 256

def process_image_dataset():
    X_new = []
    num_images = len(os.listdir('datasets/images'))

    for i in range(1, num_images + 1):
        img_path = 'datasets/images/{}.jpg'.format(i)
        img = image.load_img(img_path, target_size=(resolution, resolution))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X_new.append(x[0])

    X_new = np.array(X_new)
    Y_new = np.ones((num_images, 1))
    
    print(X_new.shape)
    print(Y_new.shape)

    #divide into sets
    x_train = X_new[:8000]
    x_test = X_new[8000:]
    y_train = Y_new[:8000]
    y_test = Y_new[8000:]

    #write final
    labels = np.array([b'non-cat', b'cat'])

    train_file = h5py.File('datasets/training_data.h5', "w")
    train_file.create_dataset('x_train', data=x_train)
    train_file.create_dataset('y_train', data=y_train)
    train_file.create_dataset('labels', data=labels)

    test_file = h5py.File('datasets/testing_data.h5', "w")
    test_file.create_dataset('x_test', data=x_test)
    test_file.create_dataset('y_test', data=y_test)
    test_file.create_dataset('labels', data=labels)

def main():
    process_image_dataset()

if __name__ == '__main__':
    main()