# Image Classifier

Image classification is the task of assigning an input image one label from a fixed set of categories. This is an **Image Classifier** that can be used to classify images based on whether they contain a certain object or not. Image Classification is one of the core problems in **Computer Vision** that, despite its simplicity, has a large variety of practical applications.

I've implemented three different versions of **Image Classifiers** using common Deep Learning Libraries (namely *TensorFlow*, *Keras* & *Numpy*) for the same task in order to gain a deeper understanding of the problem and the backend working of the frameworks and their performance.

## Dataset

For the purpose of this project, I'll be using the [Cat Image Dataset](https://www.kaggle.com/crawford/cat-dataset) from [Kaggle](https://www.kaggle.com/) that includes over 9,000 images of cats which is sufficient for a binary classification task. Download the images into `datasets/images/` dir and create a dir `models/` for saving and checkpointing models during training.

### Getting Started

You'll need to preprocess the images using:
```
python prepro.py --dataset-path datasets/images
```

Now you can run the train script using:
```
python tf_clf.py --num-epochs 10 --batch-size 64 -lr 0.001 --tensorboard-vis
```

Passing the `--tensorboard-vis` flag enables you to visualize the training loss and accuracies in your browser using the following command:
```
tensorboard --log-dir=./logs
```

## Built With

* Python
* TensorFlow
* Keras
* NumPy
* h5py
* tqdm