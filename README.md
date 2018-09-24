# Image Classifier

Image classification is the task of assigning an input image one label from a fixed set of categories. This is an **Image Classifier** that can be used to classify images based on whether they contain a certain object or not. Image Classification is one of the core problems in *Computer Vision* that, despite its simplicity, has a large variety of practical applications.

I have implemented *3* different versions of **Image Classifiers** using common Deep Learning Libraries (namely TensorFlow, Keras & Numpy) for the same task in order to gain a deeper understanding of the problem and the backend working of the frameworks and how their performance compares when analysed.

## Dataset

For the purpose of this project, I'll be using the [Cat Image Dataset](https://www.kaggle.com/crawford/cat-dataset) from [Kaggle](https://www.kaggle.com/) that includes over 9,000 images of cats which is sufficient for a binary classification task. Download the images into `datasets/images/` dir and create a dir `models/` for saving and checkpointing models during training.

### Getting Started

You'll need to preprocess the images using:
```
python preprocess.py
```

Now you can run any train script version you want:
```
python <lib>_clf.py
```

## Built With

* Python
* TensorFlow
* Keras
* NumPy
* h5py
* tqdm