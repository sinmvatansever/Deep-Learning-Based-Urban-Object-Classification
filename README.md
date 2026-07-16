# Deep-Learning-Based-Urban-Object-Classification
A deep learning system for classifying cars, pedestrians, and bikes in urban road images. The project compares a custom CNN trained from scratch with a fine-tuned MobileNetV2 transfer learning model using the RAMI London dataset.



This project develops and compares two deep learning models for classifying real-world urban road objects into three categories: cars, pedestrians, and bikes.

The system uses images from the RAMI London urban dataset. Object sub-images are extracted from the original road scenes using Pascal VOC XML annotations for cars and YOLOv8 detections for pedestrians and bikes. Each detected object is cropped with additional padding, resized to 128×128 pixels, and stored as a PNG image in the corresponding training, validation, or test directory.

Two image classification approaches are implemented:

**Custom Convolutional Neural Network**

A lightweight CNN is designed and trained from scratch using three convolutional blocks. Each block applies convolution, batch normalisation, ReLU activation, and max pooling to progressively learn visual features.

The final classification layers use global average pooling, a dense layer, dropout, and a softmax output layer for three-class prediction.

**MobileNetV2 Transfer Learning**

The second model uses MobileNetV2 pre-trained on the ImageNet dataset. A custom classification head is added for urban object classification.

Training is performed in two stages:

Feature extraction, where the MobileNetV2 backbone is frozen and only the new classification layers are trained.
Fine-tuning, where the upper layers of MobileNetV2 are unfrozen and trained using a smaller learning rate.

Data augmentation techniques, including horizontal flipping, rotation, and zooming, are applied to improve model generalisation.

Training and Optimisation

**The project includes:**

Adam optimisation,
Categorical cross-entropy loss,
Early stopping,
Learning-rate reduction,
Model checkpointing,
Batch normalisation,
Dropout regularisation,
TensorFlow data caching and prefetching,
Reproducible random seeds,
Optional learning-rate hyperparameter testing,
Model Evaluation

**The Custom CNN and MobileNetV2 models are evaluated and compared using:**

Validation accuracy,
Test accuracy,
Precision and recall for each object class,
Classification reports,
Confusion matrices,
Training and validation loss curves,
Training and validation accuracy curves,
Learning-rate sweep results

The project demonstrates an end-to-end computer vision workflow covering object extraction, dataset preparation, CNN development, transfer learning, fine-tuning, model evaluation, and performance comparison.

Technologies,
Python,
TensorFlow and Keras,
MobileNetV2,
YOLOv8,
OpenCV,
NumPy,
Scikit-learn,
Matplotlib,
Seaborn,
Pascal VOC XML annotations
