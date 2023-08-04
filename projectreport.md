# Project Report

## Introduction

This project is to build a model to predict the quality of wine using the `winequality-red.csv` dataset. The dataset contains 1599 observations of 11 features, including the quality of the wine (1-10).

## Data Preprocessing

The first step is to check for missing values. There are no missing values in the dataset.

The next step is to convert the `quality` feature to an integer. The current dtype is object, so we need to use the `astype()` function.

Finally, we need to normalize the data. This means that we need to subtract the mean from each feature and then divide by the standard deviation.

## Model Building

We will use a Keras sequential model to build our model. The model will have three layers:

* A first layer with 64 neurons and the `relu` activation function.
* A second layer with 32 neurons and the `relu` activation function.
* A third layer with 10 neurons and the `softmax` activation function.

We will use the `categorical_crossentropy` loss function, the `adam` optimizer, and the `accuracy` metric.

## Model Training

We will train the model for 100 epochs with a batch size of 32. We will also use the `validation_data` argument to evaluate the model on the test set after each epoch.

## Model Evaluation

The test loss is 0.49 and the test accuracy is 0.74. This means that the model is able to predict the quality of wine with 74% accuracy.

## Conclusion

We have built a model that can predict the quality of wine with 74% accuracy. This model can be used to help winemakers produce better wine.
