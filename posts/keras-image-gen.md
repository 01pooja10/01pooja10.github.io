Data augmentation is a very useful part of building a machine learning model, specifically with respect to image data. Augmented images (zoomed/rotated/flipped), many a time, increase a given model's performance and accuracy by helping it generalize better. Techniques like data augmentation, rescaling or resizing etc., are often used and can change the way in which our model learns the features of the input images while training.

Image data generator is a magical functionality from Python's deep learning API, Keras. Since it is a pretty underrated and misunderstood functionality in terms of its applications and usage, I'd like to walk you all through the process of using it to manipulate your image datasets and augment pictures.

Let's first clear a common misconception regarding the image data generator; though the term *data augmentation* is always used while referring to the Image data generator function, **it actually doesn't involve "creating" or adding extra images to the dataset**.

> Instead, it replaces the original training images with augmented versions of the same, while the model is trained on them. During each epoch, variations of the original image are passed through the layers of the model.

In this blog, we'll be using the Image Data Generator functionality from Keras to handle our input data and finally build a CNN model. We will see how it does literally everything for us in a few lines of code and minimizes the complexity of accessing our data.

We will be working with the *food image classification* dataset from Kaggle which can be found [here](https://www.kaggle.com/trolukovich/food5k-image-dataset). It contains 3 folders, namely — training, validation and evaluation/testing, each of which contains images of food and non-food items.

---

## Let's begin augmenting

The Image Data Generator uses various augmentation techniques to modify our input images, by providing parameters that we can tweak. Some of the significant parameters:

1. **Rotations and zoom**: Used to rotate our image by any angle between 0 and 360 degrees. When this is performed, the pixels along the edge of the part of the image that gets rotated will disappear and the image gets tilted. The zoom feature is used for zooming into the image.

2. **Horizontal or vertical flips**: Flipping actions to tilt the images horizontally or vertically. They can be set to `True` or `False`.

3. **fill_mode**: Can be set to fill in the empty pixels which were left behind due to rotation or flipping actions. The empty pixels can be filled with either `constant`, `nearest`, `reflect` or `wrap` options.

4. **rescale**: Helps us rescale or normalize an image's pixels from a range of 0-255 to values between 0.0 and 1.0. This is because any RGB image (red, green or blue) is usually 8 bits and thus is limited to a range of 0-255.

5. **Height and width shift**: These 2 parameters contribute to the vertical or horizontal shifting of the image. Note: setting these parameters to true will only shift them in the X-Y plane.

The `datagen.flow()` method takes in an image and generates batches of augmentations on it. The augmented images are obviously different from the original version — the distortion in the edges of the pictures is due to the fill_mode parameter being set to the `nearest` configuration. The pictures are variations of the same picture and this means that during each epoch, one of these distorted or augmented images will be sent through each layer of our neural network instead of the original image.

Consequently, the model receives different versions of the original image, each differing in rotation angle, zoom, etc during each iteration.

> This also prevents overfitting as the model is trained on different versions of the image every time it is passed through different layers of the model.

---

## Train and Validation/Test sets

Now let's create our data generator variable for the training, validation and test sets of the food image classification dataset.

In case we don't have a separate folder for validation/testing data, a parameter called **validation_split**, while creating the data generator variable, can be set to a value between 0.1 and 1.0 (ideally < 0.5). After creating the train_set variable, a separate validation_set or test_set variable must be created with the **subset** parameter set to `'validation'`. But in our case, since we have 3 separate folders for training, validation and testing purposes, we won't require a validation split.

Also, note that **only the training images can contain augmentations** and not the images used for validation/testing. So we add flips, zoom and rotations to the training data whereas we simply rescale the validation and testing data.

> Found 3000 images belonging to 2 classes.
> Found 1000 images belonging to 2 classes.
> Found 1000 images belonging to 2 classes.

The 3000, 1000 and 1000 images belong to the train_set, val_set and test_set respectively.

---

## Loading the images from their respective sub-directories

Notice that here we are using the `datagen.flow_from_directory()` method instead of `datagen.flow()`. This directly takes the path to our folder containing the images as the input and retrieves images from the path (folders) given. It then returns a **directory iterator** of the form `(x, y)` where x is a numpy array of batches of augmented images and y is a numpy array of the respective labels.

Since we set the `class_mode` parameter to `'binary'`, the datagen, while retrieving images from each folder, automatically labels them into food and non-food classes depending on which folders they were stored in. Also, the food and non-food categories are automatically one-hot encoded to 1s and 0s for our convenience — food items have been set to 0 and non-food items have been set to 1.

Thus, the class mode, when set to binary, ensures that the encoded labels exist in a 1D array as 1s and 0s. It can be set to `'categorical'` in case the number of classes is more than 2 and the labels will be stored in a 2D array. It can also be set to `'sparse'` and the labels will be stored in a 1D array as integer labels.

> The image data generator also supports the `flow_from_dataframe()` method which takes a pandas dataframe as the input or a path to a directory containing the dataframe and similarly, generates batches of data.

So now, the image data generator basically did the entire categorization, augmentation, rescaling and one-hot encoding of labels in a few lines of code each, for training and validation/testing purposes.

You can find the documentation for the image data generator [here](https://keras.io/api/preprocessing/image/).

---

## The model

A convolutional neural network is used extensively in image-related deep learning tasks. We shall employ a simple CNN model to deal with our dataset. The model architecture:

- **3 Conv2D layers** employing [ReLU](https://keras.io/api/layers/activation_layers/relu/) activation function with [batch normalization](https://keras.io/api/layers/normalization_layers/batch_normalization/), [MaxPooling2D](https://keras.io/api/layers/pooling_layers/max_pooling2d/) and [dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) regularization
- These layers are passed on to **2 fully connected dense layers** of 256 and 512 units each
- Connected to an **output layer** with 1 unit and a [sigmoid](https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid) activation function for classifying the inputs into food (0) and non-food (1) items
- The [ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau) callback tool is used to reduce the learning rate each time a metric (validation accuracy, in our case) stops improving

The model, compiled using the [adam](https://keras.io/api/optimizers/adam/) optimizer, was trained for 10 epochs with a batch size of 64. The loss was measured using [binary_crossentropy](https://keras.io/api/losses/probabilistic_losses/#binary_crossentropy-function) and the train_set (training directory) was passed as the input to the `model.fit_generator()` method.

The `model.fit_generator()` is specifically used for fitting a model to the training directory iterators with augmented batches of data. We'd need to pass train_set as the input, since it contains both images and labels. We can set the number of epochs, validation data, callbacks and other parameters according to our preferences.

---

## Model evaluation

The model gave out a **validation accuracy of 83.4%**. Further, the model was evaluated on the test set using `model.evaluate_generator()` and it churned out a **testing accuracy of 81.7%**. This implies that our model is now ready for making predictions. This can be done using the `model.predict_generator()` method.

---

## Conclusion

In this blog we covered the major aspects involved in handling image data entirely using Keras' image data generator, to build a model for classifying images into the food and non-food categories. Our model can now make predictions on real world input data as well. Thus, we successfully created and evaluated a CNN model entirely using the image data generator functionality to load, modify, rescale and augment our images.
