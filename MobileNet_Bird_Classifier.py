import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend
from tensorflow.keras.applications import MobileNet

BASE_DIR = 'data'
TRAIN_DIR = os.path.join(BASE_DIR, '200_species_train')
VALIDATION_DIR = os.path.join(BASE_DIR, '200_species_valid')
TEST_DIR = os.path.join(BASE_DIR, '200_species_test')


class MobileNetBirdClassifier:
    """
    Model for classifying birds' species from images based on mobilenet
    """
    def __init__(self):
        """
        initialize mobilenet model instance
        """
        self._general_datagen = ImageDataGenerator(rescale=1./255)
        self._categories = os.listdir(TRAIN_DIR)
        self._category_count = len(self._categories)
        im = load_img(f"{TRAIN_DIR}/CANARY/001.jpg")
        im = img_to_array(im)
        self._image_shape = im.shape
        self._model = self._build_model()

    def _create_datasets(self):
        """
        creates train validation and test datasets from the base directory
        :return: train validation and test keras datasets
        """
        train_data = self._general_datagen.flow_from_directory(TRAIN_DIR, target_size=(224, 224),batch_size=128)
        validation_data = self._general_datagen.flow_from_directory(VALIDATION_DIR,
                                                                    target_size=(224, 224), batch_size=128)
        test_data = self._general_datagen.flow_from_directory(TEST_DIR, target_size=(224, 224), batch_size=128)
        return train_data, validation_data, test_data

    def _build_model(self):
        """
        Builds and compiles mobilenet based keras model
        :return: compiled mobilenet keras model
        """
        backend.clear_session()
        base_mobilenet = MobileNet(weights='imagenet', include_top=False,
                                   input_shape=self._image_shape)
        base_mobilenet.trainable = False  # Freeze the mobilenet weights.
        model = Sequential()
        model.add(base_mobilenet)
        model.add(Flatten())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self._category_count))
        model.add(Activation('softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self):
        """
        Train the model on the training data and evaluate on the testing data
        """
        train_data, validation_data, test_data = self._create_datasets()
        history = self._model.fit(
            train_data,
            epochs=50,
            validation_data=validation_data,
            validation_steps=len(validation_data),
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_accuracy', patience=5,
                                     restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=2, verbose=1)])
        print('final validation accuracy', history.history['val_accuracy'])
        self._model.save('saved_model/model')
        self.evaluate(test_data)

    def evaluate(self, test_data):
        """
        Evaluates thr model's accuracy on the given test data
        :param test_data: image dataset to evaluate the model on
        """
        scores = self._model.evaluate(test_data, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def predict(self, data):
        """
        predicts bird species for the given images in the dataset
        :param data: images of birds to classify by species
        :return: array of species' numbers corresponding to the images
        """
        return self._model.predict(data)

    def save_model(self, path):
        """
        save the model to a file (xx.h5 preferred)
        :param path: path for the saved model
        :return: saved model file on given path
        """
        self._model.save(path)
