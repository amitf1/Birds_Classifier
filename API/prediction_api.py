from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from wiki_bird import open_wiki
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
UPLOAD_FOLDER = 'data/Predictions/'
MODEL = load_model('new_model.h5')

encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy', allow_pickle=True)


def get_prediction(filename):
    model = MODEL
    image = load_img(UPLOAD_FOLDER+filename, target_size=(224, 224), interpolation='bicubic')
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image/255.0
    pred = model.predict(image)
    label = pred.argmax(axis=1)
    specie = encoder.inverse_transform(label)[0].decode('ascii')
    print(label, specie, pred[0, label][0]*100)
    open_wiki(specie)

    return specie, pred[0, label][0]*100
