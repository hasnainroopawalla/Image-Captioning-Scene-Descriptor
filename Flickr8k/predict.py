from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding,BatchNormalization, Dropout, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Merge    # Keras 2.1.6
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing import image
import keras
from keras import backend 
from keras.models import load_model
import time
from PIL import Image
from keras.models import model_from_json

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x