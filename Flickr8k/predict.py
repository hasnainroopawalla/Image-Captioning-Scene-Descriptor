# Keras 2.1.6

import pickle
import time

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend
from keras.applications.inception_v3 import InceptionV3
from keras.layers import (
    LSTM, Activation, BatchNormalization, Dense, Dropout, Embedding, Flatten,
    Merge, RepeatVector, TimeDistributed)
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import image, sequence
from PIL import Image
from tqdm import tqdm

test_encoding_path = 'static/encoded_test_images_inceptionV3.p'
vocab_path = 'static/vocab.p'
inception_model_path = 'static/inception_model.h5'
caption_model_architecture_path = 'static/caption_model.json'
caption_model_path = '../../weights/flickr8k/caption_model_weights.h5'

#def initialize_models():
encoding_test = pickle.load(open(test_encoding_path, 'rb'))
vocab = pickle.load(open(vocab_path, 'rb'))
word_idx = {val:index for index, val in enumerate(vocab)}
idx_word = {index:val for index, val in enumerate(vocab)}
max_length = 40

# Load InceptionV3 Model
inception_model = load_model(inception_model_path)
print("Inception Model Loaded Successfully")

# Load Caption Generation Model
json_file = open(caption_model_architecture_path, 'r')
model_json = json_file.read()
json_file.close()
caption_model = model_from_json(model_json)
caption_model.load_weights(caption_model_path)
print("Caption Model Loaded Successfully")

graph = tf.get_default_graph()

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

def encode(image):
    global graph
    with graph.as_default():
        image = preprocess(image)
        temp_enc = inception_model.predict(image)
        temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
        return temp_enc

def greedy_search_predictions(image_file, is_test):
    global graph
    with graph.as_default():
        start_word = ["<start>"]
        while 1:
            now_caps = [word_idx[i] for i in start_word]
            now_caps = sequence.pad_sequences([now_caps], maxlen=max_length, padding='post')
            if is_test==1:
                e = encoding_test[image_file]
            else:
                e = encode(image_file)
            preds = caption_model.predict([np.array([e]), np.array(now_caps)])
            word_pred = idx_word[np.argmax(preds[0])]
            start_word.append(word_pred)
            
            # Keep predicting next word until <end> is predicted or caption length > max_length
            if word_pred == "<end>" or len(start_word) > max_length: 
                break
            
        return ' '.join(start_word[1:-1])

def beam_search_predictions(image_file, is_test, beam_index):
    global graph
    with graph.as_default():
        start = [word_idx["<start>"]]
        
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < max_length:
            temp = []
            for s in start_word:
                now_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
                if is_test==1:
                    e = encoding_test[image_file]
                else:
                    e = encode(image_file)

                preds = caption_model.predict([np.array([e]), np.array(now_caps)])
                
                word_preds = np.argsort(preds[0])[-beam_index:]
                
                # Getting the top Beam index = 3  predictions and creating a new list to put them back through the model
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])
                        
            start_word = temp
            # Sort by probability
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Get top words
            start_word = start_word[-beam_index:]
        
        start_word = start_word[-1][0]
        intermediate_caption = [idx_word[i] for i in start_word]

        final_caption = []
        
        for i in intermediate_caption:
            if i != '<end>':
                final_caption.append(i)
            else:
                break
        
        final_caption = ' '.join(final_caption[1:])
        return final_caption

def predict_caption(img_path, is_test):
    print ('Greedy search:', greedy_search_predictions(img_path, is_test))
    print ('Beam Search, k=3:', beam_search_predictions(img_path, is_test, beam_index=3))
    print ('Beam Search, k=5:', beam_search_predictions(img_path, is_test, beam_index=5))
    print ('Beam Search, k=7:', beam_search_predictions(img_path, is_test, beam_index=7))
