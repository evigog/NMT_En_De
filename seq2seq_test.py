#!/usr/b8n/env python3
# coding: utf-8

# ### Neural Machine Translation using word level language model and embeddings in Keras
import pandas as pd
import numpy as np
import string
from string import digits
#import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from keras.initializers import RandomUniform
from keras.callbacks import LearningRateScheduler, LambdaCallback
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import time
from keras.models import load_model
import pickle

# Building source to target translator

# PARAMETERS:
EMBED_SIZE = 300
EPOCHS = 100
LR = 0.7
MAX_SENT_SIZE_SRC = 10
MAX_SENT_SIZE_TRGT = 15
BATCH_SIZE = 128
SUBSAMPLE_SIZE = -1
LAYERS = 1


class MT_LSTM():

    def __init__(self, filename='fra.txt'):
        self.lines = pd.DataFrame()
        self.reverse_input_char_index = pickle.load(open('reverse_input_char_index.pkl','rb'))
        self.reverse_target_char_index = pickle.load(open('reverse_target_char_index.pkl','rb'))
        self.source_token_index = pickle.load(open('source_token_index.pkl','rb'))
        self.target_token_index = pickle.load(open('target_token_index.pkl','rb'))
        self.encoder_source_data = pickle.load(open('encoder_source_data.pkl','rb'))
        self.lines.source = pickle.load(open('lines_source.pkl','rb'))
        self.decoder_source_data = pickle.load(open('decoder_source_data.pkl','rb'))
              

    def fit(self):

        model = load_model('MT_LSTM.h5')
        model.summary()
        self.encoder_model = load_model('MT_LSTM_encoder.h5')
        self.decoder_model = load_model('MT_LSTM_decoder.h5')


    def predict(self, input_seq):

        # Function to generate sequences
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = self.target_token_index['START_']
    
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
    
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += ' '+sampled_char
    
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '_END' or
               len(decoded_sentence.split()) > MAX_SENT_SIZE_TRGT):
                stop_condition = True
    
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
    
            # Update states
            states_value = [h, c]
    
        return decoded_sentence

       
def main():
    model = MT_LSTM()
    model.fit()
    #TODO
    #Look at the some translations
    number_of_translations = 15
    indexes = [np.random.choice(list(range(len(model.encoder_source_data)))) for x in range(number_of_translations)]
    scores = []
    for seq_index in indexes:
        input_seq = model.encoder_source_data[seq_index]
        decoded_sentence = model.predict(input_seq)
        print('--------------------------------------------------------------------------')
        print('Input sentence: {}'.format(model.lines.source[seq_index]))
        print('Decoded sentence: {}'.format(decoded_sentence))
        score = sentence_bleu(model.lines.source[seq_index], decoded_sentence)
        scores.append(score)
        print("BLEU score: {}".format(score))

    print('--------------------------------------------------------------------------')
    print("Average BLEU score: {}".format(np.mean(scores)))

if __name__ == "__main__":
    main()
