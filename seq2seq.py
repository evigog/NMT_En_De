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

# Building source to target translator

# PARAMETERS:
EMBED_SIZE = 50
EPOCHS = 1000
LR = 0.1
MAX_SENT_SIZE_SRC = 10
MAX_SENT_SIZE_TRGT = 15
BATCH_SIZE = 128
SUBSAMPLE_SIZE = 2400 # -1
LAYERS = 1



class MT_LSTM():

    def __init__(self, filename='fra.txt'):
        # initialization for all LSTM parameters
        self.param_initializer = RandomUniform(minval=-0.08, maxval=0.08, seed=None)

        self.read_data(filename)
        self.clean_data()
        low_lim, up_lim = self.get_percentiles()
        low_lim1, up_lim1 = self.get_percentiles(src=0)

#        print(self.lines.head(10))
        

        print("Target sentences with length outside the range: {} - {} will be removed.".format(low_lim1, up_lim1))
        cond1 = self.lines['target_len'] > low_lim1
        cond2 = self.lines['target_len'] < up_lim1
        self.lines = self.lines[cond1 & cond2]
#        print(self.lines.head(10))

 
        print("Source sentences with length outside the range: {} - {} will be removed.".format(low_lim, up_lim))
        cond1 = self.lines['source_len'] > low_lim 
        cond2 = self.lines['source_len'] < up_lim
        self.lines = self.lines[cond1 & cond2]
        self.lines = self.lines.reset_index()
        self.create_vocabulary()
        global MAX_SENT_SIZE_SRC, MAX_SENT_SIZE_TRGT
        MAX_SENT_SIZE_SRC = self.get_max_sent_len()
        MAX_SENT_SIZE_TRGT = self.get_max_sent_len(src=0)

        print("Max length of source sentences: {}".format(MAX_SENT_SIZE_SRC))
        print("Max length of target sentences: {}".format(MAX_SENT_SIZE_TRGT))
        print(self.lines.head(10))

        self.src_size = len(self.lines.source)
        self.trgt_size = len(self.lines.target)
        self.src_vocab_size = len(self.source_token_index)
        self.trgt_vocab_size = len(self.target_token_index)

        self.encoder_source_data = np.zeros(
            (self.src_size, MAX_SENT_SIZE_SRC),
            dtype='float32')
        
        self.decoder_source_data = np.zeros(
            (self.trgt_size, MAX_SENT_SIZE_TRGT),
            dtype='float32')
        
        self.decoder_target_data = np.zeros((self.trgt_size, MAX_SENT_SIZE_TRGT, self.trgt_vocab_size), dtype='float')
        
        for i, (source_text, target_text) in enumerate(zip(self.lines.source, self.lines.target)):
            for t, word in enumerate(source_text.split()):
                self.encoder_source_data[i, t] = self.source_token_index[word]

            sentence_length = self.lines.source_len[i]
            self.encoder_source_data[i, 0:sentence_length] = np.flip(self.encoder_source_data[i, 0:sentence_length])

            for t, word in enumerate(target_text.split()):
                # decoder_target_data is ahead of decoder_source_data by one timestep
                
                # take care of words not existing in target vocabulary
                if word not in self.target_words:
                    word = 'UNK'

                self.decoder_source_data[i, t] = self.target_token_index[word]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character. 
                    self.decoder_target_data[i, t - 1, self.target_token_index[word]] = 1.

    def fit(self):
        from keras.utils import plot_model

        # #### Encoder model
        encoder_inputs = Input(shape=(None,))
        en_x=  Embedding(self.src_vocab_size, EMBED_SIZE,name='embed_encoder')(encoder_inputs)

        layer_input = en_x
        for i in range(LAYERS):
            encoder = LSTM(EMBED_SIZE, return_state=True, return_sequences=True, kernel_initializer=self.param_initializer, dropout=0.2)
            encoder_outputs, state_h, state_c = encoder(layer_input)
            layer_input = encoder_outputs


        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # #### Decoder model 
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))        
        dex=  Embedding(self.trgt_vocab_size, EMBED_SIZE,name='embed_decoder')
        final_dex= dex(decoder_inputs)

        decoder_lstm = LSTM(EMBED_SIZE, return_sequences=True, return_state=True, kernel_initializer=self.param_initializer, dropout=0.2)        
        decoder_outputs, _, _ = decoder_lstm(final_dex,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.trgt_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)        
        sgd = SGD(lr=LR) #gradient norm clipping between 10 and 25
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        #plot_model(model, to_file='model.png', show_shapes=True)

        model.summary()

        # #### Fit the model         
        lr_schedule = self.decay_schedule()

        filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        def on_epoch_end(epoch, logs={},self=self ):
            number_of_translations = 3
#            print(len(self.encoder_source_data))
            indexes = [np.random.choice(list(range(len(self.encoder_source_data)))) for x in range(number_of_translations)]
            for seq_index in indexes:
                input_seq = self.encoder_source_data[seq_index]
                decoded_sentence = self.predict(input_seq)
                print('--------------------------------------------------------------------------')
                print('Input sentence: {}'.format(self.lines.source[seq_index]))
                print('Decoded sentence: {}'.format(decoded_sentence))
                score = sentence_bleu(self.lines.source[seq_index], decoded_sentence)
                print("BLEU score: {}".format(score))

        print_callback = LambdaCallback(
        on_epoch_end=on_epoch_end)
        #Calling the subclass
   #     predictions=prediction_history()
        # Tesorboard callback
#        print(np.shape(self.encoder_source_data), np.shape(self.decoder_source_data))
        tensorboard = TensorBoard(log_dir='./logs{}'.format(time()), histogram_freq=1, embeddings_freq=0,  
          write_graph=True, write_images=True)#,embeddings_layer_names=['embed_encoder','embed_decoder'], embeddings_data=[self.encoder_source_data, self.decoder_source_data])#[np.array(self.source_token_index.values()),np.array(self.target_token_index.values())])

        callbacks_list = [checkpoint, tensorboard, print_callback]

 
        self.encoder_model = Model(encoder_inputs, encoder_states)
        #plot_model(model, to_file='encoder_model.png', show_shapes=True)

        #self.encoder_model.summary()
        
        # #### Create sampling model 
        decoder_state_input_h = Input(shape=(EMBED_SIZE,))
        decoder_state_input_c = Input(shape=(EMBED_SIZE,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        final_dex2= dex(decoder_inputs)
        
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs2] + decoder_states2)
        #plot_model(model, to_file='decoder_model.png', show_shapes=True)
       
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.source_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())


        import pickle
        pickle.dump(self.reverse_input_char_index, open('reverse_input_char_index.pkl','wb'))
        pickle.dump(self.reverse_target_char_index, open('reverse_target_char_index.pkl','wb'))
        pickle.dump(self.source_token_index, open('source_token_index.pkl','wb'))
        pickle.dump(self.target_token_index, open('target_token_index.pkl','wb'))
        self.encoder_source_data.dump('encoder_source_data.pkl')
        self.lines.source.to_pickle('lines_source.pkl')
        self.decoder_source_data.dump('decoder_source_data.pkl')


        model.fit([self.encoder_source_data, self.decoder_source_data], self.decoder_target_data,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_split=0.05, callbacks=callbacks_list)




        model.save('MT_LSTM.h5')
        self.encoder_model.save('MT_LSTM_encoder.h5')
        self.decoder_model.save('MT_LSTM_decoder.h5')




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
#            print(output_tokens[0,-1,:]) 
            # Sample a token
            #print(output_tokens)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            #print(sampled_token_index)
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

    def read_data(self, filename='./fra.txt'):
        self.lines= pd.read_table(filename, names=['source', 'target'])        
        if SUBSAMPLE_SIZE > 0:
            self.lines = self.lines[0:SUBSAMPLE_SIZE]
            print(self.lines)
        print("There are {} samples in the dataset".format(len(self.lines)))

     
    def clean_data(self):
        self.lines.source=self.lines.source.apply(lambda x: x.lower())
        self.lines.target=self.lines.target.apply(lambda x: x.lower())
        
        self.lines.source=self.lines.source.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
        self.lines.target=self.lines.target.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
        
        exclude = set(string.punctuation)
        self.lines.source=self.lines.source.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        self.lines.target=self.lines.target.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        remove_digits = str.maketrans('', '', digits)
        self.lines.source=self.lines.source.apply(lambda x: x.translate(remove_digits))
        self.lines.target=self.lines.target.apply(lambda x: x.translate(remove_digits))
            
        self.lines.target = self.lines.target.apply(lambda x : 'START_ '+ x + ' _END')

        self.lines['source_len'] = self.lines.source.apply(lambda x: len(x.split()))
        self.lines['target_len'] = self.lines.target.apply(lambda x: len(x.split()))


        
        #TODO:
        #add UNK tag to mark unknown words

    def create_vocabulary(self):
        cnt_src = Counter(' '.join(self.lines.source).split())
        cnt_trgt = Counter(' '.join(self.lines.target).split())

        #find num of words with frequency < 3
        #rq_src = list(filter(lambda x: x[1]<3, list(cnt_src.most_common()) ))
        #print('english words deleted': len(frq5_src))
        #frq_trgt = list(filter(lambda x: x[1]==1, list(cnt_trgt.most_common()) ))
        #frq_trgt_len = len(frq_trgt)

        #print('german words deleted', len(frq_trgt))

        ##discard rare words/least common words from target vocabulary
        #cnt_trgt = cnt_trgt.most_common()[:-frq_trgt_len]
        #cnt_src.most_common()[:-frq_trgt]

        self.source_words=set(cnt_src)
        self.target_words=set(cnt_trgt)
        self.target_words.add('START_')
        self.target_words.add('_END')
        self.target_words.add('UNK')

        self.source_token_index = dict(
            [(word, i) for i, word in enumerate(self.source_words)])
        self.target_token_index = dict(
            [(word, i) for i, word in enumerate(self.target_words)])

        print("There are {} words in source vocab, and {} in target.".format(len(self.source_words),len(self.target_words)))

    def get_max_sent_len(self, src=1): # src = 1 for source data, src = 0 for target data
        if src:
            return len(max([x.split() for x in self.lines.source], key=len))

        return len(max([x.split() for x in self.lines.target], key=len))


    def get_percentiles(self, src=1):
        if src:
            tab = [len(x.split()) for x in self.lines.source]
        else:
            tab =  [len(x.split()) for x in self.lines.target]

        print(np.percentile(tab, list(range(0,101,5))))
        return np.percentile(tab, [5,95])


    # learning rate: fixed rate in the beginning for 5 epochs and halving the rate every epoch
    def decay_schedule(self, initial_lr=0.1):
    
        def schedule(epoch):
            if epoch <=5:
                return initial_lr
            else:
                return float(initial_lr/8)
    
        return LearningRateScheduler(schedule)



def main():
    a = pd.read_table('../data/englishaa', names=['source'])
    b = pd.read_table('../data/germanaa', names=['target'])

#    a = a.head(4000)
#    b = b.head(4000)

    a['target'] = b.target
    a.to_csv('../data.txt', sep='\t', encoding='utf-8')
    print("There are {} samples in dataset.".format(len(a.source)))
#    print(len(b.target))
#    print(a.head(10)) 
    model = MT_LSTM('../data.txt')
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
