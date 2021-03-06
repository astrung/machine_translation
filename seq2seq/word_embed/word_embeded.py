from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import json, copy, pickle
from random import randint

def plot_history(model):
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plt.imsave('plot.jpg')


def remove_long_sentences(train_data, test_data, num_samples):
    new_train = []
    new_test = []
    new_index = 0
    for index in range(len(train_data)):
        if len(train_data[index]) < 30:
            new_train.append(train_data[index])
            new_test.append(test_data[index])
            new_index += 1
        if new_index == num_samples:
            break
    del train_data
    del test_data
    return new_train, new_test


def convert_binary_matrix(data, num_tokens):
    return to_categorical(data, num_classes=num_tokens)


def preproc(docs, max_length, name):
    token_model = Tokenizer()
    token_model.fit_on_texts(docs)
    vocab_size = len(token_model.word_index) + 1
    # integer encode the documents
    encoded_docs = token_model.texts_to_sequences(docs)
    print(encoded_docs[:10])
    # pad documents to a max length of 4 words
    # max_length = 50
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs[:10])
    with open(name, 'wb') as handle:
        pickle.dump(token_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # load the whole embedding into memory
    return vocab_size, padded_docs, token_model

def sen2vec(docs, max_length, token_model):
    # token_model = Tokenizer()
    token_model.fit_on_texts(docs)
    # integer encode the documents
    encoded_docs = token_model.texts_to_sequences(docs)
    print(encoded_docs[:10])
    # pad documents to a max length of 4 words
    # max_length = 50
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs[:10])
    # load the whole embedding into memory
    return padded_docs

def check_valid_input(text,vocab_size):
    result=[]
    new_word=False
    for word in text[0]:
        if word >= vocab_size:
            new_word=True
            word=randint(0, vocab_size)
        result.append(word)
    if new_word:
        return np.array([result])
    else:
        return text
START_SEQ = 'startseq'
END_SEQ = 'endseq'
max_length = 25
batch_size = 512  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 25000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra.txt'

np.random.seed(0)

with open('source.txt', encoding='utf-8') as f:
    source_data = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
source_data = [str(x.strip()) for x in source_data]

with open('target.txt', encoding='utf-8') as f:
    target_data = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
target_data = [(START_SEQ + ' ' + str(x.strip()) + ' ' + END_SEQ) for x in target_data]
# train_data, test_data = remove_long_sentences(train_data, test_data, num_samples)
# Vectorize the data.
source_data, target_data = remove_long_sentences(source_data, target_data, num_samples)
num_decoder_tokens, decoder_input_data, decoder_token_model = preproc(target_data, max_length, 'decoder')
num_encoder_tokens, encoder_input_data, encoder_token_model = preproc(source_data, max_length, 'encoder')
decoder_target_data = np.c_[decoder_input_data[:, 1:], np.zeros(num_samples)]

print("Decoded : " + str(num_decoder_tokens))
print("Encoded : " + str(num_encoder_tokens))
# encoder_input_data=convert_binary_matrix(encoder_input_data,num_encoder_tokens)
# decoder_input_data=convert_binary_matrix(decoder_input_data,num_decoder_tokens)
decoder_target_data = convert_binary_matrix(decoder_target_data, num_decoder_tokens)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
embed_input = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(embed_input)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
embed_target = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
# # We set up our decoder to return full output sequences,
# # and to return internal states as well. We don't use the
# # return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embed_target,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# decoder_inputs = Input(shape=(None,))
# x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
# x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
# decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1, callbacks=[early_stopping])
plot_history(model)
# for i in range(epochs):
#     for index in range(int(len(input_texts) / batch_size) + 1):
#         input_batch = input_texts[index:(index * batch_size)]
#         output_batch = target_texts[index:(index * batch_size)]
#         encoder_input_data, decoder_input_data, decoder_target_data = loop_batch(input_batch, output_batch,
#                                                                                  input_token_index, target_token_index)
#         loss = model.train_on_batch([encoder_input_data, decoder_input_data], decoder_target_data)
#         print("Epoch : " + str(i) + " Batch : " + str(index) + " Loss : " + str(loss))
# Save model
model.save('s2s_ex.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
embed_target = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(
    embed_target, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def get_words_from_token_model(token_model, index):
    for key, value in token_model.word_index.items():
        if value == index:
            return key


def decode_input_seq(input_seq):
    encoded_sentence = ''
    for index in input_seq[0]:
        encoded_sentence += get_words_from_token_model(encoder_token_model, index)
    return encoded_sentence


def decode_sequence(input_seq):
    encoder_model.reset_states()
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    # target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, decoder_token_model.word_index[START_SEQ]] = 1
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = decoder_token_model.word_index[START_SEQ]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = get_words_from_token_model(decoder_token_model, sampled_token_index)
        decoded_sentence += sampled_char
        decoded_sentence += " "

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == END_SEQ or
                len(decoded_sentence) > max_length + 2):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', source_data[seq_index])
    print('Decoded sentence:', decoded_sentence)


while(True):
    message = input("Please enter your name: ")
    print(message)
    seq = [message]
    input_seq = sen2vec(seq, max_length, encoder_token_model)
    input_seq = check_valid_input(input_seq, num_encoder_tokens)
    decoded_sentence = decode_sequence(input_seq).replace(END_SEQ, "")
    print(decoded_sentence)