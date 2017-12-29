from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model

def convert_binary_matrix(data, num_tokens):
    return to_categorical(data, num_classes=num_tokens)


def load_tokenize_model(file_name):
    with open(file_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer, len(tokenizer.word_index) + 1


def preproc(docs, max_length, token_model):
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


START_SEQ = 'startseq'
END_SEQ = 'endseq'
max_length = 25
batch_size = 128  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 25000

# Vectorize the data.

decoder_token_model, num_decoder_tokens = load_tokenize_model('decoder')
encoder_token_model, num_encoder_tokens = load_tokenize_model('encoder')

# # del source_data
# # del target_data
# # encoder_input_data=convert_binary_matrix(encoder_input_data,num_encoder_tokens)
# # decoder_input_data=convert_binary_matrix(decoder_input_data,num_decoder_tokens)
# decoder_target_data = convert_binary_matrix(decoder_target_data, num_decoder_tokens)

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
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# # Run training
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.load_weights('s2s_ex.h5')
# Run training

# for i in range(epochs):
#     for index in range(int(len(input_texts) / batch_size) + 1):
#         input_batch = input_texts[index:(index * batch_size)]
#         output_batch = target_texts[index:(index * batch_size)]
#         encoder_input_data, decoder_input_data, decoder_target_data = loop_batch(input_batch, output_batch,
#                                                                                  input_token_index, target_token_index)
#         loss = model.train_on_batch([encoder_input_data, decoder_input_data], decoder_target_data)
#         print("Epoch : " + str(i) + " Batch : " + str(index) + " Loss : " + str(loss))
# Save model
# model=load_model('s2s_ex.h5')

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
    # Encode the input as state vectors.
    model.reset_states()
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
    response = input("Please enter your sentences: ")
    seq = [response]
    input_seq = preproc(seq,max_length,encoder_token_model)
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', response)
    print('Decoded sentence:', decoded_sentence)