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
from random import randint


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


def check_valid_input(text, vocab_size):
    result = []
    new_word = False
    for word in text[0]:
        if word >= vocab_size:
            new_word = True
            word = randint(0, vocab_size)
        result.append(word)
    if new_word:
        return np.array([result])
    else:
        return text


START_SEQ = 'startseq'
END_SEQ = 'endseq'
max_length = 20
batch_size = 128  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 2048  # Latent dimensionality of the encoding space.
num_samples = 25000  # Number of samples to train on.

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
encoder_1 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs_1 = encoder_1(embed_input, initial_state=np.zeros(latent_dim))
y_1, state_h_1, state_c_1 = encoder_outputs_1
# We discard `encoder_outputs` and only keep the states.
encoder_states_1 = [state_h_1, state_c_1]

encoder_2 = LSTM(latent_dim, return_state=True)
y_2, state_h_2, state_c_2 = encoder_2(y_1, initial_state=np.zeros(latent_dim))
# We discard `encoder_outputs` and only keep the states.
encoder_states_2 = [state_h_2, state_c_2]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
embed_target = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
# # We set up our decoder to return full output sequences,
# # and to return internal states as well. We don't use the
# # return states in the training model, but we will use them in inference.
decoder_lstm_1 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs_1, _, _ = decoder_lstm_1(embed_target,
                                         initial_state=encoder_states_1)
decoder_lstm_2 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs_2, _, _ = decoder_lstm_2(decoder_outputs_1,
                                         initial_state=encoder_states_2)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs_2)
# decoder_inputs = Input(shape=(None,))
# x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
# x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
# decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# # Run training
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.load_weights('s2s_ex.h5')
# model.summary()
# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.1, callbacks=[early_stopping])
# plot_history(model)
# Save model
# model = load_model('s2s_ex.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model_1 = Model(encoder_inputs, encoder_outputs_1)
y_1 = Input(shape=(None, latent_dim))
encoder_state_input = Input(shape=(None, latent_dim))
# encoder_y_1, encoder_h_1,encoder_c_1=encoder_outputs_1
# encoder_model_2 = Model([encoder_y_1], encoder_states_2)
_, encoder_state_h_2, encoder_state_c_2 = encoder_2(y_1, encoder_state_input)
encoder_states_2 = [encoder_state_h_2, encoder_state_c_2]
encoder_model_2 = Model([y_1], encoder_states_2)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
embed_output = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_outputs_1, decoder_state_h_1, decoder_state_c_1 = decoder_lstm_1(
    embed_output, initial_state=decoder_states_inputs)
decoder_model_1 = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_1] + [decoder_state_h_1] + [decoder_state_c_1])

decoder_outputs_2, decoder_state_h_2, decoder_state_c_2 = decoder_lstm_2(
    y_1, initial_state=decoder_states_inputs)
decoder_outputs_2 = decoder_dense(decoder_outputs_2)
decoder_model_2 = Model(
    [y_1] + decoder_states_inputs,
    [decoder_outputs_2] + [decoder_state_h_2] + [decoder_state_c_2])


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
    encoder_y_1, encoder_h_1, encoder_c_1 = encoder_model_1.predict(input_seq)
    encoder_states_1 = [encoder_h_1, encoder_c_1]
    encoder_states_2 = encoder_model_2.predict(encoder_y_1)

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
    decoder_state_1_input = encoder_states_1
    decoder_state_2_input = encoder_states_2
    while not stop_condition:
        decoder_output_1, decoder_state_h_1, decoder_state_c_1 = decoder_model_1.predict(
            [target_seq] + decoder_state_1_input)
        output_tokens, decoder_state_h_2, decoder_state_c_2 = decoder_model_2.predict(
            [decoder_output_1] + decoder_state_2_input)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = get_words_from_token_model(decoder_token_model, sampled_token_index)
        if sampled_char == None:
            sampled_char = ""
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
        decoder_state_1_input = [decoder_state_h_1, decoder_state_c_1]
        decoder_state_2_input = [decoder_state_h_2, decoder_state_c_2]

    return decoded_sentence


# for seq_index in range(100):
#     # Take one sequence (part of the training test)
#     # for trying out decoding.
#     response = input("Please enter your name: ")
#     seq = [response]
#     input_seq = preproc(seq,max_length,encoder_token_model)
#     decoded_sentence = decode_sequence(input_seq)
#     print('-')
#     print('Input sentence:', response)
#     print('Decoded sentence:', decoded_sentence)

from flask import Flask, request, jsonify, json, Response
import os, requests
# from bson.objectid import ObjectId
# import datetime
from datetime import datetime

app = Flask(__name__)
# app.config['DEBUG'] = True
service_host = "127.0.0.1"
service_port = 5000
baseurl = "http://" + service_host + ":" + str(service_port) + "/"


@app.route('/vietnamese', methods=['GET'])
def get_tasks():
    return "Hello"


@app.route('/vietnamese', methods=['POST'])
def new_product():
    parameters = request.get_json(force=True)
    message = parameters["message"]
    print(message)
    seq = [message]
    input_seq = preproc(seq, max_length, encoder_token_model)
    input_seq = check_valid_input(input_seq, num_encoder_tokens)
    decoded_sentence = decode_sequence(input_seq).replace(END_SEQ, "")
    print(decoded_sentence)
    return jsonify(message=decoded_sentence)
    # return json.dump({"name": name})


ACCESS_TOKEN = "EAAYSfPA8lnsBALTzt6OFufmytCaEbFw8cnyawNag4ZBcTRyjnAzN36rmlM8GZBvLQLZC4ySEZBPeXjUHXfDEViQRroPsZAmQEYru1wZBB0NO6yQKIvPEeqRm4LZAWNgEp5fQaZA2y8JfVqmjn251T4MDsiwcAK2guhTHAa3GvYILbAZDZD"


# app = Flask(__name__)

@app.route('/', methods=['GET'])
def verify():
    # our endpoint echos back the 'hub.challenge' value specified when we setup the webhook
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == 'foo':
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    #
    return 'Hello World (from Flask!)', 200


def reply(user_id, msg):
    data = {
        "recipient": {"id": user_id},
        "message": {"text": msg}
    }
    resp = requests.post("https://graph.facebook.com/v2.6/me/messages?access_token=" + ACCESS_TOKEN, json=data)
    print(resp.content)


@app.route('/', methods=['POST'])
def handle_incoming_messages():
    data = request.json
    sender = data['entry'][0]['messaging'][0]['sender']['id']
    message = data['entry'][0]['messaging'][0]['message']['text']
    print(message)
    seq = [message]
    input_seq = preproc(seq, max_length, encoder_token_model)
    input_seq = check_valid_input(input_seq, num_encoder_tokens)
    decoded_sentence = decode_sequence(input_seq).replace(END_SEQ, "")
    print(decoded_sentence)
    reply(sender, decoded_sentence)

    return "ok"


if __name__ == '__main__':
    app.run(host=service_host, port=service_port, use_reloader=False)
