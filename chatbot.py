from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sqlite3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re

# Inputting data from database and separating to test and train set.
timeframe = '2015-05'
connection = sqlite3.connect('{}.db'.format(timeframe))
cursor = connection.cursor()
df = pd.read_sql("SELECT * FROM parent_reply WHERE parent NOT NULL AND score > 50 AND comment NOT NULL", connection)
print(df)
X = df.iloc[:, 2]
y = df.iloc[:, 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


X_train = ['starttoken ' + line + ' endtoken' for line in X_train]
y_train = ['starttoken ' + line + ' endtoken' for line in y_train]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train )
tokenizer.fit_on_texts(y_train)
vocab_size = len(tokenizer.word_index) + 1


input_sequences = tokenizer.texts_to_sequences(X_train)
target_sequences = tokenizer.texts_to_sequences(y_train)


max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)





input_sequences = pad_sequences(input_sequences, maxlen=100, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=100, padding='post')


dim = 128
hidden_size = 256

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_size, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=dim)(decoder_inputs)
decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


decoder_input_data = target_sequences[:, :-1]
decoder_target_data = target_sequences[:, 1:]


input_sequences = np.array(input_sequences)
decoder_input_data = np.array(decoder_input_data)
decoder_target_data = np.array(decoder_target_data)


model.fit([input_sequences, decoder_input_data], decoder_target_data, batch_size=64, epochs=20)

model.save('seq2seq_model.h5') 


encoder_model = Model(encoder_inputs, encoder_states)


decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding = Embedding(input_dim=vocab_size, output_dim=dim)(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['starttoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        if (sampled_char == 'endtoken' or len(decoded_sentence) > max_target_len):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

for seq_index in range(10):
    input_seq = input_sequences[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', X_train[seq_index])
    print('Decoded sentence:', decoded_sentence)


