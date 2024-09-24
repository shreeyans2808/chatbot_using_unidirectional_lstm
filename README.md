# chatbot_using_unidirectional_lstm

This repo is made for the representation of the project. Everything present related to the project is present in the repo

# This file consists ofthe code of unidirectional chatbot using seq2seq models and LSTM.

seq2seq models consists of an encoder and decoder module which uses LSTM layers to first tokenize and then encode the statements saved in sql (converted from json file while web scrapping from reddit) into vectorized form using word2vec. LSTM layers are then applied and then it is decoded to normal human understanding words.

LSTM (Long Short Term Memory)-
LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to better capture long-range dependencies in sequential data, solving the issues of traditional RNNs, such as the vanishing and exploding gradient problems.

Key Features of LSTM:
Memory Cell: LSTMs have a cell state, which acts like a conveyor belt to carry information along the sequence. This allows the network to retain important information over long sequences.
Gates:
Forget Gate: Decides what information to throw away from the cell state.
Input Gate: Decides what new information to store in the cell state.
Output Gate: Decides what information to output based on the cell state and the current input.
Better Handling of Long-Term Dependencies: Unlike traditional RNNs, LSTMs are more capable of remembering information from earlier time steps over long sequences, making them well-suited for tasks where context from distant past is important.
How LSTM Works:
At each time step, LSTM updates its internal state based on the current input and its memory from previous time steps. This allows it to "remember" important information and "forget" irrelevant parts.