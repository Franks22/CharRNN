""" A simple character-level RNN for predicting next characters
in a given text. Can also be used to generate new text.
Created by Frank Schneider."""

import tensorflow as tf

### Global Variables ###

DATA_PATH = 'GRRMsmall.txt'
READ_VOCAB = True # If True, the vocabulary of characters will be read from the data




### Functions ###

# Defines the vocabulary of possible characters
def define_vocab():
	if READ_VOCAB:
		data = open(DATA_PATH, 'r').read()
		vocab = list(sorted(set(data)))
	else:
		vocab = list(sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            " .,!?"
            "0123456789")))
	vocab_size = len(vocab)
	return vocab, vocab_size








### Main ###

def main():
	vocab, vocab_size = define_vocab()
	print(vocab)
	print(vocab_size)


if __name__ == '__main__':
	main()