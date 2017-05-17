""" A simple character-level RNN for predicting next characters
in a given text. Can also be used to generate new text.
Created by Frank Schneider."""

import tensorflow as tf

### Global Variables ###

DATA_PATH = 'Data/Shakespeare_edit.txt'
READ_VOCAB = False # If True, the vocabulary of characters will be read from the data
SEQ_LENGTH = 10 # Length of Sequence, Input to the RNN, (= size of X)
OVERLAP = SEQ_LENGTH/2 # Size of the overlap between the different Sequences
BATCH_SIZE = 4 # Batch size (=number of Sequences per "batch")





### Functions ###

# Defines the vocabulary of possible characters
def define_vocab():
	print '\n', '*** Defining vocabulary for', DATA_PATH,'***','\n'
	if READ_VOCAB:
		data = open(DATA_PATH, 'r').read()
		vocab = list(sorted(set(data)))
	else:
		vocab = list(sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			"abcdefghijklmnopqrstuvwxyz"
			" .,!?"
			"0123456789")))
	vocab_size = len(vocab)
	print 'Vocabulary :', vocab,'\n'
	print 'Size of the Vocabulary:', vocab_size, 'characters','\n'
	return vocab, vocab_size


def read_data(char2idx):
	seq = []
	for text in open(DATA_PATH, 'r').read():
		idx = char2idx[text]
		seq.append(idx)
		if len(seq) == SEQ_LENGTH:
			yield seq
			seq = seq[SEQ_LENGTH-OVERLAP:SEQ_LENGTH]

def read_batch(data_stream):
	batch = []
	for seq in data_stream:
		batch.append(seq)
		if len(batch) == BATCH_SIZE:
			yield batch
			batch = []
	yield batch

# [map(idx2char.get, s) for s in batch] "retranslate a whole batch"


### Main ###

def main():
	vocab, vocab_size = define_vocab()
	char2idx = { ch:i for i,ch in enumerate(vocab) }
	idx2char = { i:ch for i,ch in enumerate(vocab) }


if __name__ == '__main__':
	main()