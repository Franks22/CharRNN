""" A simple character-level RNN for predicting next characters
in a given text. Can also be used to generate new text.
Created by Frank Schneider."""

import tensorflow as tf
import os

### Global Variables ###

DATA_PATH = 'Data/Shakespeare.txt'
READ_VOCAB = True # If True, the vocabulary of characters will be read from the data
SEQ_LENGTH = 50 # Length of Sequence, Input to the RNN, (= size of X)
OVERLAP = SEQ_LENGTH/2 # Size of the overlap between the different Sequences
BATCH_SIZE = 64 # Batch size (=number of Sequences per "batch")
NUM_NEURONS = 500 # Number of Neurons per Layer
NUM_LAYERS = 3 #Number of Layers
LR = 0.003
CELL_TYPE = 'GRU' # Cell Type can be RNN, LSTM, GRU, or NAS (LSTM AND NAS not working!!!!)
GENERATION_SIZE = 300 # Number of characters created per generated sample
TEMPERATURE = 1.0 # TODO
EVAL_STEP = 100 # Generate a Sample and show batch_loss each 100 steps
MAX_EPOCHS = 10 # Number of epochs (1 epoch = 1 whole run through the text)


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


def model(seq,vocab_size):
	seq_onehot = tf.one_hot(seq, vocab_size)
	if CELL_TYPE == 'RNN':
		cell = tf.contrib.rnn.BasicRNNCell(NUM_NEURONS)
	elif CELL_TYPE == 'LSTM':
		cell = tf.contrib.rnn.BasicLSTMCell(NUM_NEURONS)
	elif CELL_TYPE == 'GRU':
		cell = tf.contrib.rnn.GRUCell(NUM_NEURONS)
	elif CELL_TYPE == 'NAS':
		cell = tf.contrib.rnn.NASCell(NUM_NEURONS)
	else:
		raise Exception('model type not supported')
	#cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)

	in_state = tf.placeholder_with_default(cell.zero_state(tf.shape(seq_onehot)[0], tf.float32), [None, NUM_NEURONS])
	
	output, out_state = tf.nn.dynamic_rnn(cell, seq_onehot, initial_state = in_state, dtype=tf.float32)

	logits = tf.contrib.layers.fully_connected(output, vocab_size, None)

	loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logits[:,:-1], labels = seq_onehot[:,1:]))

	sample = tf.multinomial(tf.exp(logits[:, -1] / TEMPERATURE), 1)[:, 0] 

	return loss, in_state, out_state, sample

def training(seq, loss, in_state, out_state, optimizer, char2idx, idx2char, sample):

	with tf.Session() as sess:
		
		sess.run(tf.global_variables_initializer())

		for epoch in range(MAX_EPOCHS):

			iteration = 1
			for batch in read_batch(read_data(char2idx)):
				batch_loss, _ = sess.run([loss, optimizer], {seq: batch})
				if iteration % EVAL_STEP == 0:
					print 'Epoch:', epoch, 'Iteration:', iteration, 'Batch Loss:', batch_loss
					generate(sess, seq, in_state, out_state, char2idx, idx2char, sample)
				iteration +=1

def generate(sess, seq, in_state, out_state, char2idx, idx2char, sample, seed='T'):
	sentence = seed
	state = None
	for _ in range(GENERATION_SIZE):
		batch = [[char2idx[sentence[-1]]]]
		feed = {seq: batch}
		if state is not None:
			feed.update({in_state: state})
		index, state = sess.run([sample, out_state], feed)
		sentence += idx2char[index[0]]
	print sentence


### Main ###

def main():
	vocab, vocab_size = define_vocab()
	char2idx = { ch:i for i,ch in enumerate(vocab) }
	idx2char = { i:ch for i,ch in enumerate(vocab) }
	seq = tf.placeholder(tf.int32, [None, None]) # Usually [BATCH_SIZE, SEQ_LENGHT]
	loss, in_state, out_state, sample = model(seq, vocab_size)
	optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
	training(seq, loss, in_state, out_state, optimizer, char2idx, idx2char, sample)


if __name__ == '__main__':
	main()