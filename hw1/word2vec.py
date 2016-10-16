import tensorflow as tf
import numpy as np
import sys
import cPickle as pickle
import argparse
import time
import collections
import random
import math

class Data(object):

	def __init__(self, args):

		text_file = args.text
		vocab_file = args.vocab

		self.text = pickle.load(open(text_file, "r"))
		self.vocab_dict = pickle.load(open(vocab_file, "r"))
		
		self.text_size = len(self.text)
		self.vocabulary_size = len(self.vocab_dict)

		self.index = 0

	def next_batch(self, batch_size, num_skips, skip_window):

		assert batch_size % num_skips == 0
		assert num_skips <= 2 * skip_window

		batch_w = np.zeros((batch_size, 1), dtype=np.int32)
		batch_c = np.zeros((batch_size, 1), dtype=np.int32)
		span = 2 * skip_window + 1 
		buf = collections.deque(maxlen=span)
		for _ in range(span):
			buf.append(self.text[self.index])
			self.index = (self.index + 1) % self.text_size

		for i in range(batch_size // num_skips):
			# each word has num_skips c_words
			# total words in batch: batch_size // num_skips
			target = skip_window
			targets_avoid = [target]

			for j in range(num_skips):
				while target in targets_avoid:
					target = random.randint(0, span - 1)
				targets_avoid.append(target)
				batch_w[i*num_skips+j, 0] = buf[skip_window]
				batch_c[i*num_skips+j, 0] = buf[target]

			buf.append(self.text[self.index])
			self.index = (self.index + 1) % self.text_size

		return batch_w, batch_c





		





def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--text', default='./text8_w2v.text', type=str)
	parser.add_argument('--vocab', default='./text8_w2v.vocab', type=str)

	parser.add_argument('--lr', default=0.05, type=float)
	parser.add_argument('--epochs', default=10, type=int)
	parser.add_argument('--batch', default=256, type=int)
	parser.add_argument('--num_skips', default=8, type=int)
	parser.add_argument('--neg_sampling', default=64, type=int)
	parser.add_argument('--window', default=5, type=int)
	parser.add_argument('--dim', default=100, type=int)
	parser.add_argument('--model', default="./model_w2v", type=str)
	# parser.add_argument('--vector', default='./g_vector.txt', type=str)
	args = parser.parse_args()

	return args


def dump_vector(vec_file, vocab_dict, w):
	
	w2v_w = w

	with open(vec_file, "w") as f:
		for k, v in vocab_dict.iteritems():
			out = ' '.join([str(val) for val in w2v_w[v]])
			out = k + ' ' + out + '\n'
			f.write(out)


def word2vec(train, args, display_step=1000, device='/cpu:0'):

	batch_size = args.batch
	embedding_size = args.dim
	skip_window = args.window
	num_skips = args.num_skips
	num_sampled = args.neg_sampling
	learning_rate = args.lr
	training_epochs = args.epochs


	with tf.device(device):

		w = tf.placeholder(tf.int32, [None])
		c_w = tf.placeholder(tf.int32, [None, 1])

		embeddings = tf.Variable(tf.random_uniform([train.vocabulary_size, embedding_size], -0.1, 0.1))

		embed = tf.nn.embedding_lookup(embeddings, w)

		nce_weights = tf.Variable(
			tf.truncated_normal([train.vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.random_uniform([train.vocabulary_size], -0.01, 0.01))

	cost = tf.reduce_mean(
		tf.nn.nce_loss(nce_weights, nce_biases, embed, c_w,
			num_sampled, train.vocabulary_size))

	optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

	init = tf.initialize_all_variables()

	saver = tf.train.Saver()

	print >> sys.stderr, "start session"

	print >> sys.stderr, "batch / total_batch file_time optimize_time"

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(training_epochs):

			start_time = time.time()

			all_text = 1000#train.text_size/(batch_size/num_skips)

			b_time = 0.
			o_time = 0.

			avg_cost = 0.

			for i in range(all_text):

				if (i+1) % display_step == 0:
					print >> sys.stderr, i, "/", all_text, b_time, o_time


				b_start = time.time()
				batch_w, batch_c = train.next_batch(batch_size, num_skips, skip_window)
				b_time += time.time() - b_start

				o_start = time.time()
				_, c = sess.run([optimizer, cost], feed_dict={w:batch_w[:, 0], c_w:batch_c})
				o_time += time.time() - o_start
			
				avg_cost += c

			avg_cost /= all_text

			print >> sys.stderr, "Epoch:", epoch+1, "cost=", avg_cost, "time: ", time.time() - start_time

		save_path = saver.save(sess, args.model)
		print >> sys.stderr, "Model saved in file: ", save_path

		return sess.run(embeddings)


def main():

	args = arg_parse()

	print >> sys.stderr, "load data"
	train = Data(args)

	print >> sys.stderr, "training"
	embeddings = word2vec(train, args)

	print >> sys.stderr, "dumping vectors"
	dump_vector("./text8_vector"+"[w2v][dim="+str(args.dim)+"]"+"[iter="+str(args.epochs)+"][lr="+str(args.lr)+"]", train.vocab_dict, embeddings)


if __name__ == "__main__":
	main()







