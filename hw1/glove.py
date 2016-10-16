import tensorflow as tf
import numpy as np
import sys
import cPickle as pickle
import argparse
import time


class Data(object):

	def __init__(self, args):

		cooccur_file = args.cooccur
		vocab_file = args.vocab
		self.x_max = args.x_max
		self.x_alpha = args.x_alpha


		self.data = np.load(cooccur_file)
		# np.random.shuffle(self.data)

		self.index = 0
		self.n_samples = self.data.shape[0]
		self.n_dims = self.data.shape[1]

		self.vocab_dict = pickle.load(open(vocab_file, "r"))
		self.vocabulary_size = len(self.vocab_dict)

	def next_batch(self, batch_size):

		if self.index == 0:
			np.random.shuffle(self.data)

		if self.index + batch_size > self.n_samples:

			batch = self.data[self.index :]


			self.index = 0
			
		else:
			batch = self.data[self.index : self.index + batch_size]
			self.index += batch_size
		weights = np.minimum(np.power(batch[:, 2]/self.x_max, self.x_alpha), 1.0)
		# return batch[:, 0], batch[:, 1], batch[:, 2], weights
		return batch[:, 0:2], batch[:, 2], weights


def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--cooccur', default='./text8.cooccur', type=str)
	parser.add_argument('--vocab', default='./text8.vocab', type=str)
	parser.add_argument('--x_max', default=100.0, type=float)
	parser.add_argument('--x_alpha', default=0.75, type=float)
	parser.add_argument('--lr', default=0.05, type=float)
	parser.add_argument('--epochs', default=10, type=int)
	parser.add_argument('--batch', default=100, type=int)
	parser.add_argument('--dim', default=100, type=int)
	parser.add_argument('--model', default="./model", type=str)
	# parser.add_argument('--vector', default='./g_vector.txt', type=str)
	args = parser.parse_args()

	return args


def dump_vector(vec_file, vocab_dict, w, c_w):
	
	glove_w = w + c_w

	with open(vec_file, "w") as f:
		for k, v in vocab_dict.iteritems():
			out = ' '.join([str(val) for val in glove_w[v]])
			out = k + ' ' + out + '\n'
			f.write(out)


def glove(train, args, display_step=1000):

	learning_rate = args.lr
	training_epochs = args.epochs
	batch_size = args.batch
	embedding_size = args.dim

	with tf.device('/cpu:0'):
		x = tf.placeholder(tf.int32, [None, 2])
		cooccurs = tf.placeholder(tf.float32, [None])

		f_x = tf.placeholder(tf.float32, [None])

		embeddings = tf.Variable(tf.random_uniform([train.vocabulary_size, embedding_size], -0.1, 0.1))
		c_embeddings = tf.Variable(tf.random_uniform([train.vocabulary_size, embedding_size], -0.1, 0.1))
		
		biases = tf.Variable(tf.random_uniform([train.vocabulary_size], -0.01, 0.01))
		c_biases = tf.Variable(tf.random_uniform([train.vocabulary_size], -0.01, 0.01))

		embed = tf.nn.embedding_lookup(embeddings, x[:, 0])
		c_embed = tf.nn.embedding_lookup(c_embeddings, x[:, 1])

		bias = tf.nn.embedding_lookup(biases, x[:, 0])
		c_bias = tf.nn.embedding_lookup(c_biases, x[:, 1])

	# f_x = tf.minimum(tf.constant(1.0), tf.pow(tf.div(coocurs, x_max), x_alpha))

	embed_product = tf.reduce_sum(tf.mul(embed, c_embed), reduction_indices=1)
	neg_log_cooccurs = tf.neg(tf.log(cooccurs))
	dis_exp = tf.square(tf.add_n([embed_product, bias, c_bias, neg_log_cooccurs]))
	cost = tf.reduce_sum(tf.mul(dis_exp, f_x))

	optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

	init = tf.initialize_all_variables()

	saver = tf.train.Saver()

	print >> sys.stderr, "start session"

	print >> sys.stderr, "batch / total_batch file_time optimize_time"

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(training_epochs):

			start_time = time.time()

			avg_cost = 0.

			total_batch = int((train.n_samples+batch_size) / float(batch_size))

			b_time = 0.
			o_time = 0.

			for i in range(total_batch):
				if (i+1) % display_step == 0:
					print >> sys.stderr, i, "/", total_batch, b_time, o_time

				b_start = time.time()
				batch, batch_c,  batch_w = train.next_batch(batch_size)
				b_time += time.time() - b_start
				o_start = time.time()
				_, c = sess.run([optimizer, cost], feed_dict={x:batch, cooccurs:batch_c, f_x:batch_w})
				
				o_time += time.time() - o_start
			
				avg_cost += c 

			avg_cost /= train.n_samples

			if epoch % display_step == 0:
				print >> sys.stderr, "Epoch:", epoch+1, "cost=", avg_cost, "time: ", time.time() - start_time, ""+' '

		save_path = saver.save(sess, args.model)
 		print >> sys.stderr, "Model saved in file: ", save_path

 		return sess.run(embeddings), sess.run(c_embeddings)
 

def main():

	args = arg_parse()

	train = Data(args)

	embeddings, c_embeddings = glove(train, args)

	dump_vector("./text8_vector"+"[glove][dim="+str(args.dim)+"]"+"[iter="+str(args.epochs)+"][lr="+str(args.lr)+"]", train.vocab_dict, embeddings, c_embeddings)


if __name__ == "__main__":
	main()






















