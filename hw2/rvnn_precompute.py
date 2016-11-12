from tree import *
from utils import *
import argparse
import sys
import time
import random
import tensorflow as tf
import progressbar as pb
import numpy as np
import copy
import cPickle as pickle

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--pos_tree', default='./dataset_rnn_eng/training_data.pos.tree', type=str)
	parser.add_argument('--pos_data', default='./dataset_rnn_eng/training_data.pos', type=str)
	parser.add_argument('--neg_tree', default='./dataset_rnn_eng/training_data.neg.tree', type=str)
	parser.add_argument('--neg_data', default='./dataset_rnn_eng/training_data.neg', type=str)
	parser.add_argument('--test_tree', default='./dataset_rnn_eng/testing_data.txt.tree', type=str)
	parser.add_argument('--test_data', default='./dataset_rnn_eng/testing_data.txt', type=str)

	parser.add_argument('--dim', default=300, type=int)
	parser.add_argument('--sem_size', default=2, type=int)
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--lr', default=0.01, type=float)
	# parser.add_argument('--batch', default=2, type=int)
	parser.add_argument('--l2', default=0.01, type=float)
	parser.add_argument('--name', default='rnn_eng', type=str)
	parser.add_argument('--model', default='model', type=str)
	parser.add_argument('--predict', default='predict', type=str)

	parser.add_argument('--mode', default=2, type=int)
	parser.add_argument('--conti', default=0, type=int)
	parser.add_argument('--conti_model', default='model', type=str)

	parser.add_argument('--vocab', default='./vocab', type=str)

	parser.add_argument('--embed', default=None, type=str)


	args = parser.parse_args()

	return args
def load_bin_vec(fname, vocab, dim):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    # word_vecs = {}
    embedding = np.random.uniform(-0.25,0.25,(vocab.vocab_size, dim))
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab.w2i:
                vec = np.fromstring(f.read(binary_len), dtype='float32')  
                # word_vecs[word] = vec
                embedding[vocab.w2i[word]] = vec
            	
            else:
                f.read(binary_len)
    return embedding

class rvnn(object):

	def __init__(self, args):
		self.args = args
		self.load_data()
		self.index = 0
		



	def load_data(self):
		
		if self.args.mode % 2 == 0:
			self.vocab = Vocab()
			words = open(self.args.pos_data, "r").read().strip().split() + open(self.args.neg_data, "r").read().strip().split() 
			self.vocab.construct(words)

			pos_trees = load_trees(self.args.pos_tree, 1, self.vocab)
			neg_trees = load_trees(self.args.neg_tree, 0, self.vocab)
			self.train_trees = pos_trees + neg_trees
			self.train_eval_trees = copy.deepcopy(self.train_trees)
			self.train_eval_labels = [t.sem for t in self.train_eval_trees]

			pickle.dump(self.vocab, open(self.args.vocab, "wb"))
			random.shuffle(self.train_trees)

			if self.args.embed:
				print >> sys.stderr, "load pre-trained embedding from ", self.args.embed
				self.embedding = load_bin_vec(self.args.embed, self.vocab, self.args.dim)



		if self.args.mode > 0:
			self.vocab = pickle.load(open(self.args.vocab, "rb"))

			test_trees = load_trees(self.args.test_tree, -1, self.vocab)

		
			self.test_trees = test_trees

		

	def next_batch(self, batch_size):
		trees = self.train_trees[self.index : batch_size + self.index]
		if self.index + batch_size >= len(self.train_trees):
			self.index = 0
			random.shuffle(self.train_trees)
		else:
			self.index = self.index + batch_size
		return trees

	def inference(self, trees, train=True):

		if train:
			self.logits = {}
			self.losses = {}
			self.train_ops = {}
		else:
			self.logits = {}

		pbar = pb.ProgressBar(widgets=["add_model:", pb.FileTransferSpeed(unit="trees"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(trees)).start()
		
		self.count = 0.0
		self.non_skip = 0.0
		for i, tree in enumerate(trees):
			pbar.update(i+1)
			self.add_model(tree)
			logits = self.logits[tree.t_id]
		pbar.finish()
		print "real:{}, total:{}, rate:{}".format(self.non_skip, self.count, self.non_skip / self.count)
			
		
		with tf.variable_scope('projection', reuse=True):
			U = tf.get_variable('U')
			bs = tf.get_variable('bs')

		with tf.variable_scope('composition', reuse=True):
			W1 = tf.get_variable('W1')


		l2_regular = tf.nn.l2_loss(W1) + tf.nn.l2_loss(U)

		optimizer = tf.train.AdagradOptimizer(self.args.lr)
		


		projections = {}
		pbar = pb.ProgressBar(widgets=["add_projections:", pb.FileTransferSpeed(unit="trees"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(trees)).start()
		
		for i, tree in enumerate(trees):
			pbar.update(i+1)


			# add projecttions
			projections[tree.t_id] = tf.matmul(self.logits[tree.t_id], U) + bs
		pbar.finish()


		if train:
			pbar = pb.ProgressBar(widgets=["add losses:", pb.FileTransferSpeed(unit="trees"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(trees)).start()
			
			for i, tree in enumerate(trees):
				pbar.update(i+1)
				# add loss 
				sems = [tree.sem]
				cross_entropy = tf.reduce_sum(
					tf.nn.sparse_softmax_cross_entropy_with_logits(projections[tree.t_id], sems))
				self.losses[tree.t_id] = cross_entropy + self.args.l2 * l2_regular
			pbar.finish()


			pbar = pb.ProgressBar(widgets=["add train_ops:", pb.FileTransferSpeed(unit="trees"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(trees)).start()
			
			for i, tree in enumerate(trees):
				pbar.update(i+1)
				# add loss 
				self.train_ops[tree.t_id] = optimizer.minimize(self.losses[tree.t_id])
			pbar.finish()



		self.logits = projections
		

		if train:
			eval_logits = tf.concat(0,[self.logits[tree.t_id] for tree in self.train_eval_trees])
			self.eval_predictions = self.predictions(eval_logits)
		

		# for i, tree in enumerate(trees):
		# 	# print >> sys.stderr, "inference: {} / {}".format(i, len(trees))
		# 	pbar.update(i+1)
		# 	node_tensors = self.add_model(tree)

		# 	if predict_only_root:
		# 		node_tensors = node_tensors[tree.t_id]
		# 	else:
		# 		node_tensors = [tensor for node, tensor in node_tensors.iteritems()]
		# 		node_tensors = tf.concat(0, node_tensors)

		# 	logits = self.add_projections(node_tensors)

		# 	if train:
		# 		self.logits[tree.t_id] = logits
		# 		sems = [tree.sem]
		# 		self.losses[tree.t_id] =  self.loss(self.logits[tree.t_id], sems)
				
		# 		self.train_ops[tree.t_id] = self.training(self.losses[tree.t_id])
		# 	else:
		# 		test_logits[tree.t_id] = logits
		# pbar.finish()
		# if train:
		# 	eval_logits = tf.concat(0,[self.logits[tree.t_id] for tree in self.train_eval_trees])
		# 	self.eval_predictions = self.predictions(eval_logits)
		# else:
		# 	return test_logits
		

	def add_model_vars(self):

		with tf.variable_scope('composition'):
			tf.get_variable('embedding', shape=[self.vocab.vocab_size, self.args.dim])
			tf.get_variable('W1', shape=[2 * self.args.dim, self.args.dim])
			tf.get_variable('b1', shape=[1, self.args.dim])

		with tf.variable_scope('projection'):
			tf.get_variable('U', shape=[self.args.dim, self.args.sem_size])
			tf.get_variable('bs', shape=[1, self.args.sem_size])

	def add_model(self, tree):

		with tf.variable_scope('composition', reuse=True):
			embedding = tf.get_variable('embedding')
			W1 = tf.get_variable('W1')
			b1 = tf.get_variable('b1')


		self.count += 1.0

		curr_tree_tensor = None
		if tree.isleaf:
			tensor = self.logits.get(tree.t_id)
			if tensor is not None:
				curr_tree_tensor = tensor
			else:
				self.non_skip += 1.0
				index = self.vocab.encode(tree.word)
				h = tf.gather(embedding, indices=index)
				curr_tree_tensor = tf.expand_dims(h, 0)

		else:
			self.add_model(tree.left)
			self.add_model(tree.right)
			tensor = self.logits.get(tree.t_id)
			if tensor is not None:
				curr_tree_tensor = tensor
			else:
				self.non_skip += 1.0
				hl_hr = tf.concat(1, [self.logits[tree.left.t_id], self.logits[tree.right.t_id]])
				curr_tree_tensor = tf.nn.relu(tf.matmul(hl_hr, W1) + b1)

		self.logits[tree.t_id] = curr_tree_tensor
	

	def add_projections(self, node_tensors):

		logits = None
		with tf.variable_scope('projection', reuse=True):
			U = tf.get_variable('U')
			bs = tf.get_variable('bs')

		# add some activation function?
		logits = tf.matmul(node_tensors, U) + bs

		return logits

	def loss(self, logits, labels):

		loss = None

		with tf.variable_scope('composition', reuse=True):
			W1 = tf.get_variable('W1')

		with tf.variable_scope('projection', reuse=True):
			U = tf.get_variable('U')

		l2_regular = tf.nn.l2_loss(W1) + tf.nn.l2_loss(U)

		cross_entropy = tf.reduce_sum(
			tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
		loss = cross_entropy + self.args.l2 * l2_regular


		return loss

	def training(self, loss):

		train_op = None
		optimizer = tf.train.AdagradOptimizer(self.args.lr)
		train_op = optimizer.minimize(loss)
		# self.sess.run(tf.initialize_all_variables())


		return train_op

	def predictions(self, y):

		predictions = tf.argmax(y, 1)

		return predictions

	def predict(self, trees=None, train=False):

		if not train:
			self.inference(trees, train=False)
			logits = tf.concat(0,[self.logits[tree.t_id] for tree in trees])
			predictions = self.predictions(logits)
		
			root_prediction = self.sess.run(predictions)
		else:
			root_prediction = self.sess.run(self.eval_predictions)
		
		return root_prediction

	def run_epoch(self):

		total_loss = 0.
		start_time = time.time()
		pbar = pb.ProgressBar(widgets=["run_epoch:", pb.FileTransferSpeed(unit="trees"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(self.train_trees)).start()
		random.shuffle(self.train_trees)
		for i, tree in enumerate(self.train_trees):
			pbar.update(i+1)
			# trees = self.next_batch(self.args.batch)
			# print len(trees)
				# self.train_trees[batch_size*i:batch_size*(i+1)]
			# sems = np.array([tree.sem for tree in trees])

			# logits = tf.concat(0,[self.logits[tree.t_id] for tree in self.train_trees[i:i+5]])
			# print ">>>>", logits.graph
			# sems = [tree.sem for tree in self.train_trees[i:i+5]]
			# # logits = [self.logits[tree.t_id] for tree in trees]
			# loss = self.loss(logits, sems)
			# print "@@@@", loss.graph

			# # sems = [tree.sem]
			# # loss = self.loss(self.logits[tree.t_id], sems)

			# train_op = self.training(loss)
			# print "vvvvv", train_op.graph
			cost, _ = self.sess.run([self.losses[tree.t_id], self.train_ops[tree.t_id]])

			total_loss += cost / len(self.train_trees)

		pbar.finish()

		return total_loss

	def train(self):

		acc_history = []
		loss_history = []
		with tf.Graph().as_default(), tf.Session() as sess:
		
			self.sess = sess
			self.add_model_vars()
			
			self.saver = tf.train.Saver()
			
			
				

			print >> sys.stderr, "inference logits"
			self.inference(self.train_trees)
			self.sess.run(tf.initialize_all_variables())

			if self.args.embed:
				with tf.variable_scope('composition', reuse=True):
					embedding = tf.get_variable('embedding')
					self.sess.run(embedding.assign(self.embedding))

			if self.args.conti:
				self.load_model(self.args.conti_model)


			print >> sys.stderr, "start training"
			for epoch in xrange(self.args.epochs):
				start_time = time.time()
				
				loss = self.run_epoch()
				train_acc, confmat = self.eval()
				self.acc = train_acc

				print >> sys.stderr, "Epoch {}: acc={}, loss={}, time={}".format(epoch+1, train_acc, loss, time.time()-start_time)
				print >> sys.stderr, confmat

				if (epoch+1) % 10 == 0:
					tmp_model = "[{}][dim={}][epochs={}][lr={}][l2={}][acc={}]".format(self.args.name, 
							self.args.dim, epoch, self.args.lr, self.args.l2, self.acc)
					if self.args.conti:
						tmp_model += "[conti={}]".format(self.args.conti_model)
					save_path = self.saver.save(self.sess, tmp_model, write_meta_graph=False)
					print >> sys.stderr, "tmp_model saved in file: ", save_path

			
			self.dump_model()

	def eval(self):
		train_preds = self.predict(None, True)
		train_labels = self.train_eval_labels
		train_acc = np.equal(train_preds, train_labels).mean()
		confmat = self.make_conf(train_labels, train_preds)

		return train_acc, confmat

	def make_conf(self, labels, predictions):

		confmat = np.zeros((2,2))
		for l, p in zip(labels, predictions):
			confmat[l, p] += 1
		return confmat

	def dump_model(self):
		print >> sys.stderr, "dumping model"
		save_path = self.saver.save(self.sess, self.args.model, write_meta_graph=False)
		print >> sys.stderr, "Model saved in file: ", save_path
		eval_model = "[{}][dim={}][epochs={}][lr={}][l2={}][acc={}]".format(self.args.name, 
			self.args.dim, self.args.epochs, self.args.lr, self.args.l2, self.acc)
		if self.args.conti:
			eval_model += "[conti={}]".format(self.args.conti_model)
		save_path = self.saver.save(self.sess, eval_model, write_meta_graph=False)
		print >> sys.stderr, "Eval_model saved in file: ", save_path

	def load_model(self, model):
		print >> sys.stderr, "loading model from: ", model
		self.saver.restore(self.sess, model)

	def test(self):
		self.sess = tf.Session()
		self.add_model_vars()
		self.saver = tf.train.Saver()
		self.load_model(self.args.model)
		predicts = self.predict(self.test_trees)
		pstr = "\n".join([str(i) for i in predicts])
		print >> open(self.args.predict, "w"), pstr


def main():

	args = arg_parse()

	start_time = time.time()

	model = rvnn(args)

	if args.mode % 2 == 0:

		print >> sys.stderr, "start training"

		model.train()

		print >> sys.stderr, "training time: {}".format(time.time()-start_time)


	if args.mode > 0:
		start_time = time.time()

		print >> sys.stderr, "start testing"

		model.test()

		print >> sys.stderr, "testing time: {}".format(time.time()-start_time)




if __name__ == "__main__":
	main()




