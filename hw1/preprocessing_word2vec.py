import sys
import collections
import cPickle as pickle
import argparse
import time
import random
import numpy as np




def read_corpus(corpus_file):
	return open(corpus_file, "r").read().split()

def build_dataset(words, low_freq):
	# count = [['UNK', -1]]
	# count.extend(collections.Counter(words))

	vocab_dict = {}

	C = collections.Counter(words)
	count = C.most_common()
	# vocab_dict['UNK'] = 0 # the most common
	for word, c in count:
		if C[word] >= low_freq:
			vocab_dict[word] = len(vocab_dict)
	
	text = []
	unk = 0
	for word in words:
		if C[word] < low_freq:
			# word = "UNK"
			unk += 1
		else:
			text.append(vocab_dict[word])

	inv_vocab = {v:k for k, v in vocab_dict.iteritems()}
	count = {k:C[v]  for v, k in vocab_dict.iteritems()}
	# count[0] = unk
	del words
	return text, vocab_dict, inv_vocab, count

# def build_dataset(words, low_freq):
# 	# count = [['UNK', -1]]
# 	# count.extend(collections.Counter(words))

# 	vocab_dict = {}

# 	C = collections.Counter(words)
# 	count = C.most_common()
# 	vocab_dict['UNK'] = 0 # the most common
# 	for word, c in count:
# 		if C[word] >= low_freq:
# 			vocab_dict[word] = len(vocab_dict)
	
# 	text = []
# 	unk = 0
# 	for word in words:
# 		if C[word] < low_freq:
# 			word = "UNK"
# 			unk += 1
# 		text.append(vocab_dict[word])

# 	inv_vocab = {v:k for k, v in vocab_dict.iteritems()}
# 	count = {k:C[v]  for v, k in vocab_dict.iteritems()}
# 	count[0] = unk
# 	del words
# 	return text, vocab_dict, inv_vocab, count



def generate_batch(text, count, args):


	index = 0

	num_skips = args.num_skips
	skip_window = args.window
	t = args.t

	text_size = len(text)
	batch_size = num_skips * text_size

	

	

	# assert batch_size % num_skips == 0
	# assert num_skips <= 2 * skip_window

	batch = np.zeros((batch_size, 2), dtype=np.int32)

	span = 2 * skip_window + 1 
	buf = collections.deque(maxlen=span)
	for _ in range(span):
		buf.append(text[index])
		index = (index + 1) % text_size

	for i in range(text_size):
		if (i+1)%10000 == 0:
			print >> sys.stderr, "processing: ", i, "/", text_size








		# each word has num_skips c_words
		# total words in batch: batch_size // num_skips
		# real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
		ran = (np.sqrt(count[buf[skip_window]] / (t * text_size)) + 1.0) * (t * text_size) / count[buf[skip_window]]
		r = random.random()
		if r > ran:
			buf.append(text[index])
			index = (index + 1) % text_size
			continue

		target = skip_window
		targets_avoid = [target]

		for j in range(num_skips):
			while target in targets_avoid:
				target = random.randint(0, span - 1)
			targets_avoid.append(target)
			batch[i*num_skips+j, 0] = buf[skip_window]
			batch[i*num_skips+j, 1] = buf[target]

		buf.append(text[index])
		index = (index + 1) % text_size



	return batch

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--corpus', default='./corpus/text8', type=str)
	parser.add_argument('--text', default='./w2v.text', type=str)
	parser.add_argument('--vocab', default='./w2v.vocab', type=str)
	parser.add_argument('--t', default=5e-3, type=float)
	parser.add_argument('--low', default=5, type=int)
	
	parser.add_argument('--num_skips', default=10, type=int)
	
	parser.add_argument('--window', default=5, type=int)
	
	parser.add_argument('--batch', default="./w2v.batch", type=str)

	# parser.add_argument('--vector', default='./g_vector.txt', type=str)
	args = parser.parse_args()

	return args


if __name__ == "__main__":

	args = arg_parse()

	corpus_file = args.corpus
	low_freq = args.low
	text_file = args.text
	vocab_file = args.vocab
	batch_file = args.batch

	

	words = read_corpus(corpus_file)

	print 'Data size', len(words)

	print "build_dataset"
	text, vocab_dict, inv_vocab, count = build_dataset(words, low_freq)

	# print count


	# text_size = len(text)
	# t = args.t
	# for i in range(0, len(vocab_dict)):
	# 	ran = (np.sqrt(count[i] / (t * text_size)) + 1.0) * (t * text_size) / count[i]
	# 	print inv_vocab[i], ran

	print "generate_batch"
	batch = generate_batch(text, count, args)

	print batch.shape

	print "dumping"
	pickle.dump(text, open(text_file, "w"))
	pickle.dump(vocab_dict, open(vocab_file, "w"))
	batch.dump(batch_file)







# print "da"

# def generate_data(text, skip_window):

# 	data = []

# 	for i, w in enumerate(text):
# 		print >> sys.stderr, i, "/", len(text)
# 		span_l = max(i - skip_window, 0)
# 		span_r = min(i + skip_window, len(text)-1)
# 		for s in range(span_l, span_r + 1):
# 			if s != i:
# 				data.append([text[i], text[s]])
# 				# 0 1 2 3 4 5 6 7 8
# 	return np.array(data)

# data = generate_data(text[:10], 3)
# print data[0:10]
# print text[0:10]



# data_index = 0

# def generate_batch(batch_size, num_skips, skip_window):
# 	global data_index
# 	assert batch_size % num_skips == 0
# 	assert num_skips <= 2 * skip_window
# 	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
# 	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
# 	span = 2 * skip_window + 1 # [ skip_window target skip_window ]
# 	buffer = collections.deque(maxlen=span)
# 	for _ in range(span):
# 		buffer.append(data[data_index])
# 		data_index = (data_index + 1) % len(data)
# 	print buffer
# 	for i in range(batch_size // num_skips):
# 		target = skip_window	# target label at the center of the buffer
# 		targets_to_avoid = [ skip_window ]
# 		for j in range(num_skips):
# 			while target in targets_to_avoid:
# 				target = random.randint(0, span - 1)
# 			targets_to_avoid.append(target)
# 			batch[i * num_skips + j] = buffer[skip_window]
# 			labels[i * num_skips + j, 0] = buffer[target]
# 		buffer.append(data[data_index])
# 		data_index = (data_index + 1) % len(data)
# 	return batch, labels














