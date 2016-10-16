import sys
import collections
import cPickle as pickle


def read_corpus(corpus_file):
	return open(corpus_file, "r").read().split()




def build_dataset(words, low_freq):
	# count = [['UNK', -1]]
	# count.extend(collections.Counter(words))

	vocab_dict = {}

	C = collections.Counter(words)
	count = C.most_common()
	vocab_dict['UNK'] = 0 # the most common
	for word, c in count:
		if C[word] >= low_freq:
			vocab_dict[word] = len(vocab_dict)
	
	text = []

	for word in words:
		if C[word] < low_freq:
			word = "UNK"
		text.append(vocab_dict[word])

	inv_vocab = {v:k for k, v in vocab_dict.iteritems()}
	del words
	return text, vocab_dict, inv_vocab



if __name__ == "__main__":

	corpus_file = sys.argv[1]
	low_freq = int(sys.argv[2])
	text_file = sys.argv[3]
	vocab_file = sys.argv[4]

	words = read_corpus(corpus_file)

	print 'Data size', len(words)

	print "build_dataset"
	text, vocab_dict, inv_vocab = build_dataset(words, 5)

	print "dumping"
	pickle.dump(text, open(text_file, "w"))
	pickle.dump(vocab_dict, open(vocab_file, "w"))





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














