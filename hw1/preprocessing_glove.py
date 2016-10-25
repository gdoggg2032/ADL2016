import sys
import numpy as np
import cPickle as pickle
import csv
import argparse
# def count_cooccurence(text, vocab_dict, window_size):
# 	coocur_dict = {}

# 	for i in xrange(len(text)):
# 		print "\r", i+1, "/", len(text),
		
# 		wc = vocab_dict[text[i]]
# 		for j in range(i, max(i - window_size, 0)-1, -1):
# 			w = vocab_dict[text[j]]
# 			distance = i - j + 1
# 			for (w1, w2) in [(w, wc), (wc, w)]:
# 				if (w1, w2) not in coocur_dict:
# 					coocur_dict[(w1, w2)] = 1.0 / distance
# 				else:
# 					coocur_dict[(w1, w2)] += 1.0 / distance
# 	return coocur_dict
def count_cooccurence(text, vocab_dict, window_size):
	coocur_dict = {}

	for i in xrange(len(text)):
		print "\r", i+1, "/", len(text),
		






		wc = vocab_dict[text[i]]
		for j in range(i-1, max(i - window_size, 0)-1, -1):
			w = vocab_dict[text[j]]
			distance = i - j 
			for (w1, w2) in [(w, wc), (wc, w)]:
				if (w1, w2) not in coocur_dict:
					coocur_dict[(w1, w2)] = 1.0 / distance
				else:
					coocur_dict[(w1, w2)] += 1.0 / distance
	print ""
	return coocur_dict



def get_vocab_dict(text):
	s = set(text)
	d = {ss:i for (i,ss) in enumerate(s)}
	return d


def count_freq(text):
	d = {}
	for w in text:
		if w not in d:
			d[w] = 1
		else:
			d[w] += 1

	return d

def text_filter(text, freq, low_freq):
	T = []
	for w in text:
		if freq[w] >= low_freq:
			T.append(w)
	return T

def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--corpus', default='./corpus/text8', type=str)
	parser.add_argument('--cooccur', default='./glove.cooccur', type=str)
	parser.add_argument('--vocab', default='./glove.vocab', type=str)
	
	parser.add_argument('--low', default=5, type=int)
	

	
	parser.add_argument('--window', default=10, type=int)
	
	
	# parser.add_argument('--vector', default='./g_vector.txt', type=str)
	args = parser.parse_args()

	return args



if __name__ == "__main__":

	args = arg_parse()

	text_file = args.corpus

	window_size = args.window

	low_freq = args.low

	vocab_file = args.vocab
	cooccur_file = args.cooccur


	with open(text_file, "r") as f:
		print "reading text"
		text = f.read().strip().split()
		print "done reading text"
		print "get vocab dict"
		vocab_dict = get_vocab_dict(text)

		freq = count_freq(text)
		text = text_filter(text, freq, low_freq)
		# print vocab_dict
		coocur_dict = count_cooccurence(text, vocab_dict, window_size)
		# print coocur_dict


	
	print "convert to np.array"
	with open(cooccur_file+".tmp", "w") as p:
		for w, c in coocur_dict.iteritems():
			print >> p, ",".join([str(w[0]),str(w[1]),str(c)])


	del coocur_dict

	C = np.genfromtxt(cooccur_file+".tmp", delimiter=',', dtype="float32")

	print "dumping"

	C.dump(cooccur_file)

	pickle.dump(vocab_dict, open(vocab_file, "w"))






