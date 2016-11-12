import sys
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.framework import ops
from copy import deepcopy


class Vocab(object):
	def __init__(self):
		self.w2i = {}
		self.i2w = {}
		self.word_count = defaultdict(int)
		self.total_words = 0
		self.vocab_size = 0
		self.unknown = '<unk>'
		self.add_word(self.unknown, count=0)


	def add_word(self, word, count=1):

		if word not in self.w2i:
			index = len(self.w2i)
			self.w2i[word] = index
			self.i2w[index] = word
		self.word_count[word] += count

	def construct(self, words):
		for word in words:
			self.add_word(word)
		self.total_words = sum(self.word_count.values())
		self.vocab_size = len(self.word_count)
		print >> sys.stderr, "{} total words with vocab size {}".format(self.total_words, self.vocab_size)

	def encode(self, word):
		if word not in self.w2i:
			word = self.unknown
		return self.w2i[word]

	def decode(self, index):
		assert index < len(self.word_count), "Vocab: index out of range"
		return self.i2w[index]

	def __len__(self):
		return len(self.word_count)




