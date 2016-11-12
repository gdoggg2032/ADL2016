from nltk.tree import *
import types




def left(self):
	# check if is leaf
	assert not self.isleaf, "left() to a leaf node"
	return self[0]

def right(self):
	# check if is leaf
	assert not self.isleaf, "right() to a leaf node"
	return self[1]

def isleaf(self):

	return self.height() == 2

def get_word(self):
	# check if is leaf
	assert self.isleaf, "word() not a leaf node"
	return self[0]

def set_word(self, word):
	# check if is leaf
	assert self.isleaf, "word() not a leaf node"
	self[0] = word

def get_ilabel(self):
	label = self.label().split('+')[0]
	return Tree.label_dict.get(label, None)

def set_ilabel(self, ilabel):
	self.set_label(Tree.reverse_label_dict[ilabel])

def import_label_dict(self, label_dict):
	
	Tree.label_dict.update(label_dict)
	Tree.reverse_label_dict = {v:k for k,v in label_dict.iteritems()}


def add_atrributes():

	Tree.left = property(left)
	Tree.right = property(right)
	Tree.isleaf = property(isleaf)
	Tree.word = property(get_word, set_word)
	Tree.import_label_dict = types.MethodType(import_label_dict, None, Tree)
	Tree.ilabel = property(get_ilabel, set_ilabel)
	Tree.sem = None
	Tree.label_dict = {}
	# Tree.left = types.MethodType(left, None, Tree)
	# Tree.right = types.MethodType(right, None, Tree)
	# Tree.isleaf = types.MethodType(isleaf, None, Tree)

add_atrributes()

def traverseTree(tree, D):
	label = tree.label()
	if label not in D:
		D[label] = len(D)
	for i, subtree in enumerate(tree):
		if type(subtree) == Tree:
			traverseTree(subtree, D)

# def setTreeId(tree, t_id):
# 	tree.t_id = t_id
# 	for i, subtree in enumerate(tree):
# 		if type(subtree) == Tree:
# 			setTreeId(subtree, "{}_{}".format(t_id, i))
# def setTreeId(tree):
# 	try:
# 		tree.t_id = str(tree)
# 	except:
# 		print tree.label
# 	for i, subtree in enumerate(tree):
# 		if type(subtree) == Tree:
# 			setTreeId(subtree)

def setTreeId(tree, vocab):

	if tree.isleaf:
		tree.t_id = "({})".format(vocab.encode(tree.word))
	else:
		child_t_ids = []
		for i, subtree in enumerate(tree):
			if type(subtree) == Tree:
				setTreeId(subtree, vocab)
				child_t_ids.append(subtree.t_id)
		tree.t_id = "("+"".join(child_t_ids)+")"

def unk_filter(tree, vocab):

	if tree.isleaf:
		tree.word = vocab.decode(vocab.encode(tree.word))
	else:
		for i, subtree in enumerate(tree):
			if type(subtree) == Tree:
				unk_filter(subtree, vocab)




def load_trees(tree_file, sem, vocab):

	Ts = open(tree_file, "r").read().strip().split("\n\n")
	trees = [Tree.fromstring(t) for t in Ts]
	D = {}
	for i, t in enumerate(trees):
		traverseTree(t, D)
		Tree.chomsky_normal_form(t)
		t.collapse_unary(collapsePOS=True, collapseRoot=True)
		t.sem = sem
		# setTreeId(t, "{}_{}".format(sem, i))
		# unk_filter(t, vocab)
		setTreeId(t, vocab)
	trees[0].import_label_dict(D)

	return trees
