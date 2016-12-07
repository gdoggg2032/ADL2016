import sys
import os
import argparse
import time
import json
import re




def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_dir', default='./NLG_data', type=str)
	parser.add_argument('--output_dir', default='./NLG_data_parsed', type=str)
	parser.add_argument('--test', default=False, type=bool)
	parser.add_argument('--test_data', default="./test.txt", type=str)
	parser.add_argument('--test_output', default="./test.in", type=str)
	args = parser.parse_args()

	return args


def parse_query(query, sen1, sen2, Test=False):

	# parse query
	func = re.search("(.*)\(.*\)", query).group(1)
	content = re.search(".*\((.*)\)", query).group(1)
	ql = [func]
	for token in content.split(";"):
		# XX=XX or YY
		tl = token.split("=")
		if len(tl) == 1:
			# YY
			ql.append(tl[0])
		else:
			# XX=XX
			entity = tl[0]
			replace_name = "_{}_".format(entity)
			name = tl[1]
			name = name.replace("'", "").strip()
			sen1 = sen1.replace(name, replace_name)
			sen2 = sen2.replace(name, replace_name)
			ql.extend([tl[0], replace_name])
	qs = " ".join(ql)

	if not Test:
		return [qs, qs], [sen1, sen2]
	else:
		return [qs], ["."]







def main(args):

	if args.test == False:

		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)

		# train
		train_path = args.input_dir + "/train.json"
		train_data = json.load(open(train_path, "r"))
		queries = []
		sentences = []
		for d in train_data:
			qs, sens = parse_query(*d)
			queries.extend(qs)
			sentences.extend(sens)

		output_train_query_path = args.output_dir + "/train.in"
		output_train_sen_path = args.output_dir + "/train.out"
		with open(output_train_query_path, "w") as qf:
			with open(output_train_sen_path, "w") as sf:
				for q, s in zip(queries, sentences):
					print >> qf, q
					print >> sf, s

		# valid
		valid_path = args.input_dir + "/valid.json"
		valid_data = json.load(open(valid_path, "r"))
		queries = []
		sentences = []
		for d in valid_data:
			qs, sens = parse_query(*d)
			queries.extend(qs)
			sentences.extend(sens)

		output_valid_query_path = args.output_dir + "/valid.in"
		output_valid_sen_path = args.output_dir + "/valid.out"
		with open(output_valid_query_path, "w") as qf:
			with open(output_valid_sen_path, "w") as sf:
				for q, s in zip(queries, sentences):
					print >> qf, q
					print >> sf, s

		# test
		test_path = args.input_dir + "/test.txt"
		test_data = open(test_path, "r").read().strip().split("\n")
		
		queries = []
		sentences = []
		for d in test_data:
			# print d
			qs, sens = parse_query(d, ".", ".", Test=True)
			queries.extend(qs)
			sentences.extend(sens)

		output_test_query_path = args.output_dir + "/test.in"
		output_test_sen_path = args.output_dir + "/test.out"
		with open(output_test_query_path, "w") as qf:
			with open(output_test_sen_path, "w") as sf:
				for q, s in zip(queries, sentences):
					print >> qf, q
					print >> sf, s
	else:

		# test
		test_path = args.test_data #args.input_dir + "/test.txt"
		test_data = open(test_path, "r").read().strip().split("\n")
		
		queries = []
		sentences = []
		for d in test_data:
			# print d
			qs, sens = parse_query(d, ".", ".", Test=True)
			queries.extend(qs)
			sentences.extend(sens)

		output_test_query_path = args.test_output #args.output_dir + "/test.in"
		with open(output_test_query_path, "w") as qf:
			for q, s in zip(queries, sentences):
				print >> qf, q






if __name__ == "__main__":
	s = time.time()
	args = arg_parse()
	main(args)
	print >> sys.stderr, "time cost:", time.time() - s


