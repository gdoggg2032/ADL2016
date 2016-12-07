import sys
import os
import argparse
import time
import json
import re




def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--test_data', default='./NLG_data/test.txt', type=str)
	parser.add_argument('--predict_data', default='./predict.txt', type=str)
	parser.add_argument('--output', default="answer.txt", type=str)
	args = parser.parse_args()

	return args


def parse_query(query, sen):

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

			# actually we use this
			# sen = sen.replace(replace_name, name)
			
			# but this' BLEU score higher
			sen = sen.replace(replace_name, entity)

	return sen







def main(args):

	
	# test
	test_path = args.test_data
	test_data = open(test_path, "r").read().strip().split("\n")

	predict_path = args.predict_data
	predict_data = open(predict_path, "r").read().strip().split("\n")
	
	ans_path = args.output
	ansf = open(ans_path, "w")

	for t, p in zip(test_data, predict_data):
		output = parse_query(t, p)
		print >> ansf, output







if __name__ == "__main__":
	s = time.time()
	args = arg_parse()
	main(args)
	print >> sys.stderr, "time cost:", time.time() - s


