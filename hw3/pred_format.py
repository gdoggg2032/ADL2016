import sys
import argparse



def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--input', default='./model_tagging/test_results/tagging.test.hyp.txt', type=str)
	parser.add_argument('--output', default='./predict.txt', type=str)
	

	args = parser.parse_args()

	return args

def pred_format(args):

	data = []
	with open(args.input, "r") as f:
		all_data = f.read().split("\n\n")
		for d in all_data:
			pl = []
			dl = d.strip().split('\n')
			for i, dd in enumerate(dl):
				if i == len(dl)-1: 
					continue
				ddl = dd.split(" ")
				p = ddl[2]
				pl.append(p)
			data.append(" ".join(pl))
	with open(args.output, "w") as p:
		for d in data:
			p.write(d+'\n')



if __name__ == "__main__":

	args = arg_parse()

	pred_format(args)


