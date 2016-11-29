import sys
import argparse
import random
import os





def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--train', default='./data/indent_prediction/atis.train.w-intent.iob', type=str)
	parser.add_argument('--test', default='./data/indent_prediction/atis.test.iob', type=str)
	parser.add_argument('--dir', default='./data/hw3', type=str)
	parser.add_argument('--val_rate', default=0.1, type=float)
	parser.add_argument('--mode', default="train", type=str)

	args = parser.parse_args()

	return args


def pre_format(args):

	

	if not os.path.isdir(args.dir):
		os.makedirs(args.dir)

	
	if args.mode == "train":

		

		train_data = []
		with open(args.train, "r") as f:
			for line in f:
				ll = line.strip().split("\t")
				train_in_l = ll[0].split(" ")
				train_out_l = ll[1].split(" ")
				train_in_l = train_in_l[1:-1]
				label = train_out_l[-1]
				train_out_l = train_out_l[1:-1]

				train_data.append((" ".join(train_in_l), " ".join(train_out_l), label))

		random.shuffle(train_data)

		val_size = int(len(train_data) * args.val_rate)
		val_data = train_data[0: val_size]
		train_data = train_data[val_size:]

		if not os.path.isdir(args.dir+'/train'):
			os.makedirs(args.dir+'/train')

		with open(args.dir+'/train/train.seq.in', "w") as pin:
			with open(args.dir+'/train/train.seq.out', "w") as pout:
				with open(args.dir+'/train/train.label', "w") as pl:
					for (i, o, l) in train_data:
						pin.write(i+'\n')
						pout.write(o+'\n')
						pl.write(l+'\n')

		if not os.path.isdir(args.dir+'/valid'):
			os.makedirs(args.dir+'/valid')

		with open(args.dir+'/valid/valid.seq.in', "w") as pin:
			with open(args.dir+'/valid/valid.seq.out', "w") as pout:
				with open(args.dir+'/valid/valid.label', "w") as pl:
					for (i, o, l) in val_data:
						pin.write(i+'\n')
						pout.write(o+'\n')
						pl.write(l+'\n')

	if args.mode == "test":
		test_data = []
		with open(args.test, "r") as f:
			for line in f:
				ll = line.strip().split(" ")
				test_in_l = ll[1:-1]
				test_out_l = ["O"] * len(test_in_l)
				label = "O"
				test_data.append((" ".join(test_in_l), " ".join(test_out_l), label))

		if not os.path.isdir(args.dir+'/test'):
			os.makedirs(args.dir+'/test')

		with open(args.dir+'/test/test.seq.in', "w") as pin:
			with open(args.dir+'/test/test.seq.out', "w") as pout:
				with open(args.dir+'/test/test.label', "w") as pl:
					for (i, o, l) in test_data:
						pin.write(i+'\n')
						pout.write(o+'\n')
						pl.write(l+'\n')










if __name__ == "__main__":

	args = arg_parse()

	pre_format(args)



