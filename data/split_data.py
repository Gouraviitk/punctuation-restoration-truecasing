import csv, numpy as np, sys

def process_and_split(file):

	train = open('train.tsv','w')
	val = open('val.tsv','w')
	test = open('test.tsv','w')

	with open(file, 'r', encoding='UTF-8') as f:
		lines = [line for line in f.read().split('\n') if line.strip()]

		for line in lines:
			x = np.random.rand()

			if x < 0.08:
				train.writelines(line+'\n')
			elif x <0.09:
				val.writelines(line+'\n')
			elif x<0.1:
				test.writelines(line+'\n')

	train.close()
	val.close()
	test.close()


if __name__=="__main__":
	f_path = sys.argv[1]
	process_and_split(f_path)