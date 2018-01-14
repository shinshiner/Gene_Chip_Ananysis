#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def pl(path, fname, label_name):
	data = []
	max_length = 0
	for i in range(-3, 2):
		if label_name != 'loss':
			data.append([0] + processfile(os.path.join(path, '1e%d'%i, '%s.txt'%fname)))
		else:
			data.append(processfile(os.path.join(path, '1e%d'%i, '%s.txt'%fname)))
		max_length = max(max_length, len(data[-1]))
	data = list(map(lambda x: pad(x, max_length), data))
	print(data)
	fig = plt.figure(num=1, figsize=(8, 6))
	ax = fig.add_subplot(111)
	if label_name != 'loss':
		ax.axis([0, max_length, 0, 100])
	for i in range(5):
		ax.plot(np.arange(max_length), data[i], label="$\lambda = 10^{%d}$"%(i-3))
	plt.legend(loc='upper right')
	ax.set_xlabel('epoch')
	ax.set_ylabel(label_name)
	#plt.show()
	plt.savefig(os.path.join(path, '%s.png'%fname))
	plt.close('all')

def pad(x, length):
	res = np.zeros((length))
	l = len(x)
	for i in range(l):
		res[i] = x[i]
	for i in range(l, length):
		res[i] = res[i - 10]
	return res

def processfile(fname):
	with open(fname) as f:
		s = f.readlines()
		res = list(map(lambda x: float(x), s))
	return res

if __name__ == '__main__':
	#pl('SVM', 'acc_train', 'accuracy')
	#pl('SVM', 'acc_test', 'accuracy')
	#pl('SVM', 'loss', 'loss')
	pl('Softmax', 'acc_train', 'accuracy')
	pl('Softmax', 'acc_test', 'accuracy')
	pl('Softmax', 'loss', 'loss')