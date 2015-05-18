import gensim
import logging
import os

# TEST_FILENAME = 'word2vec-0.7.1/train_test.txt'
TEST_FILENAME = 'train_test.txt'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.split()

sentences1 = [['first', 'sentence'], ['second', 'sentence']]
sentences2 = [['third', 'sentence'], ['fourth', 'sentence']]
model = gensim.models.Word2Vec(sentences1, min_count=1)
model.build_vocab(sentences1)
model.train(sentences2)
# model.accuracy(TEST_FILENAME)