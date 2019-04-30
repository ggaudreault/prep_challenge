#!/usr/local/bin/python3
import os
import io
import pickle
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Lidstone


class corpus_ngram():
	text = []
	model = None
	unticked = 0
	tickered = 0
	ngram_order = 3

	def __init__(self, text=[]):
		self.text = text

	def load_text(self, file_path):
		with io.open(file_path) as fp:
			text_full = fp.read().split("\n")
			self.text = [sent.strip().split(" ") for sent in text_full]

	def create_model(self):
		self.model = Lidstone(0.5, self.ngram_order)
		train, vocab = padded_everygram_pipeline(self.ngram_order, self.text)
		#for i in train:
		#	for j in i:
		#		print(j)
		#print(train)
		print("creating ngram")
		self.model.fit(train, vocab)
		print("done")
		"""
		print(self.model.counts["<TBD>"])
		print(self.model.score("at"))
		print(self.model.score("<TBD>"))
		print(self.model.counts[["<TICKER>"]]["<TBD>"])
		print(self.model.counts[["<TICKER>", "<TICKER>"]]["<TBD>"])
		print(self.model.score("<TBD>", ["<TICKER>"]))
		print(self.model.score("<TBD>", ["<TICKER>", "<TICKER>"]))
		print(self.model.context_counts(("<TICKER>", "<TICKER>")))
		print(self.model.context_counts(("<TICKER>")))
		print(self.model.counts.N())
		print(self.model.counts[2].keys())
		print(self.model.counts[["<TICKER>"]].items())
		"""

	def write_model(self, model_path="model.pickle"):
		try:
			os.makedirs(os.path.dirname(model_path))
		except:
			pass
		with io.open(model_path, 'wb') as f:
			pickle.dump(self.model, f)


	def load_model(self, model_path):
		with io.open(model_path, 'rb') as f:
			self.model = pickle.load(f)
			self.ngram_order = self.model.order
