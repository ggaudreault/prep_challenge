#!/usr/local/bin/python3
import os
import io
import pickle
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Lidstone, WittenBellInterpolated, KneserNeyInterpolated, Vocabulary


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

	def create_model(self, model_nm):
		self.model = {"lidstone": Lidstone(0.5, self.ngram_order),
			"kneserney": KneserNeyInterpolated(self.ngram_order),
			"wittenbell": WittenBellInterpolated(self.ngram_order)
			}[model_nm]
		train, vocab = padded_everygram_pipeline(self.ngram_order, self.text)
		vocab = Vocabulary(vocab, unk_cutoff=2, unk_label="<UNK>")
		print("Creating ngram...")
		self.model.fit(train, vocab)
		print("done")

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
