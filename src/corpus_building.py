#!/usr/local/bin/python3
import re
import os
import io
import nltk
from nltk.stem.porter import *

class corpus_builder():
	output_path = ""
	header_pattern = re.compile("(^[^\*]*\n\*[^\n]*\*\n)\n*(Produced[^\n]*\n)?")
	header_sub = r""
	footer_pattern = re.compile("End of( the)* Project Gutenberg(.|\n)*$")
	footer_sub = r""
	newline_pattern = re.compile("\n")
	newline_pattern_sub = r" "
	unnecessary_newline_pattern = re.compile("\n\n+")
	unnecessary_newline_pattern_sub = r"\n"
	stemmer = PorterStemmer()


	def __init__(self):
		self.full_corpus = []

	def import_corpus_from_dir(self, directory):
		corpora = os.listdir(directory)
		for corpus in corpora:
			self.add_text(os.path.join(directory, corpus))

	def add_text(self, text_path):
		text = self.normalize_text_file(text_path)
		self.append_text_to_corpus(text)

	def append_text_to_corpus(self, text):
		#self.full_corpus += ["<s>"] + text + ["</s>"] 
		self.full_corpus += text

	def normalize_text_file(self, text_path):
		print(text_path)
		with io.open(text_path) as tp:
			text = tp.read()
		return self.normalize_full_text(text)


	def normalize_full_text(self, text=None):
		if text is None:
			text = self.full_corpus

		# splitting this in a bunch of operations for later when we'll only need some but not all of them
		text = self.remove_header(text)
		text = self.remove_footer(text)
		text = self.stringify(text)
		text = self.normalize_line(text)
		text = self.stem_line(text)
		return text

	def normalize_line(self, text):
		text = nltk.word_tokenize(text)
		return text

	def stem_line(self, text):
		#text = [self.ps(t) for t in text]
		text = [self.stemmer.stem(t) for t in text]
		return text

	def remove_header(self, text):
		#print(text[:200])
		text = self.header_pattern.sub(self.header_sub, text, re.DOTALL)
		#print(text[:200])
		return text

	def stringify(self, text):
		text = self.newline_pattern.sub(self.newline_pattern_sub, text)
		text = self.unnecessary_newline_pattern.sub(self.unnecessary_newline_pattern_sub, text)
		return text

	def remove_footer(self, text):
		#print(text[-200:])
		text = self.footer_pattern.sub(self.footer_sub, text, re.DOTALL)
		#print(text[-200:])
		return text

	def re_init_corpus(self, corpus_path):
		os.remove(self.corpus_path)
		print("Deleted corpus at: {}".format(self.corpus_path))

	def dump_text(self, output_path="."):
		with io.open(output_path, 'w') as op:
			print
			op.write(" ".join(self.full_corpus))


