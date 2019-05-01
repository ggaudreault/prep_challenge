#!/usr/local/bin/python3
import re
import os
import io
import nltk
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

class corpus_builder():
	output_path = ""
	patterns = {"header_pattern" : re.compile("(^[^\*]*\n\*[^\n]*\*\n)\n*(Produced[^\n]*\n)?"),
		"header_sub" : r"",
		"footer_pattern" : re.compile("End of( the)* Project Gutenberg(.|\n)*$"),
		"footer_sub" : r"",
		"newline_pattern" : re.compile("\n"),
		"newline_pattern_sub" : r" ",
		"unnecessary_newline_pattern" : re.compile("\n\n+"),
		"unnecessary_newline_pattern_sub" : r"\n",
		"unnecessary_punct" : re.compile("['\-\_,]"),
		"unnecessary_punct_sub" : r" ",
		"unnecessary_punct_alt" : re.compile("['\-,]"),
		"unnecessary_punct_alt_sub" : r" ",
		"sentence_markers_begin" : re.compile("[\(\[“]"),
		"sentence_markers_begin_sub" : r" <s> ",
		"sentence_markers_end" : re.compile("[\.\?\!\(\)\[\]”]"),
		"sentence_markers_end_sub" : r" </s> ",
		"sentence_markers_double" : re.compile("(\-\-|:|;)"),
		"sentence_markers_double_sub" : r"</s> <s>",
		"mult_sentences_begin": re.compile("<s>( <s>)+"),
		"mult_sentences_begin_sub": r"<s>",
		"mult_sentences_end": re.compile("</s>( </s>)+"),
		"mult_sentences_end_sub": r"</s>",
		"space" : re.compile("\s+"),
		"space_sub" : r" "
	}
	stemmer = PorterStemmer()
	lemmatizer = WordNetLemmatizer()


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
		self.full_corpus += text

	def _sub_pattern(self, patt_key, patt_sub, line):
		return self.patterns[patt_key].sub(self.patterns[patt_sub], line)

	def normalize_text_file(self, text_path):
		print("Normalizing corpus {}".format(text_path))
		with io.open(text_path) as tp:
			text = tp.read()
		return self.normalize_full_text(text)


	def normalize_full_text(self, text=None):
		if text is None:
			text = self.full_corpus

		# splitting this in a bunch of operations for later when we'll only need some but not all of them
		text = self.remove_header(text)
		text = self.remove_footer(text)
		text = self.normalize_line(text)
		text = self.lemma_line(text)
		return text


	def normalize_line(self, line, underscore=True):
		if underscore:
			ops = ["newline_pattern", "sentence_markers_begin", "sentence_markers_end", "sentence_markers_double", "unnecessary_punct", "mult_sentences_begin", "mult_sentences_end", "space"]
		else:
			ops = ["newline_pattern", "sentence_markers_begin", "sentence_markers_end", "sentence_markers_double", "unnecessary_punct_alt", "mult_sentences_begin", "mult_sentences_end", "space"]

		for op in ops:
			line = self._sub_pattern(op, op + "_sub", line)

		return line.split(" ")

	def stem_line(self, text):
		text = [self.stemmer.stem(t) for t in text]
		return text

	def lemma_line(self, text):
		text = [self.lemmatizer.lemmatize(t) for t in text]
		return text

	def remove_header(self, text):
		text = self.patterns["header_pattern"].sub(self.patterns["header_sub"], text, re.DOTALL)
		return text

	def stringify(self, text):
		text = self.newline_pattern.sub(self.newline_pattern_sub, text)
		text = self.unnecessary_newline_pattern.sub(self.unnecessary_newline_pattern_sub, text)
		return text


	def rm_unnecessary_punct(self, match):
		match = self.unnecessary_punct.sub(self.unnecessary_punct_sub, match)
		return match


	def remove_footer(self, text):
		text = self.patterns["footer_pattern"].sub(self.patterns["footer_sub"], text, re.DOTALL)
		return text

	def re_init_corpus(self, corpus_path):
		os.remove(self.corpus_path)
		print("Deleted corpus at: {}".format(self.corpus_path))

	def dump_text(self, output_path="."):
		with io.open(output_path, 'w') as op:
			op.write(" ".join(self.full_corpus))


