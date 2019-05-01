#!/usr/local/bin/python3
import re
import os
import io
import nltk
import pickle
from nltk.stem.porter import *
import src.corpus_building as cpb
import time

class prep_predictor():
	header = ""
	text = ""
	footer = ""
	live = False
	prep_list = ["of", "in", "to", "for", "with", "on", "at", "from", "by", "about"]
	header_pattern = re.compile("(^[^\*]*\n\*[^\n]*\*\n)\n*(Produced[^\n]*\n)?")
	footer_pattern = re.compile("End of( the)* Project Gutenberg(.|\n)*$")
	section_pattern = re.compile("((^[^\*]*\n\*[^\n]*\*\n)\n*(Produced[^\n]*\n)?)((.|\n)*)(End of( the)* Project Gutenberg(.|\n)*\Z)")
	prep_pattern = re.compile("(\W)({})(?=\W)".format("|".join(prep_list)), re.IGNORECASE)
	prep_pattern_sub = lambda self, m: m.group(1) + "__PREP__"
	prep_pattern_prediction = re.compile("(\S+\s+){3}\S+\s*(\W|\-\-)?(\_\_PREP\_\_)(?=\W)", re.IGNORECASE)
	prep_pattern_prediction_sub = lambda self, m: self.predict_prep(m)
	unnecessary_punct = re.compile("[“”'\-\[\]]")
	unnecessary_punct_sub = r""
	cpb_c = cpb.corpus_builder()
	normalizer = cpb_c.normalize_line
	stemmer = cpb_c.stem_line
	lemmatizer = cpb_c.lemma_line

	def __init__(self, text_path=None):
		if text_path:
			self.load_text(text_path)

	def load_text(self, text_path):
		with io.open(text_path) as tp:
			text = tp.read()
			self.split_text_into_sections(text)

	def load_model(self, model_path):
		with io.open(model_path, 'rb') as f:
			self.model = pickle.load(f)
			self.ngram_order = self.model.order

	def split_text_into_sections(self, text):
		full_text_match = self.section_pattern.match(text)
		self.header, self.text, self.footer = full_text_match.group(1), full_text_match.group(4), full_text_match.group(6)
		self.text = "<s> <s>" + self.text + "</s> </s>"

	def replace_preps(self):
		self.text = self.prep_pattern.sub(self.prep_pattern_sub, self.text)


	def dump_text(self, output_path="."):
		with io.open(output_path, 'w') as op:
			op.write(self.header)
			op.write(self.text.strip("</s>"))
			op.write(self.footer)

	def predict_text(self, live=False):
		self.live = live
		text_bak = self.text
		print("Predicting...")
		self.text = self.prep_pattern_prediction.sub(self.prep_pattern_prediction_sub, self.text)
		while text_bak != self.text:
			print("...")			
			text_bak = self.text
			self.text = self.prep_pattern_prediction.sub(self.prep_pattern_prediction_sub, self.text)


	def predict_prep(self, full_match):
		group_match = full_match.group()
		if self.live:
			print("---")
			print(group_match)
		match = self.normalizer(group_match, underscore=False)
		match = self.lemmatizer(match)
		prep = match[-1]
		context = match[-3:-1]
		# we wait to replace the preposition if there's another preposition in its context
		if "__PREP__" in match[:-1]:
			if self.live:
				print(group_match)
			return group_match
		predicted = self.predict_from_context(context, self.prep_list)
		match_predicted = group_match.replace("__PREP__", predicted)
		if self.live:
			print(match_predicted)
		return match_predicted

	def predict_from_context(self, context, contenders):
		prep_scores = {}
		highest_score = -float("inf")
		for prep in contenders:
			score = self.model.score(prep, context)

			prep_scores[prep] = score
			if score > highest_score:
				highest_score = score

		highest_preps = []
		for prep in contenders:
			if prep_scores[prep] == highest_score:
				highest_preps.append(prep)

		if len(highest_preps) > 1:
			if len(context) > 1:
				return self.predict_from_context(context[1:], highest_preps)
			else:
				return highest_preps[0]
		else:
			return highest_preps[0]



