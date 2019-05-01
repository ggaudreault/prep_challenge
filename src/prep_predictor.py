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
		#print(self.prep_pattern)
		#print("\s\({}\)\s".format("|".join(self.prep_list)))
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
		#try:
		#full_text_match = self.section_pattern.match(text, re.DOTALL)
		#full_text_match = self.section_pattern.match(text, re.MULTILINE)
		full_text_match = self.section_pattern.match(text)
		#print(full_text_match)
		#print(full_text_match.group(0))
		self.header, self.text, self.footer = full_text_match.group(1), full_text_match.group(4), full_text_match.group(6)
		self.text = "<s> <s>" + self.text + "</s> </s>"
		#print(full_text_match.group(5))
		#self.dump_text("head.txt", self.header)
		#self.dump_text("foot.txt", self.footer)
		#self.dump_text("text.txt", self.text)

	def replace_preps(self):
		self.text = self.prep_pattern.sub(self.prep_pattern_sub, self.text)


	def dump_text(self, output_path="."):
		with io.open(output_path, 'w') as op:
			op.write(self.header)
			op.write(self.text.strip("</s>"))
			op.write(self.footer)

	def predict_text(self):
		text_bak = self.text
		print("Predicting...")
		self.text = self.prep_pattern_prediction.sub(self.prep_pattern_prediction_sub, self.text)
		while text_bak != self.text:
			#print("HEYYYY BIG BOY")
			text_bak = self.text
			self.text = self.prep_pattern_prediction.sub(self.prep_pattern_prediction_sub, self.text)


	def predict_prep(self, full_match):
		#print(time.time())
		group_match = full_match.group()
		#print(time.time())
		#print(group_match)
		match = self.normalizer(group_match, underscore=False)
		#print(time.time())
		#match = self.stemmer(match)
		match = self.lemmatizer(match)
		#print(time.time())
		#print(match)


		"""
		#print(match)
		mod_match = self.rm_unnecessary_punct(match)
		#print(mod_match)
		#mod_match = self.my_tokenizer(mod_match)
		# NLTK WORD TOKENIZE DOESN'T LIKE TAGS
		#mod_match = nltk.word_tokenize(mod_match)
		#print(mod_match)
		"""

		prep = match[-1]
		context = match[-3:-1]
		#print(prep)
		#print(context)
		# we wait to replace the preposition if there's another preposition in its context
		if "__PREP__" in context:
			return group_match
		#if "have" in context or "I" in context or "”" in context:
		#	print(context)
		predicted = self.predict_from_context(context, self.prep_list)
		#print(time.time())
		#print(context)
		#print(predicted)
		#if predicted not in self.prep_list:
		#	print(predicted)
		return group_match.replace("__PREP__", predicted)

	"""
	def my_tokenizer(self, line):
		space_patt = re.compile(" +")
		space_sub = r" "
		line = line.replace("\n", " ")
		line = space_patt.sub(space_sub, line)
		return line.split(" ")


	def rm_unnecessary_punct(self, match):
		match = self.unnecessary_punct.sub(self.unnecessary_punct_sub, match)
		return match
	"""

	def predict_from_context(self, context, contenders):
		#print("---------")
		#print(context)
		prep_scores = {}
		highest_score = -float("inf")
		for prep in contenders:
			#print(prep)
			#context = ["a", "book"]
			#score = 0
			score = self.model.score(prep, context)
			#score = self.model.logscore(prep, context)
			#score2 = self.model.score(i, context[1:])
			#print(score)
			#print(score2)

			"""
			try:
				score = self.model.logscore(i, context)
			except:
				print("in")
				score = 0

			if score == 0:
				print("in")
				try:
					score = self.model.logscore(i, context[1:])
				except:
					score = 0

				if score == 0:
					try:
						score = self.model.logscore(i)
					except:
						score = 0
			"""

			prep_scores[prep] = score
			if score > highest_score:
				highest_score = score


			"""
			print("--------")
			print(context)
			print(i)
			print(self.model.counts[context][i])
			print(context[-1:])
			print(self.model.counts[context[-1:]][i])
			print(self.model.logscore(i, context[-1:]))
			print(self.model.score(i, context[-1:]))
			print("-u-")
			#print(self.model.prob)
			print(self.model.unmasked_score(i, context[2:]))
			print(self.model.unmasked_score(i, context[1:]))
			#print(self.model.unmasked_score(i, context))
			print(self.model.unmasked_score(i, context[-1:]))
			#print(self.model.logscore(i, context))
			#print(self.model.unmasked_score(i, context))
			"""
		#print(prep_scores)
		#exit()

		highest_preps = []
		for prep in contenders:
			if prep_scores[prep] == highest_score:
				highest_preps.append(prep)

		#print(prep_scores)
		#print(highest_score)
		#print(highest_preps)
		if len(highest_preps) > 1:
			if len(context) > 1:
				return self.predict_from_context(context[1:], highest_preps)
			else:
				return highest_preps[0]
		else:
			return highest_preps[0]



