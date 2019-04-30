#!/usr/local/bin/python3
import os
import argparse
import src.corpus_building as cpb
import src.corpus_ngram as cpng


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("-a", "--action", help="What do you want to do?\nOptions:\n1) ticker ngram training (ticker)\n2) full text training (train)\n3) evaluation (eval)\n4) everything (all)", default="all", choices=["build_corpus", "train_model", "classify", "all"])
	args = parser.parse_args()
	return args.action

def build_corpus():
	corpus_build = cpb.corpus_builder()
	corpus_build.import_corpus_from_dir(train_dir)
	corpus_build.dump_text(training_corpus)

def train_model():
	corpus_ngram = cpng.corpus_ngram()
	corpus_ngram.load_text(training_corpus)
	corpus_ngram.create_model()
	corpus_ngram.write_model(ngram_model)

def classify_prepositions():
	corpus_build = cpb.corpus_builder()
	


train_dir = "data/train"
training_corpus = "output/training_corpus.txt"
ngram_model = "output/ngram.pickle"
test_data = "data/test/the_hound_of_the_baskervilles.txt"


action = parse_input()

if action in ["build_corpus", "all"]:
	build_corpus()
if action in ["train_model", "all"]:
	train_model()
if action in ["classify", "all"]:
	classify_prepositions()