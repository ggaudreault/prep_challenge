#!/usr/local/bin/python3
import os
import argparse
import src.corpus_building as cpb


def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("-a", "--action", help="What do you want to do?\nOptions:\n1) ticker ngram training (ticker)\n2) full text training (train)\n3) evaluation (eval)\n4) everything (all)", default="all", choices=["build_corpus", "train_model", "classify", "all"])
	args = parser.parse_args()
	return args.action

def build_corpus():
	corpus_build = cpb.corpus_builder()
	corpus_build.import_corpus_from_dir(train_dir)
	corpus_build.dump_text(training_corpus)


train_dir = "data/test"
training_corpus = "output/training_corpus.txt"

action = parse_input()

if action in ["build_corpus", "all"]:
	build_corpus()