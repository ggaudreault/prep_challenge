#!/usr/local/bin/python3
import os
import argparse
import src.corpus_building as cpb
import src.corpus_ngram as cpng
import src.prep_predictor as pcl
from src.evaluate_predictions import eval_predictions


#no need for validation

def parse_input():
	parser = argparse.ArgumentParser()
	parser.add_argument("-a", "--action", help="What do you want to do?\nOptions:\n1) ticker ngram training (ticker)\n2) full text training (train)\n3) evaluation (eval)\n4) everything (all)", default="all", choices=["build_corpus", "train_model", "predict", "all"])
	args = parser.parse_args()
	return args.action

def build_corpus():
	corpus_build = cpb.corpus_builder()
	corpus_build.import_corpus_from_dir(train_dir)
	corpus_build.dump_text(training_corpus)

def train_model():
	corpus_ngram = cpng.corpus_ngram()
	corpus_ngram.load_text(training_corpus)
	corpus_ngram.create_model(model_nm)
	corpus_ngram.write_model(ngram_model)

def predict_prepositions():
	
	prep_predictor = pcl.prep_predictor(test_data)
	prep_predictor.replace_preps()
	prep_predictor.dump_text(prep_output_path)
	#prep_predictor.load_text(predicted_text_path)
	prep_predictor.load_model(ngram_model)
	prep_predictor.predict_text()
	prep_predictor.dump_text(predicted_text_path)
	
	eval_predictions(test_data, predicted_text_path, comparison_log, result_logs)



train_dir = "data/train"
training_corpus = "output/training_corpus.txt"
# LIDSTONE BAD
#model_nm = "lidstone"
#model_nm = "kneserney"
model_nm = "wittenbell"
lemma_stem = "lemma"
#lemma_stem = "stem"
ngram_model = "output/ngram_{}.pickle".format(model_nm)
test_data = "data/test/the_hound_of_the_baskervilles.txt"
prep_output_path = "output/prep_text.txt"
predicted_text_path = "output/predicted_text_{}_{}.txt".format(model_nm, lemma_stem)
comparison_log = "output/comparison.txt"
result_logs = "output/result.log"


action = parse_input()

if action in ["build_corpus", "all"]:
	build_corpus()
if action in ["train_model", "all"]:
	train_model()
if action in ["predict", "all"]:
	predict_prepositions()