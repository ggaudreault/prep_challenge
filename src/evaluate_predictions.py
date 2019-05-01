#!/usr/local/bin/python3
import pandas as pd
import io
import re
from collections import Counter

def eval_predictions(reference_path, predicted_path, full_comparison_path="comparison.log", result_logs="results.log"):
	prep_list = ["of", "in", "to", "for", "with", "on", "at", "from", "by", "about"]
	prep_pattern = re.compile("(\W|^)({})(?=(\W|$))".format("|".join(prep_list)), re.IGNORECASE)
	header_pattern = re.compile("(^[^\*]*\n\*[^\n]*\*\n)\n*(Produced[^\n]*\n)?")
	header_sub = r""
	footer_pattern = re.compile("End of( the)* Project Gutenberg(.|\n)*$")
	footer_sub = r""
	ref_preps_full = []
	pred_preps_full = []

	with io.open(reference_path) as gp, io.open(predicted_path) as pp:
		reference_text = gp.read()
		reference_text = header_pattern.sub(header_sub, reference_text)
		reference_text = footer_pattern.sub(footer_sub, reference_text).split("\n")

		predicted_text = pp.read()
		predicted_text = header_pattern.sub(header_sub, predicted_text)
		predicted_text = footer_pattern.sub(footer_sub, predicted_text).split("\n")
		
		for index, line in enumerate(reference_text):
			ref_preps = prep_pattern.findall(line)
			pred_preps = prep_pattern.findall(predicted_text[index])
			if len(ref_preps) != len(pred_preps):
				print("BIG PROBLEM")
				print(line)
				print(predicted_text[index])
				print(ref_preps)
				print(pred_preps)
				exit()
			#print([prep[1] for prep in ref_preps])
			ref_preps_full += [prep[1].lower() for prep in ref_preps]
			pred_preps_full += [prep[1].lower() for prep in pred_preps]

	zipped_comp = zip(ref_preps_full, pred_preps_full)
	with io.open(full_comparison_path, 'w') as cp:
		for zipped in zipped_comp:
			#cp.write("{}\t{}\n".format(zipped[0][1], zipped[1][1]))
			cp.write("{}\t{}\n".format(zipped[0], zipped[1]))

	with io.open(result_logs, 'w') as rl:
		count = 0
		for i in range(len(ref_preps_full)):
			if ref_preps_full[i] == pred_preps_full[i]:
				count += 1
		accuracy = float(count)/len(ref_preps_full)


		rl.write("Total number of prepositions: {}\n\n".format(len(ref_preps_full)))

		rl.write("Total accuracy: {}%\n\n".format(round(100*accuracy)))

		ref_prep_counter = Counter(ref_preps_full)
		pred_prep_counter = Counter(pred_preps_full)
		rl.write("Count per preposition (reference)\n")
		for prep in prep_list:
			rl.write("{}: {}\n".format(prep, ref_prep_counter[prep]))
		rl.write("\n")
		rl.write("Count per preposition (predicted)\n")
		for prep in prep_list:
			rl.write("{}: {}\n".format(prep, pred_prep_counter[prep]))
		rl.write("\n")

		ref_pd = pd.Series(ref_preps_full, name='Reference')
		pred_pd = pd.Series(pred_preps_full, name="Predicted")
		confusion = pd.crosstab(ref_pd, pred_pd, rownames=['Reference'], colnames=['Predicted'], margins=True)
		print(confusion)

	with open(result_logs, 'a') as rl:
		confusion.to_csv("hello", sep="\t")




y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)





