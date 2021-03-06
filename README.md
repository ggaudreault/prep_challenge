# prep_challenge


Task description

1) Tag all occurrences of the ten most common english prepositions in the text The Hound of the Baskerville and output the resulting text
2) Write a script that would navigate through the modified copy of the text and accurately predict the original preposition


Approach

I have chosen to go with a simple approach here:
- Train an n-gram model on a corpus of other A. Conan Doyle texts -- about a million words
- At each __PREP__ position, look back at the previous tokens and let the n-gram decide which preposition is more likely to occur

I have divided the task into three steps:
1) Corpus building: I have gathered the texts from http://www.gutenberg.org . Each text was normalized and lemmatized to only keep the core representation of each token. The header and footer sections were removed

2) Model training: Using nltk, I have generated a trigram model from the training corpus. The standard Lidstone and interpolated Kneser-Ney models did not go well with the prediction decision rule I decided to use, as they tend to output identical scores on unseen trigrams, hence making the decision to pick a winner difficult. Instead I used a Witten-Bell smoothing

3) Preposition predictor: The code I wrote works directly on the text as it is formatted, rather than re-formatting it completely and then parsing it. The final output text with the predictions can then simply be diff'd with the original one for easy comparison. The predictor works as follows:
- Using regex, it looks for the __PREP__ token and the words preceeding it
- When it finds such a group, it normalizes it, then split it on space characters
- The context w_0 w_1 is then passed to a function that generates using the n-gram model the score w_0 w_1 PREP for each of the ten most common prepositions
- The preposition with the best score is returned as winner
- If the top score is shared by more than one preposition, the process is repeated on the context[1:] 


Repo

- data: divided into "train" -- where the training corpora is kept -- and "test" -- where The Hounds of the Baskervilles is placed
- src:
- - corpus_building: class to use to normalize the text and also generate a training corpus
- - corpus_ngram: class to use to generate an n-gram model from some corpus
- - prep_predictor: used to replace all prepositions by __PREP__ and to then predict the value it should take using the n-gram model
- - evaluate_predictions: compares the original text to the predicted one, and outputs a simple log file containing metrics and accuracy data
- run.py: this script calls the above scripts and contains task-specific parameters. Pass of the four parameters as "action": build_corpus, train_model, predict, all


To-do

1) Figure out why the accuracy is so low. 50% overall accuracy is not the worst, but is not great either. I expected this kind of approach to give me a higher accuracy. More common prepositions like "of" and "to" scored well, but that is probably caused by their overall popularity. Maybe I overlooked something or mis-coded a part of the scripts

2) Improve the prediction process. This approach of looking at tri-grams, then bi-grams, then uni-grams is not accurate enough. A quick way to improve this might be to also have a look at the following word, e.g. 
- maximize preplexity(context[-2] + __PREP__ + forward_context[:2])
or simply implement a logistic regressor on the tri/bi/uni-grams and forward bi/tri-grams

3) Using NLTK, it took me about 80 minutes to predict 5975 prepositions, which is very long. I'd like to look into how to use other libraries to implement something similar, e.g. SpaCy, scikit-learn, kenlm. NLTK seemed to me to be the most straight-forward library, as the first two do not, as far as I know, have any comprehensive n-gram libraries -- no n-gram scoring, smoothing, or backoff -- and I haven't used kenlm yet

4) Document. The code I wrote isn't long -- around 450-500 lines -- and so isn't too hard to read, but it would be a good thing to sit down and document it more.

5) Catch failing cases and cases for exceptions, make the code more robust


