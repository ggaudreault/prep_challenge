# prep_challenge


Task description

1) Tag all occurrences of the ten most common english prepositions in the text The Hound of the Baskerville and output the resulting text
2) Write a script that would navigate through the modified copy of the text and accurately predict the original preposition


Approach

I have chosen to go with a simple approach here:
- Train an n-gram model on a corpus of other A. Conan Doyle texts -- about a million words
- At each __PREP__ position, look back at the previous tokens and let the n-gram decide which preposition is more likely to occur

I have divided the task into three operations:
1) Corpus building: I have gathered the texts from http://www.gutenberg.org . Each text was normalized and STEMMED to only keep the core representation of each token. The header and footer were removed

2) Model training: Using nltk, I have generated a trigram model from the training corpus. The standard Lidstone and interpolated Witten-Bell models did not go well with the prediction decision rule I decided to use, as they tend to output identical scores on unseen contexts, hence making the decision to pick a winner difficult.

3) Preposition predictor: The code I wrote works directly on the text as it is formatted, rather than re-formatting it completely and then parsing it. The predictor works as follows:
- Using regex, it looks for the __PREP__ token and the words preceeding it
- When it finds such a group, it normalizes it, then split it on space characters
- The context w_0 w_1 is then passed to a function that generates using the n-gram model the score w_0 w_1 PREP for each of the ten most common prepositions
- The preposition with the best score is returned as winner
- If the top score is shared by more than one preposition, the process is repeated on the context[1:] 


ADD FILES AND EVAL FN