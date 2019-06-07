import io

#embeddings_index = dict()
with io.open('glove.6B/glove.6B.50d.txt') as glove, io.open("output/glove_reduced.txt", 'w') as gred, io.open("output/vocab.txt") as vocab_f:
	vocab = vocab_f.read().split("\n")
	glove_vocab = []
	print(len(vocab))
	for line in glove:
		values = line.split()
		word = values[0]
		glove_vocab.append(word)
		if word in vocab:
			#print(word)
			gred.write(line)
			#coefs = asarray(values[1:], dtype='float32')
			#embeddings_index[word] = coefs

	with io.open("output/glove_vocab.txt", 'w') as gv:
		gv.write("\n".join(glove_vocab))