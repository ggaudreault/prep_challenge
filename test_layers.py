from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import io
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import array
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l1

print(tf.__version__)


EMB_DIM = 50
INPUT_LENGTH = 6


def build_word_index(vocab):
  word_index = {word:vocab.index(word) for word in vocab}
  return word_index


prep_list = ["of", "in", "to", "for", "with", "on", "at", "from", "by", "about"]
training_text = "output/training_corpus_glove_lemma.txt"
corpus_out = "output/corpus_out."

with io.open(training_text) as tt:
  full_text = [".", "."] +  tt.read().split(" ") + [".", "."]
  #vocab = list(set(full_text))


#with io.open("output/vocab.txt", 'w') as vcb:
#  vcb.write("\n".join(vocab))

#word_index = build_word_index(vocab)

text_data = [full_text[index:index+3] + full_text[index+4:index+7] for index, word in enumerate(full_text[3:-4]) if word in prep_list]
#text_data = [full_text[index+4:index+7] for index, word in enumerate(full_text[:-4]) if word in prep_list]


text_labels = [prep_list.index(word) for word in full_text if word in prep_list]

#print(text_data[:10])
#print(text_labels[:10])

tokens = Tokenizer()
#tokens.fit_on_texts([full_text])
tokens.fit_on_texts(text_data)
vocab_size = len(tokens.word_index) + 1
text_data = tokens.texts_to_sequences(text_data)

#text_data = [[word_index[w] for w in quatro] for quatro in text_data]

emb_index = {}
with io.open("output/glove_reduced_{}.txt".format(EMB_DIM)) as glove:
  for line in glove:
    word_emb = line.split()
    word = word_emb[0]
    weights = asarray(word_emb[1:], dtype='float32')
    emb_index[word] = weights


emb_matrix = zeros((vocab_size, EMB_DIM))
for word, index in tokens.word_index.items():
  emb_vector = emb_index.get(word)
  if emb_vector is not None:
    emb_matrix[index] = emb_vector


print(len(text_labels))
print(len(text_data))
print(text_labels[:5])
print(text_data[:5])

#with io.open(corpus_out, 'w') as co:
#  co.write(str(zip(training_text, text_labels)))

test_data = np.array(text_data[:5000])
train_data = np.array(text_data[15000:])
test_labels = to_categorical(np.array(text_labels[:5000]))
train_labels = to_categorical(np.array(text_labels[15000:]))
x_val = np.array(text_data[5000:15000])
y_val = to_categorical(np.array(text_labels[5000:15000]))

#vocab_size = len(vocab)

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
print(x_val.shape)
print(y_val.shape)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, EMB_DIM, weights=[emb_matrix], input_length=INPUT_LENGTH, trainable=True)),
#model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Concatenate(axis=-1))
model.add(keras.layers.Reshape((INPUT_LENGTH * EMB_DIM,)))
model.add(keras.layers.Dense(50, activation=tf.nn.relu))
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax, activity_regularizer=l1(0.1)))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#x_val = train_data[:10000]
#partial_x_train = train_data[10000:]

#y_val = train_labels[:10000]
#partial_y_train = train_labels[10000:]

#history = model.fit(train_data, train_labels, epochs=200, batch_size=512, validation_data=(x_val, y_val), verbose=1)
history = model.fit(train_data, train_labels, epochs=10, batch_size=512, validation_data=(x_val, y_val), verbose=1)
#history = model.fit(train_data, train_labels, epochs=40, batch_size=512, verbose=1)

results = model.evaluate(test_data, test_labels)



print(results)
history_dict = history.history

history_dict.keys()




acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.show()
