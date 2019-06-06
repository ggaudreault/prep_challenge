from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import io
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print(tf.__version__)

"""
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1])
# A dictionary mapping words to an integer index

word_index = imdb.get_word_index()



# The first indices are reserved

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0

word_index["<START>"] = 1

word_index["<UNK>"] = 2  # unknown

word_index["<UNUSED>"] = 3



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[0])
train_data = keras.preprocessing.sequence.pad_sequences(train_data,

                                                        value=word_index["<PAD>"],

                                                        padding='post',

                                                        maxlen=256)



test_data = keras.preprocessing.sequence.pad_sequences(test_data,

                                                       value=word_index["<PAD>"],

                                                       padding='post',

                                                       maxlen=256)
len(train_data[0]), len(train_data[1])
print(train_data[0])
# input shape is the vocabulary count used for the movie reviews (10,000 words)



# not yet
#training_text = [full_text[index-2:index] + full_text[index+1:index+3] for word, index in enumerate(full_text[2:-3])]
#for i,v in enumerate(full_text[2:-3]):
#  if v in prep_list:
#    print(full_text[i-2:i])

"""









def build_word_index(vocab):
  word_index = {word:vocab.index(word) for word in vocab}
  return word_index


prep_list = ["of", "in", "to", "for", "with", "on", "at", "from", "by", "about"]
training_text = "output/training_corpus_glove_lemma.txt"
corpus_out = "output/corpus_out."

with io.open(training_text) as tt:
  full_text = tt.read().split(" ")
  vocab = list(set(full_text))

with io.open("output/vocab.txt", 'w') as vcb:
  vcb.write("\n".join(vocab))

word_index = build_word_index(vocab)

text_data = [full_text[index:index+3] + full_text[index+4:index+7] for index, word in enumerate(full_text[3:-4]) if word in prep_list]
text_data = [[word_index[w] for w in quatro] for quatro in text_data]

text_labels = [prep_list.index(word) for word in full_text if word in prep_list]

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

vocab_size = len(vocab)

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
print(x_val.shape)
print(y_val.shape)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#x_val = train_data[:10000]
#partial_x_train = train_data[10000:]

#y_val = train_labels[:10000]
#partial_y_train = train_labels[10000:]

history = model.fit(train_data, train_labels, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
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
