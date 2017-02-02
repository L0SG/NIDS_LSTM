import tflearn as tfl
import tensorflow as tf
import encoding
import os

flags = tf.flags
logging = tf.logging
FILE_PATH= os.getcwd()+'/input/'
SAVE_PATH= os.getcwd()
INPUT_SIZE = 13
flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", FILE_PATH, "Where the training/test data is stored.")
flags.DEFINE_string("save_path", SAVE_PATH, "Model output directory.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS

raw_data = encoding.input_data(FLAGS.data_path)
train_data, valid_data, test_data, vocabulary = raw_data

input_data, targets = encoding.input_producer(train_data, None, None, name=None)


g = tfl.input_data(shape=[None, 10, INPUT_SIZE])
g = tfl.lstm(g, 512)
g = tfl.dropout(g, 0.5)
g = tfl.fully_connected(g, INPUT_SIZE, activation='softmax')
g = tfl.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

m = tfl.SequenceGenerator(g)
m.fit(input_data, targets)