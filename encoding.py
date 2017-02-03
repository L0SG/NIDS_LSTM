from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import tensorflow as tf

INPUT_SIZE=13

# idx of discrete and continuous features
idx_discrete = [0, 1, 3, 6, 8]
idx_continuous = [2, 4, 5, 7, 9, 10, 11, 12, 13]

# separator between log samples
# sequence will be (first FW log line) | (second FW log line) | ... | (last log line)
# last line is a time window, which is manual at this point
sample_separator = '-2'
time_window = 10

# token for elements not in the training vocabulary
unknown_token = '-1'

def _read_set(dirname):
    filenames = os.listdir(dirname)
    inputs=[]
    for i in range(INPUT_SIZE):
        inputs.append([])
    for filename in filenames:
        full_filename = os.path.join(dirname,filename)
        f=open(full_filename, "r")
        lines = f.readlines()
        for line in lines:
            inputline = line.split()
            for i in range(INPUT_SIZE):
                inputs[i].append(inputline[i])

        # why is this needed?
        # adds symbol after each .txt file
        # it makes inputs[0] = [ feature[0] from first file, -1, feature[0] from second file, -1 , ...]
        """
        for i in range(INPUT_SIZE):
            if i in idx_discrete:
                inputs[i].append(-1)
            else:
                inputs[i].append("<eol>")
        """
    return inputs

"""
def _read_set(dirname):
    filenames = os.listdir(dirname)
    inputs=[]
    for file in filenames:
        full_filename = os.path.join(dirname, filename)
        f=open(full_filename, 'r')
        lines = f.readlines()
        for line in lines:
            inputline = line.split()
            inputs.append(inputline)
"""

def _build_vocab(inputs):
    counter = collections.Counter(inputs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(inputs, word_to_id):
    # return [word_to_id[word] for word in inputs if word in word_to_id]
    # the above method shrinks the length of certain features if not in vocabulary
    # this makes inconsistent input length per feature

    # fill in unknown_token if not in vocabulary instead
    results = []
    for word in inputs:
        try:
            ids = word_to_id[word]
        except KeyError:
            ids = unknown_token
        results.append(ids)
    return results


def input_data(data_path=os.getcwd()):
    train_path = os.path.join(data_path, "train/")
    valid_path = os.path.join(data_path, "valid/")
    test_path = os.path.join(data_path, "test/")
    train_input = _read_set(train_path)
    valid_input = _read_set(valid_path)
    test_input = _read_set(test_path)
    vocabulary = {}

    # generate vocabulary per feature
    # likely to generate the same number between different features, reducing performance
    for i in range(INPUT_SIZE):
        if i in idx_continuous:
            word_to_id = _build_vocab(train_input[i])
            train_input[i]= _file_to_word_ids(train_input[i],word_to_id)
            valid_input[i]= _file_to_word_ids(valid_input[i],word_to_id)
            test_input[i]= _file_to_word_ids(test_input[i],word_to_id)
            vocabulary[i]=len(word_to_id)
    return train_input, valid_input, test_input, vocabulary


def input_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "inputProducer", [raw_data, batch_size, num_steps]):

        # convert raw data type from string to int
        len_raw_data = len(raw_data)
        raw_data_int_converted = []
        for i in xrange(0, len_raw_data):
            raw_data_int_converted.append(map(int, raw_data[i]))
        raw_data = raw_data_int_converted

        # convert to tensor format
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        # now raw_data has [num_features, num_samples] format

        # transpose the data for sequence generation
        # raw_data = tf.transpose(raw_data)

        # tf.size returns num_features x num_samples
        data_len = tf.size(raw_data)

        # reshape the data to [num_features x time_window, num_batch]
        # this makes the set [ [seq of 10 FW logs], [another seq of 10 FW logs], ...]
        # implement here

        # make x and y
        # y is the prediction for x
        # just make shifted-by-one matrices for testing
        x = tf.slice(raw_data, [0, 0], [int(raw_data.get_shape()[0]), int(raw_data.get_shape()[1])-1])
        y = tf.slice(raw_data, [1, 0], [int(raw_data.get_shape()[0]), int(raw_data.get_shape()[1])-1])

        return x, y



        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0:batch_size*batch_len][:], [batch_size, batch_len])
        epoch_size = (batch_len-1) // num_steps
        assertion = tf.assert_positive(epoch_size,
                message="spoch_size ==0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                [batch_size, (i+1) * num_steps],[1,1])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i*num_steps + 1],
                [batch_size, (i+1) * num_steps+ 1],[1,1])
        y.set_shape([batch_size, num_steps])
        return x, y