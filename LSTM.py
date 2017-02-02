from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import numpy as np
import tensorflow as tf
import encoding


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


def data_type():
      return tf.float16 if FLAGS.use_fp16 else tf.float32


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to AD data directory")
    raw_data = encoding.input_data(FLAGS.data_path)
    train_data, valid_data, test_data, vocabulary = raw_data
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                config.init_scale)
        with tf.name_scope("Train"):
            train_input = ADInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = ADModel(is_training=True, config=config, input_=train_input, vocab    =vocabulary)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)
        with tf.name_scope("Valid"):
            valid_input = ADInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = ADModel(is_training=False, config=config, input_=valid_input,     vocab=vocabulary)
            tf.summary.scalar("Validation Loss", mvalid.cost)
        with tf.name_scope("Test"):
            test_input = ADInput(config=config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = ADModel(is_training=False, config=eval_config,
                        input_=test_input, vocab=vocabulary)
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                        verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
if __name__ == "__main__":
    tf.app.run()