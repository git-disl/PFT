# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import modeling
import optimization
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import sys
import pickle as pkl
import time

flags = tf.flags
FLAGS = flags.FLAGS

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## Required parameters
flags.DEFINE_string(
    "bert_config_file", "./pplm_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "checkpointDir", "./pseudo_ckpt_dir",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("strategy", "diff", "Strategy for fake pair generation")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_integer("max_seq_length", 50, "")
flags.DEFINE_integer("max_predictions_per_seq", 5, "")
flags.DEFINE_integer("batch_size", 256, "")
flags.DEFINE_integer("epoch", 1, "")
flags.DEFINE_float("learning_rate", 1e-4, "")
flags.DEFINE_integer("num_train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("num_warmup_steps", 100, "Number of warmup steps.")
flags.DEFINE_integer("save_checkpoints_steps", 3000, "")
flags.DEFINE_integer("iterations_per_loop", 2000, "How many steps to make in each estimator call.")
flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("dataset_name", 'eth', "dataset name.")
flags.DEFINE_bool("cross", True, "whether use residual mlp")
flags.DEFINE_integer("neg_sample_num", 1024, "The number of negative samples in a batch")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")

def input_fn(input_files,
             is_training,
             num_cpu_threads=4):
    """ The actual input function"""

    name_to_features = {
        "src_address":
            tf.FixedLenFeature([1], tf.int64),
        "src_input_ids":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "src_input_positions":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "src_input_counts":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "src_input_mask":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "src_input_io_flags":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "src_input_values":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "src_masked_lm_positions":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "src_masked_lm_ids":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "src_masked_lm_weights":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32),
        # ----------------------------------------------------------------- #
        "dst_address":
            tf.FixedLenFeature([1], tf.int64),
        "dst_input_ids":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "dst_input_positions":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "dst_input_counts":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "dst_input_mask":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "dst_input_io_flags":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "dst_input_values":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "dst_masked_lm_positions":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "dst_masked_lm_ids":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.int64),
        "dst_masked_lm_weights":
            tf.FixedLenFeature([FLAGS.max_predictions_per_seq], tf.float32)
    }

    if is_training:
        d = tf.data.TFRecordDataset(input_files)
        d = d.repeat(FLAGS.epoch).shuffle(100)

    else:
        d = tf.data.TFRecordDataset(input_files)

    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
    d = d.batch(batch_size=FLAGS.batch_size)

    iterator = d.make_one_shot_iterator()
    features = iterator.get_next()

    return features


def model_fn(features, mode, bert_config, vocab, init_checkpoint, learning_rate,
             num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("name = %s, shape = %s" % (name,
                                                   features[name].shape))

    src_input_ids = features["src_input_ids"]
    src_input_positions = features["src_input_positions"]
    src_input_mask = features["src_input_mask"]
    src_input_io_flags = features["src_input_io_flags"]
    src_input_values = features["src_input_values"]
    src_input_counts = features["src_input_counts"]
    src_masked_lm_positions = features["src_masked_lm_positions"]
    src_masked_lm_ids = features["src_masked_lm_ids"]
    src_masked_lm_weights = features["src_masked_lm_weights"]
    # --------------------------------------------------------- #
    dst_input_ids = features["dst_input_ids"]
    dst_input_positions = features["dst_input_positions"]
    dst_input_mask = features["dst_input_mask"]
    dst_input_io_flags = features["dst_input_io_flags"]
    dst_input_values = features["dst_input_values"]
    dst_input_counts = features["dst_input_counts"]
    dst_masked_lm_positions = features["dst_masked_lm_positions"]
    dst_masked_lm_ids = features["dst_masked_lm_ids"]
    dst_masked_lm_weights = features["dst_masked_lm_weights"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Siamese Network for Pseudo-supervised Learning
    src_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=src_input_ids,
        input_positions=src_input_positions,
        input_io_flags=src_input_io_flags,
        input_amounts=src_input_values,
        input_counts=src_input_counts,
        input_mask=src_input_mask,
        token_type_ids=None,
        use_one_hot_embeddings=use_one_hot_embeddings)

    dst_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=dst_input_ids,
        input_positions=dst_input_positions,
        input_io_flags=dst_input_io_flags,
        input_amounts=dst_input_values,
        input_counts=dst_input_counts,
        input_mask=dst_input_mask,
        token_type_ids=None,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # length averaging
    # first delete self-transaction and then do averaging
    # src_mask = tf.cast(tf.tile(tf.expand_dims(src_input_mask, axis=2), [1, 1, bert_config.hidden_size]), tf.float32)
    # src_embeddings = tf.reduce_mean(tf.multiply(src_model.get_sequence_output(), src_mask)[:,1:,:], axis=1)
    # dst_mask = tf.cast(tf.tile(tf.expand_dims(dst_input_mask, axis=2), [1, 1, bert_config.hidden_size]), tf.float32)
    # dst_embeddings = tf.reduce_mean(tf.multiply(dst_model.get_sequence_output(), dst_mask)[:,1:,:], axis=1)

    # # directly use self transaction.
    src_embeddings = src_model.get_sequence_output()[:, 0, :]
    dst_embeddings = dst_model.get_sequence_output()[:, 0, :]

    print("src_embeddings:", src_embeddings)
    print("dst_embeddings:", dst_embeddings)
    # print("deself!!!")

    loss = get_classifier_cross_entropy(src_embeddings, dst_embeddings)
    total_loss = loss

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint,
                                              assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                 num_train_steps,
                                                 num_warmup_steps, use_tpu)

        return train_op, total_loss

    elif mode == tf.estimator.ModeKeys.EVAL:

        pass

        return total_loss

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))


def get_classifier_cross_entropy(src_embeddings, dst_embeddings):

    sequence_shape = modeling.get_shape_list(src_embeddings, expected_rank=2)
    hidden_size = sequence_shape[1]

    V = tf.constant(np.arange(FLAGS.batch_size), dtype=tf.int64)
    V = tf.expand_dims(V, axis=1)

    neg_src_ids, _, _ = tf.nn.uniform_candidate_sampler(true_classes=V,
                                                        num_true=1,
                                                        num_sampled=FLAGS.neg_sample_num,
                                                        unique=False,
                                                        range_max=FLAGS.batch_size,
                                                        seed=1234)

    neg_dst_ids, _, _ = tf.nn.uniform_candidate_sampler(true_classes=V,
                                                        num_true=1,
                                                        num_sampled=FLAGS.neg_sample_num,
                                                        unique=False,
                                                        range_max=FLAGS.batch_size,
                                                        seed=4321)

    neg_src_embeddings = tf.nn.embedding_lookup(src_embeddings, neg_src_ids)
    neg_dst_embeddings = tf.nn.embedding_lookup(dst_embeddings, neg_dst_ids)

    new_src_embeddings = tf.concat([src_embeddings, neg_src_embeddings], axis=0)
    new_dst_embeddings = tf.concat([dst_embeddings, neg_dst_embeddings], axis=0)

    label = tf.concat([tf.ones(FLAGS.batch_size, dtype=tf.float32),
                       tf.zeros(FLAGS.neg_sample_num, dtype=tf.float32)],
                       axis=0)


    with tf.variable_scope("MLP", reuse=tf.AUTO_REUSE):
        if FLAGS.cross:

            residual_inp = tf.abs(new_src_embeddings - new_dst_embeddings)
            multiply_inp = new_src_embeddings * new_dst_embeddings
            inp = tf.concat([new_src_embeddings, new_dst_embeddings, residual_inp, multiply_inp], axis=1)

        else:

            residual_inp = tf.abs(new_src_embeddings - new_dst_embeddings)
            print("No cross aware!!!")
            inp = tf.concat([new_src_embeddings, new_dst_embeddings, residual_inp], axis=1)

        dnn1 = tf.layers.dense(inp, hidden_size, activation=tf.nn.relu, name="f1")
        dnn2 = tf.layers.dense(inp, hidden_size, activation=tf.nn.relu, name="f2")
        logit = tf.squeeze(tf.layers.dense(dnn2+dnn1, 1, activation=None, name="logit"))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    return loss


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]
    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example


def run_finetune():

    mode = tf.estimator.ModeKeys.TRAIN

    input_file_name = "./data/eth.pair"
    tf.logging.info("*** Reading from input fake pair files ***")
    tf.logging.info("  %s", input_file_name)

    # load data
    features = input_fn(input_file_name, is_training=True)

    # modeling
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    train_op, total_loss = model_fn(features, mode, bert_config, vocab, FLAGS.init_checkpoint,
                                    FLAGS.learning_rate,
                                    FLAGS.num_train_steps, FLAGS.num_warmup_steps, False, False)

    # saver define
    tvars = tf.trainable_variables()
    saver = tf.train.Saver(max_to_keep=30, var_list=tvars)

    # start session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        losses = []
        iter = 0
        start = time.time()
        while True:
            try:
                _, loss = sess.run([train_op, total_loss])
                losses.append(loss)

                if iter % 100 == 0:
                    end = time.time()
                    loss = np.mean(losses)
                    print("iter=%d, loss=%f, time=%.2fs" % (iter, loss, end - start))
                    losses = []
                    start = time.time()

                if iter % FLAGS.save_checkpoints_steps == 0 and iter > 0:
                    saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(round(iter))))

                iter += 1

            except Exception as e:
                # print("Out of Sequence, end of training...")
                print(e)
                # save model
                saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(round(iter))))
                break

    return

def main(_):
    run_finetune()


if __name__ == '__main__':
    tf.app.run()