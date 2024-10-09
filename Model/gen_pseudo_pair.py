# -*- coding:utf-8 -*-
import numpy.random
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import sys
from vocab import FreqVocab
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import six
import multiprocessing
import time
import collections
from itertools import permutations

tf.logging.set_verbosity(tf.logging.INFO)

random_seed = 12345
rng = random.Random(random_seed)

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", 50, "max sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 5, "max_predictions_per_seq.")
flags.DEFINE_float("masked_lm_prob", 0.1, "Masked LM probability.")
flags.DEFINE_integer("sliding_step", 15, "sliding window step size.")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("dataset_name", 'eth', "dataset name.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(
    ",")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

ENS_set = {"0x283af0b28c62c092c9727f1ee09c02ca627eb7f5", "0x4976fb03c32e5b8cfe2b6ccb31c09ba78ebaba41",
           "0x57f1887a8bf19b14fc0df6fd9b2acc9af147ea85", "0xff252725f6122a92551a5fa9a6b6bf10eb0be035"}

# ENS_set = {}

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address, tokens, masked_lm_positions, masked_lm_labels):

        self.address = [address]
        self.tokens = list(map(lambda x: x[0], tokens))
        self.block_timestamps = list(map(lambda x: x[2], tokens))
        self.values = list(map(lambda x: x[3], tokens))

        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        self.io_flags = list(map(map_io_flag, tokens))
        self.cnts = list(map(lambda x: x[5], tokens))
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = "address: %s\n" % (self.address[0])
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_embedding_predictions(tokens):
    """Creates the predictions for the masked LM objective."""
    address = tokens[0][0]
    output_tokens = tokens
    masked_lm_positions = []
    masked_lm_labels = []
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def gen_embedding_samples(sequences):
    instances = []
    # create train
    start = time.time()
    for tokens in sequences:
        (address, tokens, masked_lm_positions,
         masked_lm_labels) = create_embedding_predictions(tokens)
        instance = TrainingInstance(
            address=address,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, rng):
    """Creates the predictions for the masked LM objective."""

    address = tokens[0][0]
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = [list(i) for i in tokens]  # note that change the value of output_tokens will also change tokens
    # num_to_predict = min(max_predictions_per_seq,
    #                      max(1, int(round(len(tokens) * masked_lm_prob))))
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(len(tokens) * masked_lm_prob)))
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = "[MASK]"
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index][0]))
        output_tokens[index][0] = masked_token

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position


# pos_seq_pair_list
def write_instance_to_example_files(instance_pairs, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    def parser(instance, flag):

        assert flag in ("src", "dst")

        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        address = vocab.convert_tokens_to_ids(instance.address)
        counts = instance.cnts
        block_timestamps = instance.block_timestamps
        values = instance.values
        io_flags = instance.io_flags
        positions = convert_timestamp_to_position(block_timestamps)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features[flag + "_address"] = create_int_feature(address)
        features[flag + "_input_ids"] = create_int_feature(input_ids)
        features[flag + "_input_positions"] = create_int_feature(positions)
        features[flag + "_input_counts"] = create_int_feature(counts)
        features[flag + "_input_io_flags"] = create_int_feature(io_flags)
        features[flag + "_input_values"] = create_int_feature(values)

        features[flag + "_input_mask"] = create_int_feature(input_mask)
        features[flag + "_masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features[flag + "_masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features[flag + "_masked_lm_weights"] = create_float_feature(masked_lm_weights)

        return features

    for inst_index in tqdm(range(len(instance_pairs))):

        src_instance = instance_pairs[inst_index][0]
        dst_instance = instance_pairs[inst_index][1]
        src_features = parser(src_instance, "src")
        features = parser(dst_instance, "dst")

        features.update(src_features)
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

        if inst_index < 3:
            tf.logging.info("*** Example ***")
            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)

def difftime_sequence_split():

    # load sequence first
    print("===========Load Sequence===========")
    with open("./data/eoa2seq.pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    # reduce ens address
    eoa2seq_new = {}
    cnt = 0
    for eoa, seq in eoa2seq.items():
        seq_new = []
        for trans in seq:
            if trans[0] in ENS_set:
                cnt += 1
                continue
            seq_new.append(trans)

        if len(seq_new)>1:
            eoa2seq_new[eoa] = seq_new

    print("ENS_trans_cnt:", cnt)
    eoa2seq = eoa2seq_new
    # clip
    # sliding_step = FLAGS.max_seq_length - 1
    sliding_step = FLAGS.sliding_step - 1
    max_num_tokens = FLAGS.sliding_step - 1
    seqs = []
    idx = 0

    # My split strategy:
    # if length <=3, no split
    # if 3< length <= 2*max_sequence_length (100)
    # if 2*max_sequence_length < length,
    # if after split, beg_idx > 500, then downsample to 500

    print("===========Generate Splited Sequence Pair Samples==========")

    for eoa, seq in eoa2seq.items():
        if len(seq) <= 3:
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq
            idx += 1

        elif len(seq) > 3 and len(seq) <= 2* max_num_tokens:
            half = round(len(seq)/2)
            # former
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq[:half]
            idx += 1
            # later
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq[half:]
            idx += 1

        elif len(seq) > 2*max_num_tokens:

            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * sliding_step))
            beg_idx.append(0)

            if len(beg_idx) > 100:
                beg_idx = list(np.random.permutation(beg_idx)[:100])
                for i in beg_idx:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

            else:
                for i in beg_idx[::-1]:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

    return seqs

def sequence_combination(seq_instance_list):
    pos_seq_pair_list = []
    address_to_seq = {}

    for seq_instance in seq_instance_list:
        eoa = seq_instance.address[0] # address is a list containing only one element
        try:
            address_to_seq[eoa].append(seq_instance)
        except:
            address_to_seq[eoa] = [seq_instance]

    pbar = tqdm(total=len(address_to_seq))
    for address in address_to_seq.keys():
        seq_list = address_to_seq[address]
        length = len(seq_list)
        comb_list = list(permutations(list(range(length)), 2))

        # shuffle
        comb_list = list(map(lambda x: (x[0], x[1]) if random.randint(0, 1) else (x[1], x[0]), comb_list))
        random.shuffle(comb_list)

        half_comb_list = []
        # reduce a half
        for comb in comb_list:
            if len(half_comb_list) > 30:
                break

            if comb not in half_comb_list and (comb[1], comb[0]) not in half_comb_list:
                half_comb_list.append(comb)
                pos_seq_pair_list.append([seq_list[comb[0]], seq_list[comb[1]]])

        pbar.update(1)

    print(len(pos_seq_pair_list))

    # set self transaction to "[pad]"
    pos_seq_pair_list_new = []

    for pos_seq_pair in pos_seq_pair_list:
        pos_seq_pair[0].tokens[0] = "[pad]"
        pos_seq_pair[1].tokens[0] = "[pad]"
        pos_seq_pair_list_new.append([pos_seq_pair[0], pos_seq_pair[1]])

    return pos_seq_pair_list_new


def counted_jaccard_distance(seq_pair_list):

    counted_jaccard_score_list = []
    for seq_pair in seq_pair_list:
        src_seq = seq_pair[0]
        dst_seq = seq_pair[1]

        src_tokens = src_seq.tokens[1:]
        src_cnts = src_seq.cnts[1:]

        dst_tokens = dst_seq.tokens[1:]
        dst_cnts = dst_seq.cnts[1:]

        inter_set = set(src_tokens).intersection(set(dst_tokens))
        if len(inter_set) == 0:
            score = 0
        else:
            src_inter_cnt = 0
            dst_inter_cnt = 0
            for idx in range(len(src_tokens)):
                if src_tokens[idx] in inter_set:
                    src_inter_cnt += src_cnts[idx]

            for idx in range(len(dst_tokens)):
                if dst_tokens[idx] in inter_set:
                    dst_inter_cnt += dst_cnts[idx]

            score = (src_inter_cnt + dst_inter_cnt) / (np.sum(src_cnts) + np.sum(dst_cnts))

        counted_jaccard_score_list.append(score)

    return counted_jaccard_score_list

def normal_jaccard_distance(seq_pair_list):

    jaccard_score_list = []
    fenzi_list = []
    fenmu_list = []

    for seq_pair in seq_pair_list:
        src_seq = seq_pair[0]
        dst_seq = seq_pair[1]

        src_tokens = src_seq.tokens[1:]
        dst_tokens = dst_seq.tokens[1:]

        inter_set = set(src_tokens).intersection(set(dst_tokens))
        union_set = set(src_tokens).union(set(dst_tokens))

        fenzi = len(inter_set)
        fenmu = len(union_set)

        score = 1.0 * fenzi / fenmu
        jaccard_score_list.append(score)
        fenzi_list.append(fenzi)
        fenmu_list.append(fenmu)

    return jaccard_score_list, fenzi_list, fenmu_list


def main():

    # vocab must load from gen_pretrain_data
    print("============Load Vocab==============")
    vocab_file_name = "./data/" + FLAGS.vocab_filename
    print("vocab pickle file: " + vocab_file_name)
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    print("===================================================")
    print("The fake generation strategy is DIFF.")
    seqs = difftime_sequence_split()
    seq_instance_list = gen_embedding_samples(seqs)
    pos_seq_pair_list = sequence_combination(seq_instance_list)

    jaccard_list, fenzi_list, fenmu_list = normal_jaccard_distance(pos_seq_pair_list)
    print("jaccrd:", np.mean(jaccard_list))
    print("fenzi:", np.mean(fenzi_list))
    print("fenmu:", np.mean(fenmu_list))
    print("pair_num:", len(jaccard_list))
    print("=====================Done==========================")

    # must run shuffle, otherwise positive pair together
    rng.shuffle(pos_seq_pair_list)

    # 写入新的Training instance中
    output_filename = FLAGS.data_dir + FLAGS.dataset_name + ".pair"
    tf.logging.info("*** Writing to output embedding files ***")
    tf.logging.info("  %s", output_filename)
    print("the number of pairs:" + str(len(pos_seq_pair_list)))
    write_instance_to_example_files(pos_seq_pair_list, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, vocab,
                                    [output_filename])

if __name__ == '__main__':
    main()


