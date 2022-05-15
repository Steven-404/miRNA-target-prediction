import numpy as np
import pandas as pd

import torch
import gensim
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader
bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'


import argparse
import regex
import Bio
from Bio import SeqIO
from Bio.Seq import Seq


def read_fasta(mirna_fasta_file, mrna_fasta_file):
    # 读取原始fasata序列数据
    # 输入miRNA-fasta文件和mRNA-fasta文件，两个地址
    # 输出     mirna_ids = [hsa-miR-4777-5p]
    #         mirna_seqs = [UUCUAGAUGAGAGAUAUAUAUA]
    #         mrna_ids = [NM_003629]
    #         mrna_seqs = [AGAGGAAGUGGGAAGAGAGGUGGUUCUCUGGCAUUUUUUU] 四个列表
    mirna_list = list(SeqIO.parse(mirna_fasta_file, 'fasta'))
    mrna_list = list(SeqIO.parse(mrna_fasta_file, 'fasta'))

    mirna_ids = []
    mirna_seqs = []
    mrna_ids = []
    mrna_seqs = []

    for i in range(len(mirna_list)):
        mirna_ids.append(str(mirna_list[i].id))
        mirna_seqs.append(str(mirna_list[i].seq))

    for i in range(len(mrna_list)):
        mrna_ids.append(str(mrna_list[i].id))
        mrna_seqs.append(str(mrna_list[i].seq))

    return mirna_ids, mirna_seqs, mrna_ids, mrna_seqs


def nucleotide_to_int(nucleotides, max_len):
    # 将每一行的核苷酸字母转换为数字
    # 输入：序列，最大序列长度
    # 输出：编码为int的序列
    dictionary = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}

    chars = []
    nucleotides = nucleotides.upper()
    for c in nucleotides:
        chars.append(c)

    ints_enc = np.full((max_len,), fill_value=0)  # to post-pad inputs
    for i in range(len(chars)):
        try:
            ints_enc[i] = dictionary[chars[i]]
        except KeyError:
            continue
        except IndexError:
            break

    return ints_enc


def sequence_to_int(sequences, max_len):
    # 把整个文件的序列转为数字
    # 输入：序列的向量 , 最大长度
    # 输出: 编码为int的序列的张量
    import itertools

    if type(sequences) is list:
        seqs_enc = np.asarray([nucleotide_to_int(seq, max_len) for seq in sequences])
    else:
        seqs_enc = np.asarray([nucleotide_to_int(seq, max_len) for seq in sequences])
        seqs_enc = list(itertools.chain(*seqs_enc))
        seqs_enc = np.asarray(seqs_enc)

    return seqs_enc


def pad_sequences(sequences, max_len=None, padding='pre', fill_value=0):
    # 在序列前加0填充
    # 输入：序列，最大长度，填充=前或后，填充内容=0
    # 输出：填充后的序列
    n_samples = len(sequences)

    lengths = []
    for seq in sequences:
        try:
            lengths.append(len(seq))
        except TypeError:
            raise ValueError("sequences expected a list of iterables, got {}".format(seq))
    if max_len is None:
        max_len = np.max(lengths)

    input_shape = np.asarray(sequences[0]).shape[1:]
    padded_shape = (n_samples, max_len) + input_shape
    padded = np.full(padded_shape, fill_value=fill_value)

    for i, seq in enumerate(sequences):
        if padding == 'pre':
            truncated = seq[-max_len:]
            padded[i, -len(truncated):] = truncated
        elif padding == 'post':
            truncated = seq[:max_len]
            padded[i, :len(truncated)] = truncated
        # else:
        #     raise ValueError("padding expected 'pre' or 'post', got {}".format(truncating))

    return padded


def one_hot_mi(ints):
    # one hot encoding for nucleotides
    # 输入：经过int处理的序列
    # 输出：one-hot矩阵

    dictionary_k = 5  # maximum number of nucleotides核苷酸的最大数目
    ints_len = len(ints)
    ints_enc = np.zeros((ints_len, dictionary_k))
    ints_enc[np.arange(ints_len), [k for k in ints]] = 1
    ints_enc = ints_enc[:, 1:5]  # to handle zero-padded values
    ints_enc = ints_enc.tolist()

    mi_vectors = KeyedVectors.load_word2vec_format(
        'Data/pretrained_vector/mi_vec/mi2vec_50.model')
    l=50
    for i in range(0,30):
        if ints_enc[i] == [1, 0, 0, 0] :
            ints_enc[i] = list(mi_vectors['A'])

        elif ints_enc[i] == [0, 1, 0, 0] :
            ints_enc[i] = list(mi_vectors['C'])

        elif ints_enc[i] == [0, 0, 1, 0] :
            ints_enc[i] = list(mi_vectors['G'])

        elif ints_enc[i] == [0, 0, 0, 1] :
            ints_enc[i] = list(mi_vectors['U'])

        else:
            ints_enc[i] = list([0.0]*l)
    return (ints_enc)

def one_hot_enc_mi(seqs_enc):
    # one hot encoding for sequences  对于整个文件的序列
    # 输入：多行序列
    # 输出：编码后的N维张量
    one_hot_encs = []

    for i in range(len(seqs_enc)):
        one_hot_encs.append(one_hot_mi(seqs_enc[i]))

    one_hot_encs = np.array(one_hot_encs)

    return one_hot_encs

def one_hot_m(ints):
    # one hot encoding for nucleotides
    # 输入：经过int处理的序列
    # 输出：one-hot矩阵

    dictionary_k = 5  # maximum number of nucleotides核苷酸的最大数目
    ints_len = len(ints)
    ints_enc = np.zeros((ints_len, dictionary_k))
    ints_enc[np.arange(ints_len), [k for k in ints]] = 1
    ints_enc = ints_enc[:, 1:5]  # to handle zero-padded values
    ints_enc = ints_enc.tolist()

    m_vectors = KeyedVectors.load_word2vec_format(
        'Data/pretrained_vector/m_vec/m2vec_50_n.model')
    l = 50
    for i in range(0, 30):
        if ints_enc[i] == [1, 0, 0, 0]:
            ints_enc[i] = list(m_vectors['A'])

        elif ints_enc[i] == [0, 1, 0, 0]:
            ints_enc[i] = list(m_vectors['C'])

        elif ints_enc[i] == [0, 0, 1, 0]:
            ints_enc[i] = list(m_vectors['G'])

        elif ints_enc[i] == [0, 0, 0, 1]:
            ints_enc[i] = list(m_vectors['U'])

        else:
            ints_enc[i] = list([0.0]*l)
    return (ints_enc)


def one_hot_enc_m(seqs_enc):
    #m_vectors = KeyedVectors.load_word2vec_format('/Users/sunyuzhuo/PycharmProjects/seq/data/pre_train_data/mi_vec/m2vec_50.model')

    # one hot encoding for sequences  对于整个文件的序列
    # 输入：多行序列
    # 输出：编码后的N维张量
    one_hot_encs = []

    for i in range(len(seqs_enc)):
        one_hot_encs.append(one_hot_m(seqs_enc[i]))

    one_hot_encs = np.array(one_hot_encs)

    return one_hot_encs

def to_categorical(labels, n_classes=None):
    # matrix assigning labels to a number of classes 将标签分配给矩阵，将矩阵分类
    # 输入：标签向量
    # 输出：分类结果
    labels = np.array(labels, dtype='int').reshape(-1)

    n_samples = labels.shape[0]
    if not n_classes:
        n_classes = np.max(labels) + 1

    categorical = np.zeros((n_samples, n_classes))
    categorical[np.arange(n_samples), labels] = 1

    return categorical


def preprocess_data(x_query_seqs, x_target_seqs, y=None, cts_size=None, pre_padding=False):
    # getting encoded data form miRNA and mRNA sequences 预处理数据
    # 输入：mirna-seqs,mrna-seqs
    # 输出：one-hot后的miRNA序列和mRNA序列
    if cts_size is not None:
        max_len = cts_size
    else:
        max_len = max(len(max(x_query_seqs, key=len)), len(max(x_target_seqs, key=len)))

    x_mirna = sequence_to_int(x_query_seqs, max_len)
    x_mrna = sequence_to_int(x_target_seqs, max_len)

    if pre_padding:
        x_mirna = pad_sequences(x_mirna, max_len, padding='pre')
        x_mrna = pad_sequences(x_mrna, max_len, padding='pre')

    x_mirna_embd = one_hot_enc_mi(x_mirna)
    x_mrna_embd = one_hot_enc_m(x_mrna)
    if y is not None:
        y_embd = to_categorical(y, np.unique(y).size)

        return x_mirna_embd, x_mrna_embd, y_embd
    else:
        return x_mirna_embd, x_mrna_embd


def read_ground_truth(ground_truth_file, header=True, train=True):
    # read the trainign and test files containing pairs of miRNA-mRNA
    # input format: [miRNA_ID, mRNA_ID, LABEL]
    # 输入：文件地址，参数：header，train
    # 输出：mir-id ，编码基因id ，基因名
    if header is True:
        records = pd.read_csv(ground_truth_file, header=0, sep='\t')
    else:
        records = pd.read_csv(ground_truth_file, header=None, sep='\t')

    query_ids = np.asarray(records.iloc[:, 0].values)
    #train-data第一列
    target_ids = np.asarray(records.iloc[:, 1].values)
    #train-data第二列
    if train is True:
        labels = np.asarray(records.iloc[:, 2].values)
    else:
        labels = np.full((len(records),), fill_value=-1)

    return query_ids, target_ids, labels




def find_candidate(mirna_sequence, mrna_sequence, seed_match):
    positions = set()

    if seed_match == '10-mer-m6':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 6
        TOLERANCE = (SEED_END - SEED_START + 1) - MIN_MATCH
    elif seed_match == '10-mer-m7':
        SEED_START = 1
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END - SEED_START + 1) - MIN_MATCH
    elif seed_match == 'offset-9-mer-m7':
        SEED_START = 2
        SEED_END = 10
        SEED_OFFSET = SEED_START - 1
        MIN_MATCH = 7
        TOLERANCE = (SEED_END - SEED_START + 1) - MIN_MATCH
    elif seed_match == 'strict':
        positions = find_strict_candidate(mirna_sequence, mrna_sequence)

        return positions
    else:
        raise ValueError(
            "seed_match expected 'strict', '10-mer-m6', '10-mer-m7', or 'offset-9-mer-m7', got '{}'".format(seed_match))

    seed = mirna_sequence[(SEED_START - 1):SEED_END]
    rc_seed = str(Seq(seed).complement())
    match_iter = regex.finditer("({}){{e<={}}}".format(rc_seed, TOLERANCE), mrna_sequence)

    for match_index in match_iter:
        # positions.add(match_index.start()) # slice-start indicies
        positions.add(match_index.end() + SEED_OFFSET)  # slice-stop indicies

    positions = list(positions)

    return positions


def find_strict_candidate(mirna_sequence, mrna_sequence):
    # find position of matches without tolerance

    positions = set()

    SEED_TYPES = ['8-mer', '7-mer-m8', '7-mer-A1', '6-mer', '6-mer-A1', 'offset-7-mer', 'offset-6-mer']
    for seed_match in SEED_TYPES:
        if seed_match == '8-mer':
            SEED_START = 2
            SEED_END = 8
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '7-mer-m8':
            SEED_START = 1
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '7-mer-A1':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '6-mer':
            SEED_START = 2
            SEED_END = 7
            SEED_OFFSET = 1
            seed = mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == '6mer-A1':
            SEED_START = 2
            SEED_END = 6
            SEED_OFFSET = 0
            seed = 'U' + mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == 'offset-7-mer':
            SEED_START = 3
            SEED_END = 9
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START - 1):SEED_END]
        elif seed_match == 'offset-6-mer':
            SEED_START = 3
            SEED_END = 8
            SEED_OFFSET = 0
            seed = mirna_sequence[(SEED_START - 1):SEED_END]

        rc_seed = str(Seq(seed).complement())
        match_iter = regex.finditer(rc_seed, mrna_sequence)

        for match_index in match_iter:
            # positions.add(match_index.start()) # slice-start indicies
            positions.add(match_index.end() + SEED_OFFSET)  # slice-stop indicies

    positions = list(positions)

    return positions


def get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match):
    # using the find_candidate function we can find actual candidates and positions
    positions = find_candidate(mirna_sequence, mrna_sequence, seed_match)

    candidates = []
    for i in positions:
        site_sequence = mrna_sequence[max(0, i - cts_size):i]
        rev_site_sequence = site_sequence[::-1]
        rc_site_sequence = str(Seq(rev_site_sequence).complement())
        candidates.append(rev_site_sequence)  # miRNAs: 5'-ends to 3'-ends,  mRNAs: 3'-ends to 5'-ends
        # candidates.append(rc_site_sequence)

    return candidates, positions


def make_pair(mirna_sequence, mrna_sequence, cts_size, seed_match):
    # and finally identify mirna_querys and mrna_targets
    candidates, positions = get_candidate(mirna_sequence, mrna_sequence, cts_size, seed_match)

    mirna_querys = []
    mrna_targets = []
    if len(candidates) == 0:
        return (mirna_querys, mrna_targets, positions)
    else:
        mirna_sequence = mirna_sequence[0:cts_size]
        for i in range(len(candidates)):
            mirna_querys.append(mirna_sequence)
            mrna_targets.append(candidates[i])

    return mirna_querys, mrna_targets, positions


def read_ground_truth(ground_truth_file, header=True, train=True):
    # read the trainign and test files containing pairs of miRNA-mRNA
    # input format: [miRNA_ID, mRNA_ID, LABEL]
    if header is True:
        records = pd.read_csv(ground_truth_file, header=0, sep='\t')
    else:
        records = pd.read_csv(ground_truth_file, header=None, sep='\t')

    query_ids = np.asarray(records.iloc[:, 0].values)
    target_ids = np.asarray(records.iloc[:, 1].values)
    if train is True:
        labels = np.asarray(records.iloc[:, 2].values)
    else:
        labels = np.full((len(records),), fill_value=-1)

    return query_ids, target_ids, labels


def make_input_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file, cts_size=30, seed_match='offset-9-mer-m7',
                    header=True, train=True):
    # from sequences, ids and ground truth we generate the dataset
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = read_fasta(mirna_fasta_file, mrna_fasta_file)
    query_ids, target_ids, labels = read_ground_truth(ground_truth_file, header=header, train=train)

    dataset = {
        'mirna_fasta_file': mirna_fasta_file,
        'mrna_fasta_file': mrna_fasta_file,
        'ground_truth_file': ground_truth_file,
        'query_ids': [],
        'query_seqs': [],
        'target_ids': [],
        'target_seqs': [],
        'target_locs': [],
        'labels': []
    }

    for i in range(len(query_ids)):
        try:
            j = mirna_ids.index(query_ids[i])
        except ValueError:
            continue
        try:
            k = mrna_ids.index(target_ids[i])
        except ValueError:
            continue

        query_seqs, target_seqs, locations = make_pair(mirna_seqs[j], mrna_seqs[k], cts_size=cts_size,
                                                       seed_match=seed_match)

        n_pairs = len(locations)
        if n_pairs > 0:
            queries = [query_ids[i] for n in range(n_pairs)]
            dataset['query_ids'].extend(queries)
            dataset['query_seqs'].extend(query_seqs)

            targets = [target_ids[i] for n in range(n_pairs)]
            dataset['target_ids'].extend(targets)
            dataset['target_seqs'].extend(target_seqs)
            dataset['target_locs'].extend(locations)

            dataset['labels'].extend([[labels[i]] for p in range(n_pairs)])

    return dataset


def get_negative_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file=None, cts_size=30,
                      seed_match='offset-9-mer-m7', header=False, predict_mode=True):
    mirna_ids, mirna_seqs, mrna_ids, mrna_seqs = read_fasta(mirna_fasta_file, mrna_fasta_file)

    dataset = {
        'query_ids': [],
        'target_ids': [],
        'predicts': []
    }

    if ground_truth_file is not None:
        query_ids, target_ids, labels = read_ground_truth(ground_truth_file, header=header)

        for i in range(len(query_ids)):
            try:
                j = mirna_ids.index(query_ids[i])
            except ValueError:
                continue
            try:
                k = mrna_ids.index(target_ids[i])
            except ValueError:
                continue

            query_seqs, target_seqs, locations = make_pair(mirna_seqs[j], mrna_seqs[k], cts_size=cts_size,
                                                           seed_match=seed_match)

            n_pairs = len(locations)
            if (n_pairs == 0) and (predict_mode is True):
                dataset['query_ids'].append(query_ids[i])
                dataset['target_ids'].append(target_ids[i])
                dataset['predicts'].append(0)
            elif (n_pairs == 0) and (predict_mode is False):
                dataset['query_ids'].append(query_ids[i])
                dataset['target_ids'].append(target_ids[i])
                dataset['predicts'].append(labels[i])
    else:
        for i in range(len(mirna_ids)):
            for j in range(len(mrna_ids)):
                query_seqs, target_seqs, locations = make_pair(mirna_seqs[i], mrna_seqs[j], cts_size=cts_size,
                                                               seed_match=seed_match)

                n_pairs = len(locations)
                if n_pairs == 0:
                    dataset['query_ids'].append(mirna_ids[i])
                    dataset['target_ids'].append(mrna_ids[j])
                    dataset['predicts'].append(0)

    dataset['target_locs'] = [-1 for i in range(len(dataset['query_ids']))]
    dataset['probabilities'] = [0.0 for i in range(len(dataset['query_ids']))]

    return dataset


def postprocess_result(dataset, probabilities, predicts, predict_mode=True, output_file=None, cts_size=30,
                       seed_match='offset-9-mer-m7', level='site'):
    neg_pairs = get_negative_pair(dataset['mirna_fasta_file'], dataset['mrna_fasta_file'], dataset['ground_truth_file'],
                                  cts_size=cts_size, seed_match=seed_match, predict_mode=predict_mode)

    query_ids = np.append(dataset['query_ids'], neg_pairs['query_ids'])
    target_ids = np.append(dataset['target_ids'], neg_pairs['target_ids'])
    target_locs = np.append(dataset['target_locs'], neg_pairs['target_locs'])
    probabilities = np.append(probabilities, neg_pairs['probabilities'])
    predicts = np.append(predicts, neg_pairs['predicts'])

    # output format: [QUERY, TARGET, LOCATION, PROBABILITY]
    records = pd.DataFrame(columns=['MIRNA_ID', 'MRNA_ID', 'LOCATION', 'PROBABILITY'])
    records['MIRNA_ID'] = query_ids
    records['MRNA_ID'] = target_ids
    records['LOCATION'] = np.array(
        ["{},{}".format(max(1, l - cts_size + 1), l) if l != -1 else "-1,-1" for l in target_locs])
    records['PROBABILITY'] = probabilities
    if predict_mode is True:
        records['PREDICT'] = predicts
    else:
        records['LABEL'] = predicts

    # records = records.sort_values(by=['PROBABILITY', 'MIRNA_ID', 'MRNA_ID'], ascending=[False, True, True])
    unique_records = records.drop_duplicates(subset=['MIRNA_ID', 'MRNA_ID'], keep='first')

    if level == 'site':
        if output_file is not None:
            records.to_csv(output_file, index=False, sep='\t')

        return records
    elif level == 'gene':
        if output_file is not None:
            unique_records.to_csv(output_file, index=False, sep='\t')

        return unique_records
    #else:
        #raise ValueError("level expected 'site' or 'gene', got '{}'".format(mode))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mirna_fasta_file, mrna_fasta_file, ground_truth_file, cts_size=30, seed_match='offset-9-mer-m7',
                 header=True, train=True):
        self.dataset = make_input_pair(mirna_fasta_file, mrna_fasta_file, ground_truth_file, cts_size=cts_size,
                                       seed_match=seed_match, header=header, train=train)
        self.mirna, self.mrna = preprocess_data(self.dataset['query_seqs'], self.dataset['target_seqs'])
        self.labels = np.asarray(self.dataset['labels']).reshape(-1, )

        # self.mirna = self.mirna.transpose((0, 2, 1))
        # self.mrna = self.mrna.transpose((0, 2, 1))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        mirna = self.mirna[index]
        mrna = self.mrna[index]
        label = self.labels[index]

        return (mirna, mrna), label

    def __len__(self):
        return len(self.labels)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, ground_truth_file, cts_size=30):
        self.records = pd.read_csv(ground_truth_file, header=0, sep='\t')
        #test new data
        #self.records = pd.read_csv(ground_truth_file, header=0)
        mirna_seqs = self.records['MIRNA_SEQ'].values.tolist()
        mrna_seqs = self.records['MRNA_SEQ'].values.tolist()

        self.mirna, self.mrna = preprocess_data(mirna_seqs, mrna_seqs, cts_size=cts_size)
        self.labels = self.records['LABEL'].values.astype(int)

        # self.mirna = self.mirna.transpose((0, 2, 1))
        # self.mrna = self.mrna.transpose((0, 2, 1))

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        mirna = self.mirna[index]
        mrna = self.mrna[index]
        label = self.labels[index]

        return (mirna, mrna), label

    def __len__(self):
        return len(self.labels)

