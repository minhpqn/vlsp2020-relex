# coding=utf-8
"""
Utility functions for data loading
"""
import numpy as np
import unittest


class RelexSample:
    
    def __init__(self, sentence, e1_start, e1_end, e2_start, e2_end):
        self.sentence = sentence
        self.e1_start = e1_start
        self.e1_end = e1_end
        self.e2_start = e2_start
        self.e2_end = e2_end


def create_sequence_with_markers(sample, e1_start_token='[E1]', e1_end_token='[/E1]',
                                 e2_start_token='[E2]', e2_end_token='[/E2]'):
    """Create sample with entity markers
    """
    tokens = sample.sentence.split(' ')
    e1_start, e1_end = sample.e1_start, sample.e1_end
    e2_start, e2_end = sample.e2_start, sample.e2_end
    
    res = []
    positions = [e1_start, e1_end+1, e2_start, e2_end+1]
    symbols = [e1_start_token, e1_end_token, e2_start_token, e2_end_token]
    
    if e2_start == e1_end+1:
        indexes = [0, 1, 2, 3]
    elif e1_start == e2_end + 1:
        indexes = [2, 3, 0, 1]
    else:
         indexes = np.argsort(positions)
    
    for i in range(len(tokens)):
        for j in range(len(indexes)):
            if i == positions[indexes[j]]:
                res.append(symbols[indexes[j]])
        res.append(tokens[i])
    
    if e1_end+1 == len(tokens):
        res.append(e1_end_token)
    if e2_end+1 == len(tokens):
        res.append(e2_end_token)
        
    return ' '.join(res)


def load_relex_samples(file_path):
    """Loading list of RelexSamples from data in SemEval 2010 format
    """
    samples = []
    labels = []
    with open(file_path, "r") as fi:
        for line in fi:
            line = line.strip()
            if line == "":
                continue
            fields = line.split("\t")
            lb = int(fields[0])
            sample = RelexSample(fields[7], int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]))
            samples.append(sample)
            labels.append(lb)
    return samples, labels


def load_id2label(file_path):
    """Load id2label from file id2label.txt
    """
    id2label = {}
    with open(file_path, 'r') as fi:
        for line in fi:
            line = line.strip()
            if line == "":
                continue
            i, lb = line.split("\t")
            id2label[int(i)] = lb
    return id2label


class TestDataUtils(unittest.TestCase):
    
    def test_create_sequence_with_marker(self):
        sample = RelexSample("Tàu sân bay Mỹ tập trận với Nhật gần bán đảo Triều Tiên Tàu Ronald Reagan đang tập trận với các tàu chiến Nhật phía nam bán đảo Triều Tiên , hành động phô trương sức mạnh khi Bình Nhưỡng doạ thử hạt nhân .",
                             9, 12, 3, 3)
        print(create_sequence_with_markers(sample))
        self.assertEqual("Tàu sân bay [E2] Mỹ [/E2] tập trận với Nhật gần [E1] bán đảo Triều Tiên [/E1] Tàu Ronald Reagan đang tập trận với các tàu chiến Nhật phía nam bán đảo Triều Tiên , hành động phô trương sức mạnh khi Bình Nhưỡng doạ thử hạt nhân .",
                         create_sequence_with_markers(sample))

        sample = RelexSample("Tàu sân bay Mỹ tập trận với Nhật gần bán đảo Triều Tiên Tàu Ronald Reagan đang tập trận với các tàu chiến Nhật phía nam bán đảo Triều Tiên , hành động phô trương sức mạnh khi Bình Nhưỡng doạ thử hạt nhân .",
                              9, 12, 9, 10)
        print(create_sequence_with_markers(sample))
        
        sample = RelexSample("Hạnh Chi ( T / h ) Theo Đời sống Plus / GĐVN", 0, 1, 12, 12)
        print(create_sequence_with_markers(sample))

        sample = RelexSample("Hạnh Chi ( T / h ) Theo Đời sống Plus / GĐVN", 11, 11, 12, 12)
        print(create_sequence_with_markers(sample))

        sample = RelexSample("Hạnh Chi ( T / h ) Theo Đời sống Plus / GĐVN", 12, 12, 11, 11)
        print(create_sequence_with_markers(sample))
        

