# coding=utf-8
"""
Utility functions for data loading
"""


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
    tokens.insert(e1_start, e1_start_token)
    tokens.insert(e1_end + 2, e1_end_token)
    tokens.insert(e2_start + 2, e2_start_token)
    tokens.insert(e2_end + 4, e2_end_token)
    
    return ' '.join(tokens)


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