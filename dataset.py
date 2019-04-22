from torch.utils.data import Dataset, DataLoader
import torch
import os
from raw_tag_alignment import *

def prepare_sequence(seq, to_ix):
    idxs = []
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    #idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

'''Make Dataset for the task'''
class PosDataset(Dataset):

    def __init__(self,POS_FILE,word_dict, tag_dict):
        if  not os.path.exists(POS_FILE):
            ReadAllFiles()
        f = open(POS_FILE)
        self.sent = f.readlines()
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        f.close()

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, id):
        sentences = self.sent[id].split()
        word = []
        tag = []
        for sent in sentences:
            sent = TagProcess(sent)
            a, b = sent.split('/')
            word.append(a)
            tag.append(b)
        word = prepare_sequence(word,self.word_dict.id)
        tag = prepare_sequence(tag, self.tag_dict.id)

        return  word, tag

 # pos_dataset = PosDataset(POS_FILE)
 # print(pos_dataset[0])