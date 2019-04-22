from torch.utils.data import Dataset, DataLoader
import torch
from dataset import PosDataset
import torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim
from model import LSTMTagger
from config import LSTMConfig
from util import *
import time

configs = LSTMConfig()

word_dict = make_dict(configs.WORD_FILE)
tag_dict = make_dict(configs.TAG_FILE)
id2word_dict = {v : k for k, v in word_dict.id.items()}
id2tag_dict = {v : k for k, v in tag_dict.id.items()}

model = LSTMTagger(configs.word_dim, configs.hidden_dim, word_dict.size, tag_dict.size)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr= configs.lr, momentum=configs.lr_decay)
loss_function = nn.NLLLoss()
data = PosDataset(configs.POS_FILE, word_dict, tag_dict)

def makeDataset(test_rate = 0.2, validation_rate = 0):
    test_id = int(data.__len__() * test_rate)
    test_data = DataLoader([data[i] for i in range(test_id)], batch_size=1, num_workers=configs.num_workers)
    validation_id = int(data.__len__() * (test_rate + validation_rate))
    validation_data = DataLoader([data[i] for i in range(test_id, validation_id)], batch_size=1,
                                 num_workers=configs.num_workers)
    train_data = DataLoader([data[i] for i in range(validation_id, data.__len__())], batch_size=1, shuffle=True,
                                 num_workers=configs.num_workers)
    return train_data, test_data, validation_data


def categoryFromOutput(pre, lab):
    top_n, top_i = pre.topk(1, dim =1)
    top_i = top_i.squeeze(dim =1)
    pre_seq = top_i.numpy()
    lab_seq = lab.squeeze(dim =0).numpy()
    equallist = [int(pre_seq[i]==lab_seq[i]) for i in range(len(lab_seq)) ]

    predict = [id2tag_dict[c] for c in pre_seq]
    label = [id2tag_dict[c] for c in lab_seq]
    return predict, label, equallist.count(1)

for epoch in range(configs.num_epochs): # again, normally you would NOT do 300 epochs, it is toy data

    start = time.time()
    model.train()

    train_data, test_data, _ = makeDataset()
    checkpoint = 0
    for sentence, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.

        # Step 3. Run our forward pass.
        sentence = sentence.to(device)
        tags = tags.to(device)
        tag_scores = model(sentence)
        tag_scores = tag_scores.view(-1, tag_dict.size)
        tags = tags.squeeze(dim = 0)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()

        loss = loss_function(tag_scores, tags)
        loss.backward()
        optimizer.step()

        c = time.time() - start
        checkpoint = checkpoint + 1
        if checkpoint % 100 ==0:
            print('epoch ', epoch, ' has started ', c ,'s,checkpoint ' ,checkpoint, ' loss is ',loss.item())
            model.eval()
            total_n = 0
            total_correct = 0
            for test_sent, test_tag in test_data:
                test_sent = test_sent.to(device)
                test_tag = test_tag.to(device)
                test_scores = model(test_sent)
                predict, label, correct = categoryFromOutput(test_scores, test_tag)
                total_correct = total_correct + correct
                total_n = total_n + test_sent.size()[1]
            print(total_n, total_correct)
    torch.save(model.state_dict(), '{}_{}.tar'.format(epoch,'checkpoint'))





