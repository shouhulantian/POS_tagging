import glob
import re
from torch.utils.data import Dataset, DataLoader
import torch
import os
import re
from collections import Counter

#debug = True
TAG_PATH = 'penn-tree-bank/treebank/tagged'
RAW_PATH = 'penn-tree-bank/treebank/raw'
POS_FILE = 'pos_tag.txt'
WORD_FILE = 'word_en.txt'
TAG_FILE = 'tag_en.txt'

'''file all files in the directory'''
def Findfiles(path):
    return glob.glob(path)

'''process all sentences in the corpus'''
def makeSent(filename):
    sentences = []
    f = open(filename,'r',encoding='UTF-8')
    sent = f.readlines()
    for i in range(len(sent)):
        if (i !=0 and sent[i] != '\n'):
            sentences.append(sent[i])
    return sentences

# if debug:
#     debugA = makeSent('penn-tree-bank/treebank/raw/wsj_0003')
#     print(debugA)


'''go through all the words in the sentences'''
def CountWords(path):
    filelist = Findfiles(path + ('/wsj_*'))
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    sentences = []
    word_set = []
    for file in filelist:
        sentences = sentences + makeSent(file)
        print(file)

    for sent in sentences:
        word = re.split(pattern, sent)
        word_set = word_set+(set(sent))


'''process the tagged directory in the corpus, get the sentences and every tag for each word'''
def makeTags(filename):
    tags = []
    f = open(filename)
    tag = f.readlines()

    for i in range( len(tag)):
        # if (tag[i] == '\n' or i == len(tag)-1):
            # if (len(tag_persen) > 0): tags.append(tag_persen)
            # tag_persen = []
        if(tag[i]!= '\n' ):
            con = tag[i].split()
            if ('[' in con): con.remove('[')
            if (']' in con): con.remove(']')
            if '======================================' not in con: tags.extend(con)
    return tags

# if debug:
#     debugB = makeTags('penn-tree-bank/treebank/tagged/wsj_0001.pos')
#     print(debugB)

'''process the format difference between two files'''
def TagProcess(word):
    word = word.replace('``', '"')
    word = word.replace("''",'"')
    word = word.replace('\/', ' ')
    return word

def RawProcess(word):

    return len(word)

'''process all files in the tagged directory, and write pos result to POS_FILE'''
def ReadAllFiles():
    f = open(POS_FILE,'w')

    tag_file = Findfiles(TAG_PATH+('/wsj_*.pos'))
    tag_list = []

    raw_file = Findfiles(RAW_PATH+("/wsj*"))
    sent_list = []

    for i in range(len(tag_file)):
        tag_per_file = makeTags(tag_file[i])
        tag_len = []
        word_count = 0
        for con in tag_per_file:
            #print(con)
            con = TagProcess(con)
            if len(con.split('/')) == 2:
                word, tag = con.split('/')

            word_count = word_count + len(word)
            tag_len.append(word_count)
        # tag_list = tag_list +tag_per_file

        raw_per_file = makeSent(raw_file[i])
        # sent_list = sent_list + raw_per_file
        start_mark = 0
        token_num = 0
        for line in raw_per_file:

            line = line.strip(' \n')
            line = line.split()
            #print(i ,'  ', line)

            for a in line:
                token_num = token_num + RawProcess(a)
            end_mark = tag_len.index(token_num)

            for s in tag_per_file[start_mark:end_mark+1]:
                f.write(str(s+' '))
            f.write('\n')
            start_mark = end_mark + 1

    f.close()

# if debug:
#     ReadAllFiles()

'''count the frequency of all words'''
def count_fre(words):
    fre = Counter(words).most_common()

    return fre


'''count tags and words appeared in all sentences'''
def word_tag_count():
    if not os.path.exists(POS_FILE):
        ReadAllFiles()
    f = open(POS_FILE)
    sentences = f.readlines()
    f.close()

    words = []
    tags = []

    for sent in sentences:
        for s in sent.split():
            s = TagProcess(s)
            a, b = s.split('/')
            words.append(a)
            tags.append(b)

    tags_fre = count_fre(tags)
    words_fre = count_fre(words)

    f = open(TAG_FILE, 'w')
    for a, b in tags_fre:
        f.write(a+' '+str(b)+'\n')
    f.close()

    f = open(WORD_FILE,'w')
    for a, b in words_fre:
        f.write(a+' '+str(b) +'\n')
    f.close()
    return











