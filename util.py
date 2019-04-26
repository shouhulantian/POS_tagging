import os

'''count vocab/tag size'''
def CountSize(file, min = 3):
    f = open(file)
    lines = f.readlines()

    count = 0
    for s in lines:
        c = s.split()
        if int(c[-1]) >= min: count = count+1
    return count

#len = CountSize('tag_en.txt')

'''make word_id'''
class make_dict(object):
    def __init__(self,file, min =3):
        f = open(file)
        lines = f.readlines()
        self.id={'Other': 0}
        for s in lines:
            s = s.split()
            w = ''
            for l in s[:-1]:
                w = w+l
            c = s[-1]

            if int(c)>=3: self.id[w] = len(self.id)

        self.size = len(self.id)

# a = Word_dict('word_en.txt')
# print(a)


