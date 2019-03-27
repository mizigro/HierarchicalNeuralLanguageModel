import os
import sys
import re
from gensim.models import Word2Vec

def load_texts(path):
    # Newsgroups data is split between many files and folders.
    # Directory stucture 20_newsgroup/<newsgroup label>/<post ID>
    texts = []         # list of text samples
    # Go through each directory

    
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        f = open(fpath, encoding='latin-1')
        t = f.read()
        i = t.find('\n\n')  # skip header in file (starts with two newlines.)
        if 0 < i:
            t = t[i:]
        texts.append(t)
        f.close()


    # Cleaning data - remove punctuation from every newsgroup text
    sentences = []
    # Go through each text in turn
    for ii in range(len(texts)):
        sentences = [re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', 
                            repl='', 
                            string=x
                        ).strip().split(' ') for x in texts[ii].split('\n') 
                        if not x.endswith('writes:')]
        sentences = [x for x in sentences if x != ['']]
        texts[ii] = sentences

    return sentences 


if __name__=='__main__':
    dpath = './data'
    outpath = './dumps/embedding_partial.model'
    embedding_size = 50

    sentences = load_texts(dpath)[:10000]
    print(sentences[0])
    print('Found %s texts.' % len(sentences))

    model = Word2Vec(sentences, size = embedding_size)

    print(model.vector_size)
    print(len(model.wv.vocab))

    # print(model.most_similar('135g'))

    model.save(outpath)