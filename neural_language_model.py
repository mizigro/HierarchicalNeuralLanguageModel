import pickle 
import sys, os
import keras
import re
import numpy as np
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model
import math
from node import Node

def get_sentences(path, embed, max_sentences = 10000):
    n_samples = 0

    with open(path, encoding='latin-1')as f:
        texts = f.read()
        i = texts.find('\n\n')  # skip header in file (starts with two newlines.)
        if 0 < i:
            texts = texts[i:]
        texts = texts

    # Cleaning data - remove punctuation from every newsgroup text
    sentences = [re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', 
                        repl='', 
                        string=x
                    ).strip().split(' ') for x in texts.split('\n') 
                    if not x.endswith('writes:')]
    sentences = [x for x in sentences if len(x)>=6][:max_sentences]

    for s in sentences:
        for i in range(len(s)-5):
            words = s[i:i+6]
            x = words[:5]
            y = words[5]

            if all([True if xi in embed else False for xi in x]) and y in embed:
                n_samples += 1

    return sentences, n_samples

def gen_xy(words, embed):

    x = [item for word in words[:5] for item in embed[word][0]]
    y = embed[words[5]][1]

    return x, y


def data_generator(sentences, embed, tree, batch_size = 64, return_words = False):
    X, Y, W = [], [], []
    while True:
        sentences = np.array(sentences)
        for s in sentences:
            for i in range(len(s)-5):
                words = s[i:i+6]
                x = words[:5]
                y = words[5]

                if all([True if xi in embed else False for xi in x]) and y in embed:
                    x, y = gen_xy(words, embed)
                    W.append(words)
                    X += [x]
                    Y += [y]
                    if len(X) >= batch_size:
                        if return_words:
                            yield np.array(X), np.array(Y), W
                        else:
                            yield np.array(X), np.array(Y)
                        X, Y, W = [], [], []
        np.random.shuffle(sentences)
        
        

def create_model(in_size, out_size):
    model = keras.models.Sequential([
        Dense(100, input_shape = (in_size,), activation = 'relu'),
        BatchNormalization(),
        Dense(100, input_shape = (in_size,), activation = 'relu'),
        BatchNormalization(),
        Dense(out_size, activation = 'sigmoid')
    ])
    return model


def train_model(sentences, embed, n_samples, model_path):
    batch_size = 256
    epochs = 5
    lr = 10

    train_gen = data_generator(sentences, embed, batch_size)
    test_gen = data_generator(sentences, embed, batch_size)

    sample = next(train_gen)
    print('In Shape:',sample[0].shape, 'Out Shape:', sample[1].shape)

    model = create_model(in_size = 250, out_size = sample[1].shape[1])
    
    sgd = SGD(lr = lr, decay = (lr/epochs)/2)
    model.compile(loss = 'mse',optimizer = sgd)
    model.summary()

    model.fit_generator(
        generator = train_gen,
        steps_per_epoch = n_samples//batch_size,
        epochs = epochs
    )

    model.save(model_path)

def observe_predictions(model, test_data):
    
    X, Y, W = next(test_data)

    Y_ = model.predict(X)

    for i in range(len(X)):
        print(W[i],',Label:',tree.get_word(Y[i]), ',Prediction:',tree.get_word(Y_[i]))

def test_model(sentences, embed, tree, n_samples, model_path):
    model = load_model(model_path)
    # model.summary()
    observe_predictions(
        model, 
        data_generator(sentences, embed, tree, batch_size = 128, return_words = True)
    )

    test_data = data_generator(sentences, embed, tree, batch_size = 1, return_words = True)

    sum_log_prob = 0
    n_words = 1000
    
    for i in range(n_words):
        if i%(n_words//100)==0:
            print(i*100//n_words,'%',end='\r')
        X,Y,W = next(test_data)
        Y_ = model.predict(X)

        prob = tree.prob_word(Y_[0],W[0][-1])
        sum_log_prob += math.log(prob, 2)
    
    print('Likelihood',sum_log_prob)
    print('Perplexity',2**(-sum_log_prob/n_words))


if __name__=='__main__':
    embed_path = './dumps/embed.dat'
    # data_path = './data/xad.dat'
    tree_path = './dumps/tree.dat'
    model_path = './models/language.model'

    with open(tree_path,'rb') as f:   
        print('Tree ...',end='\r')
        tree = pickle.load(f)
        print('Tree Loaded')
        print('Tree height:',tree.height())

    with open(embed_path,'rb') as f: 
        print('Loading Embedding ...',end='\r')  
        embed = pickle.load(f)
        print('Embedding Loaded. No. of Words:',len(embed), 'Memory:', sys.getsizeof(embed))


    data_paths = ['./data/xad.dat']
    for data_path in data_paths:
        print('Loading Sentence / Data Samples ... ',end='\r')
        sentences, n_samples = get_sentences(data_path, embed)
        print('Sentences prepared. No. of Sentences:',len(sentences), 'Samples Found:', n_samples)

        # gen = data_generator(sentences, embed, tree, batch_size = 128, return_words = True)
        # while True:
        #     X, Y, W = next(gen)
        #     for i in range(len(X)):
        #         print(W[i],',Label:',tree.get_word(Y[i]))
        #     input('Press any key ...')
        try:
            print('\nLoading Model ...','\r')
            test_model(sentences, embed, tree, n_samples, model_path)
        except OSError as e:
            print(e)
            print('\nTraining Model ...')
            train_model(sentences, embed, n_samples, model_path)
            print('\nModel Trained.')