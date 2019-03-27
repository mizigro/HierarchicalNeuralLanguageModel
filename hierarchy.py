
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import pickle
import random
from node import Node

def getKeysVectors(model):
    vectors = []
    keys = list(model.wv.vocab.keys())
    for key in keys:
        vectors.append(model.wv[key])
    vectors = np.array(vectors)

    return keys, vectors


def getHierarchy(vectors, keys, id, all_ids):
    print(id)
    if len(vectors) == 1:
        return [(vectors[0], keys[0], id)], Node(keys[0])
    
    all_ids.append(id)

    node = Node(len(all_ids)-1)
    kmeans = list(KMeans(n_clusters=2, random_state=0).fit(vectors).labels_)

    # splitting
    set_1 = [vectors[i] for i in range(len(kmeans)) if kmeans[i] == 0]
    keys_1 = [keys[i] for i in range(len(kmeans)) if kmeans[i] == 0]
    h1, node.left = getHierarchy(set_1, keys_1, 2*id, all_ids)
    del set_1, keys_1

    set_2 = [vectors[i] for i in range(len(kmeans)) if kmeans[i] == 1]
    keys_2 = [keys[i] for i in range(len(kmeans)) if kmeans[i] == 1]
    h2, node.right = getHierarchy(set_2, keys_2, 2*id+1, all_ids)
    del set_2, keys_2

    return h1+h2, node

if __name__=='__main__':
    # model_path = './dumps/embedding.model'
    model_path = './dumps/embedding_partial.model'
    data_path = './dumps/embed.dat'
    tree_path = './dumps/tree.dat'

    model = Word2Vec.load(model_path)

    print('Vocabulary Size:',len(model.wv.vocab))

    # print(model.vector_size)
    # print(model.wv.vocab.keys())
    # print(model.wv['early'])

    keys, vectors = getKeysVectors(model)

    all_ids = []
    hierarchy, tree = getHierarchy(vectors, keys, 1, all_ids)
    print('Hierarchy Created ( Size :',len(hierarchy),')')

    with open(tree_path, 'wb') as f:
        pickle.dump(tree, f)
        print('Tree Saved')

    final = {}
    print('Height of Tree:',tree.height())
    print('Creating Embedding ... ')
    counter = 0
    for vector, word, id in hierarchy:
        if len(hierarchy)>100:
            percentage = counter*100//len(hierarchy)
            if counter % (len(hierarchy)//100) == 0:
                print(percentage,'%', end='\r')
            counter += 1
            
        mask = tree.get_mask(word) 
        mask += [0]*(tree.height() - len(mask)-1)

        final[word] = np.array([vector, mask])

    print('\n\nSaving Dictionary.')
    with open(data_path, 'wb') as f:
        pickle.dump(final, f)
    
    print('Embedding Dictionary Saved.')
