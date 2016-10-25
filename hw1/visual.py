#! encoding=utf-8
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
 
 
def main():
 
    embeddings_file = sys.argv[1]
    wv, vocabulary = load_embeddings(embeddings_file)
 
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:,:])

    # plt.figure(figsize=(50, 50))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
 
    plt.scatter(Y[:, 0], Y[:, 1], s=0.1)
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=3)
        
    plt.plot([-19.1248726889,-19.1845086249], [-4.14398420682, -4.96695244044], '-', color='r')
    plt.plot([-18.5855625202,-18.5805200979], [-8.10239031388, -12.4214918389], '-', color='r')
    plt.plot([-6.08998956172,-6.14564501426], [1.00415005626, 0.335774691849], '-', color='r')
    plt.savefig('test.pdf', format='pdf')
    plt.show()
 #美女 -6.08998956172 1.00415005626
 #帥哥 -6.14564501426 0.335774691849
 #女生 -19.1248726889 -4.14398420682
 #男生 -19.1845086249 -4.96695244044
 #女 -18.5855625202 -8.10239031388
 #男 -18.5805200979 -12.4214918389
 #女朋友 -20.6634056503 -3.69698216295
 #男朋友 -20.6135839095 -3.65125905196
def load_embeddings(file_name):
 
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in 
f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary
 
if __name__ == '__main__':
    main()