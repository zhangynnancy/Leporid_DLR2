import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_one_graph(N, M, DIM, repeats):
    g = nx.barabasi_albert_graph(N, M)
    W = nx.to_numpy_array(g)
    degree = W.sum(axis = 1)
    data = {}
    frequency = []

    _, U_gold = embeddings(W, DIM)

    for i in range(N):
        if degree[i] == 100 or degree[i] == 0:
            continue

        # increment frequency
        frequency.append(degree[i])

        norms = np.zeros([repeats])
        for j in range(repeats):
            W2 = add_random_edge(W, N, i)
            lam, U = embeddings(W2, DIM)
            # print(j)
            norms[j] = np.linalg.norm(U[i, :] - U_gold[i, :])
            # print('trial {} norm {} \n'.format(j, norms[j]))

        #variance = np.var(norms) # mean?
        mean = np.mean(norms) # mean?
        if degree[i] in data:
            data[degree[i]].append(mean)
        else:
            data[degree[i]] = [mean]

    return data, frequency

def merge_dict(data, new_data):
    for key, item in new_data.items():
        if key in data:
            data[key] = data[key] + item
        else:
            data[key] = item
    return data


def main():
    M = 10
    N = 100
    DIM = 40
    REPEAT = 30
    NUM_GRAPH = 50

    data = {}
    frequency = []

    for i in tqdm(range(NUM_GRAPH)):
        new_data, new_frequency = test_one_graph(N, M, DIM, REPEAT)
        data = merge_dict(data, new_data)
        frequency = frequency + new_frequency

    x = []
    y = []
    comb = []

    for key, item in data.items():
        mean = np.mean(np.array(item))
        comb.append((key, mean))
        print('key {} var {}'.format(key, mean))

    comb.sort(key=lambda x: x[0])


    for a1, b1 in comb:
        if b1 > 1e-10:
            x.append(a1)
            y.append(b1)

    start = int(next(i for i in range(len(x)) if x[i] >= 10))
    end = int(next((i for i in range(len(x)) if x[i] > 60), len(x)))
    x = x[start:end]
    y = y[start:end]
    fig, ax = plt.subplots()
    # plt.subplot(2, 1, 1)
    plt.xlabel('Node degree', fontsize=18)
    plt.ylabel('Change in L2 norm from 1 extra edge', fontsize=18)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.yscale('log')
    plt.scatter(x, y, s=50.0, color='black')


    # plt.subplot(2, 1, 2)
    # b = int(max(frequency) - min(frequency) + 1)
    # print(b)
    # plt.hist(frequency, density=True, bins=b)
    
    plt.show()

    # print(lam)
    # print(U)

    # print(np.dot(U[:, 1], U[:, 4]))
    # print(np.dot(U[:, 3], U[:, 6]))
    

def embeddings(W, k):
    degree = W.sum(axis = 1)
    D = np.diag(degree)
    L = D - W
    lam, U = np.linalg.eigh(L)
    return lam, U[:, 0:k]

def add_random_edge(W, N, i):
    lst = [j for j in range(0, N) if W[i, j] == 0]
    # print(lst)
    c = np.random.choice(lst)
    W2 = np.copy(W)
    # print('adding edge between {}, {}, {}'.format(c, i, W[i, c]))
    W2[i, c] = 1
    W2[c, i] = 1
    return W2  

if __name__ == "__main__":
    main()