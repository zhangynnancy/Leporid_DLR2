import argparse
import pandas as pd
import numpy as np
import time
import os
from scipy import stats
from tqdm import tqdm
from scipy.sparse.linalg import eigs


def load_user_dict(file_name):
    df = pd.read_csv(file_name, sep=' ', names=["users", "items", "ratings", "time"])
    print('size = ', df.shape)
    test = {}
    user_dict = {}
    for name, g in df.groupby('users'):
        items = []
        test[name] = g.sort_values(by=["time"], ascending=[1])["items"].values
        for i in test[name]:
            if i not in items:
                items.append(i)
        user_dict[name] = np.array(items)
    return user_dict


def load_item_dict(file_name):
    df = pd.read_csv(file_name, sep=' ', names=["users", "items", "ratings", "time"])
    print('size = ', df.shape)
    test = {}
    item_dict = {}
    for name, g in (df.groupby('items')):
        items = []
        test[name] = g.sort_values(by=["time"], ascending=[1])["users"].values
        for i in test[name]:
            if i not in items:
                items.append(i)
        item_dict[name] = np.array(items)
    return item_dict


def jaccard_dis(x, y):
    inter = np.intersect1d(x, y)
    union = np.union1d(x, y)
    return len(inter) / len(union)


def cosine_dis(x, y, length):
    a = np.zeros((1, length))
    a[:, x] = 1.
    b = np.zeros((1, length))
    b[:, y] = 1.
    result = (a.dot(b.T))/np.sqrt(np.sum(a) * np.sum(b))
    return result


def build_graph(data, data_length, distance):
    """
    have not take different similarity into consideration
    default similarity function is jaccard
    :param data:
    :param similarity:
    :return:
    """
    user_num = len(data.keys())
    assert user_num == max(data.keys()) + 1

    init_user_matrix = np.zeros((user_num, user_num))
    for i in tqdm(range(user_num)):
        for j in range(i+1, user_num):
            if distance == 'cosine':
                s = cosine_dis(data[i], data[j], data_length)
            else:  # default Jaccard
                s = jaccard_dis(data[i], data[j])
            init_user_matrix[i][j] = s
            init_user_matrix[j][i] = s

    return init_user_matrix


def keep_topk(init_user_matrix, k):
    if k == 'all' or k > init_user_matrix.shape[0]:
        return init_user_matrix
    else:
        user_num = init_user_matrix.shape[0]
        assert k <= user_num
        # keep top k
        final_user_matrix = np.zeros((user_num, user_num))
        for i in tqdm(range(user_num)):
                ind = init_user_matrix[i].argsort()[-k:]
                # cur_sum = np.sum(init_user_matrix[i][ind])
                for j in ind:

                    final_user_matrix[i][j] = init_user_matrix[i][j]
                    final_user_matrix[j][i] = init_user_matrix[j][i]

        return final_user_matrix


def getEigVec(matrix, cluster):
    eigval, eigvec = np.linalg.eig(matrix)
    eigval = np.real(eigval)
    dim = len(eigval)
    dictEigval = dict(zip(eigval, range(0, dim)))
    kEig = np.sort(eigval)[:cluster]
    ix = [dictEigval[k] for k in kEig]

    vec = np.real(eigvec[:, ix])
    print('vec = ', vec)
    print('vec = ', vec.shape)
    return vec


def GetEignVec(matrix, cluster):
    N = (np.eye(matrix.shape[0]) * float(np.max(matrix))) - matrix
    vals, vecs = eigs(N, k=cluster, which='LM')
    vecs = np.real(vecs)
    return vecs


def laplacian(W, norm_type, d_type, weak_lambda):
    """Computes the symetric normalized laplacian.
    L = D^{-1/2} A D{-1/2}

    d_type = 'leporid' --> LEPORID
    d_type = 'le' --> LE
    """
    D = np.zeros(W.shape)
    D1 = np.zeros(W.shape)
    if d_type == 'degree_centrality':
        w = np.sum(W != 0, axis=1, dtype=float)
    elif d_type == 'strong_reg':
        w1 = np.sum(W != 0, axis=1, dtype=float)
        w2 = np.sum(W, axis=1, dtype=float)
        w = w1 + w2
    elif d_type == 'leporid':
        w1 = np.sum(W, axis=1, dtype=float)
        # print('w1 = ', w1)
        e = np.max(w1)
        # print('e = ', e)
        w = w1 + weak_lambda*(e-w1)
        # print('w = ', w)
        # input('debug')
    else:
        w = np.sum(W, axis=1, dtype=float)
    D1.flat[::len(w) + 1] = w  # set the diag of D to w
    A = D1 - W

    if norm_type == 'sym':
        D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
        return D.dot(A).dot(D)
    elif norm_type == 'rw':
        D.flat[::len(w) + 1] = w ** (-1)  # set the diag of D to w
        return D.dot(A)
    else:
        return A


# codes from https://github.com/axelv/recursive-nystrom

def gauss(X: np.ndarray, Y: np.ndarray=None, gamma=0.01):
    # todo make this implementation more python like!
    if Y is None:
        Ksub = np.ones((X.shape[0], 1))
    else:
        nsq_rows = np.sum(X ** 2, axis=1, keepdims=True)
        nsq_cols = np.sum(Y ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - np.matmul(X, Y.T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = np.exp(-gamma * Ksub)

    return Ksub


def uniformNystrom(X, n_components: int, kernel_func=gauss):
    indices = np.random.choice(X.shape[0], n_components)
    C = kernel_func(X, X[indices,:])
    SKS = C[indices, :]
    W = np.linalg.inv(SKS + 10e-6 * np.eye(n_components))

    return C, W


def nys(G, m, k):
    n = G.shape[0]
    indices = np.random.choice(n, m, replace=False)
    C = G[:, indices]
    W = C[indices, :]

    # C, W = uniformNystrom(G, m)
    V, D, _ = np.linalg.svd(W, full_matrices=False)
    # print('D = ', D)
    # print('D = ', D[-k:])
    # input('debug')
    # print('D = ', min(D) ** -1)
    # print('D**-1 = ', D**-1)
    D = D[-k:]
    # D = np.diag(D[-k:])
    V = V[:, -k:]
    temp = (m/n)**0.5
    # print('np.isnan(np.min(c)) = ', np.isnan(np.min(D**-1)))
    U = C.dot((temp * V).dot(np.diag(D**-1)))
    # print('np.isnan(np.min(c)) = ', np.isnan(np.min(U)))
    # input('debug')
    # print('mean1 = ', np.mean(U))
    # print('val1 = ', np.std(U))
    return U


def normalization(x):
    # x = x / np.linalg.norm(x)
    # print('x = ', x)
    print('x = ', x.shape)
    print('mean1 = ', np.mean(x))
    print('val1 = ', np.std(x))
    x = stats.zscore(x)
    x = np.real(x)
    # print('x = ', x)
    print('mean = ', np.mean(x))
    print('val = ', np.std(x))
    return x


if __name__ == "__main__":
    # Initialize the parameters
    parser = argparse.ArgumentParser(description="Leporid Initialization")
    parser.add_argument('--data_folder', default='../dataset/ml_1m_all', help='the data folder path')
    parser.add_argument('--dataset', default='/ml_1m_all', help='the selected dataset name')
    parser.add_argument('--adj_graph', default=0, help='1 for loading the existing adjacent graph, otherwise 0')
    parser.add_argument('--distance', default='jaccard', help='the distance for building adjacent graph')
    parser.add_argument('--topk', default=1000, help='the topk for building KNN graph')
    parser.add_argument('--d_type', default='leporid',
                        help='Laplacian Eigenmaps Function, including degree_centrality, strong_reg,  leporid and '
                             'le, where le refers to LE and leporid refers to Leporid in paper')
    parser.add_argument('--weak_lambda', default=0.5, help='regularization coefficient, used when d_type is leporid')
    parser.add_argument('--norm', default='sym', help='the normalization function, including sym, rw, and nonorm')
    parser.add_argument('--cluster_num', default=64, help='output embedding size')
    parser.add_argument('--seed', default=123, help='numpy.random.seed')

    args = parser.parse_args()
    print('args = ', args)
    print('time = ', time.asctime(time.localtime(time.time())))

    np.random.seed(args.seed)

    f_train = args.data_folder + args.dataset + '_train.txt'
    result_path = args.data_folder + '/emb/' + args.d_type + str(args.weak_lambda)

    train_users = load_user_dict(f_train)
    train_items = load_item_dict(f_train)
    user_num = len(train_users.keys())
    item_num = len(train_items.keys())

    adj_graph_user = args.data_folder + args.dataset + '_init_user_' + args.distance + '.npy'
    adj_graph_item = args.data_folder + args.dataset + '_init_item_' + args.distance + '.npy'

    folder = os.path.exists(result_path)
    if not folder:
        os.makedirs(result_path)
        print('create new folders')

    result_file_path_t = str(args.cluster_num) + '_k' + str(args.topk) + args.norm + args.distance

    if 1:
        # For user Embedding
        if args.adj_graph:
            user_l = np.load(adj_graph_user)
        else:
            user_l = build_graph(train_users, item_num, args.distance)
            np.save(adj_graph_user, user_l)

        print('user = ', user_l.shape)
        user_l = keep_topk(user_l, args.topk)
        # print('user_l = ', user_l)
        user_lap = laplacian(user_l, args.norm, args.d_type, args.weak_lambda)
        # print('user_lap = ', user_lap)
        print('user_lap = ', user_lap.shape)

        user_emb_file1 = result_path + args.dataset + '_user' + result_file_path_t + '.npy'
        user_emb_file = result_path + args.dataset + '_user' + result_file_path_t + '_norm.npy'

        u_vec = GetEignVec(user_lap, args.cluster_num)

        np.save(user_emb_file1, u_vec)

        # normalize and save to file
        u_vec = normalization(u_vec)
        np.save(user_emb_file, u_vec)

        print('finish user embedding.')

    if 1:
        # For Item Embedding
        if args.adj_graph:
            item_l = np.load(adj_graph_item)
        else:
            item_l = build_graph(train_items, user_num, args.distance)
            np.save(adj_graph_item, item_l)

        print('item_lap = ', np.isnan(np.min(item_l)))
        print('item = ', item_l.shape)
        item_l = keep_topk(item_l, args.topk)
        # print('item_l = ', item_l)
        item_lap = laplacian(item_l, args.norm, args.d_type, args.weak_lambda)
        print('item_lap = ', item_lap.shape)

        item_emb_file1 = result_path + args.dataset + '_item' + result_file_path_t + '.npy'
        item_emb_file = result_path + args.dataset + '_item' + result_file_path_t + '_norm.npy'

        i_vec = GetEignVec(item_lap, args.cluster_num)

        # save to file
        np.save(item_emb_file1, i_vec)
        i_vec = normalization(i_vec)
        np.save(item_emb_file, i_vec)

        print('finish item embedding.')

    if 0:
        # for paper example --> Figure 1
        # Load data
        adj_graph = np.load(args.data_folder + args.dataset + args.dataset + '.npy')
        laplacian_matrix = laplacian(adj_graph, args.norm, args.d_type, args.weak_lambda)
        print('laplacian_matrix = ', laplacian_matrix.shape)

        i_vec = GetEignVec(laplacian_matrix, args.cluster_num)
        print('i_vec = ', i_vec)

        # save to file
        np.savetxt(args.data_folder + args.dataset + args.dataset + args.d_type + '.txt', i_vec, fmt='%.04f')
        i_vec = normalization(i_vec)
        np.savetxt(args.data_folder + args.dataset + args.dataset + args.d_type + '_norm.txt', i_vec, fmt='%.04f')
        print('finish item embedding.')
    print('finish All.')

