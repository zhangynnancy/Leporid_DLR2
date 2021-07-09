import pandas as pd
import numpy as np
import math
import xlwt
import random
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from torch.nn import init
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import SGD
# from jacobian import JacobianReg
# from torch.autograd import Variable
# from torch.optim import RMSprop
# import torch.nn.functional as F

from model import Feature, ActorModel, CriticModel
from memory import Memory
from util import to_tensor, to_numpy


criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda') if USE_CUDA else torch.device('cpu')


# metrics
def PRECISION(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set)) / float(N)
    elif isinstance(N, list):
        return np.array([PRECISION(actual, predicted, n) for n in N])


def RECALL(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set)) / float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([RECALL(actual, predicted, n) for n in N])


def HR(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return min(1, len(inter_set))
    elif isinstance(N, list):
        return np.array([HR(actual, predicted, n) for n in N])


def load_file_dict(args, type):
    if type == 1:
        file_name = args.data_folder + args.dataset + '_train.txt'
        print('file_name = ', file_name)
    elif type == 2:
        file_name = args.data_folder + args.dataset + '_validate.txt'
    else:
        file_name = args.data_folder + args.dataset + '_test.txt'  # Yinan For bottom users
    df = pd.read_csv(file_name, sep=' ', names=["users", "items", "ratings", "time"])
    print('size = ', df.shape)
    test = {}
    user_dict = {}
    all_items = []
    for name, g in df.groupby('users'):
        items = []
        test[name] = g.sort_values(by=["time"], ascending=[1])["items"].values
        for i in test[name]:
            all_items.append(i)
            if i not in items:
                items.append(i)
        user_dict[name] = np.array(items)
    return user_dict, set(all_items)


def load_emb_file(args, topk, distance, norm, dim, type):
    print('Initialize embedding with ', args.initialization)
    if type == 1:
        print('load item emb')
        file_name = args.data_folder + 'emb/' + args.initialization + str(
            args.coefficient) + '/' + args.dataset + '_item' + str(dim) + '_k' + str(
            topk) + norm + distance + '_norm.npy'
    else:
        print('load user emb')
        file_name = args.data_folder + 'emb/' + args.initialization + str(
            args.coefficient) + '/' + args.dataset + '_user' + str(dim) + '_k' + str(
            topk) + norm + distance + '_norm.npy'
    print('file_name = ', file_name)
    emb = np.load(file_name)
    print('emb shape = ', emb.shape)
    return emb


def load_bottom_users(args):
    file_name = args.data_folder + args.dataset + '_train.txt'
    data_file = pd.read_csv(file_name, sep=' ', names=["users", "items", "ratings", "time"])

    user_per_rating = data_file['users'].value_counts()
    bottom_user = np.array(user_per_rating.reset_index().values.tolist())
    bottom_user = bottom_user[:, 0]

    length = len(bottom_user)
    q3 = int(length / 4 * 3)
    # q3 = int(length / 2)
    bottom_user = bottom_user[q3:]
    return bottom_user


def load_bottom_items(args):
    file_name = args.data_folder + args.dataset + '_train.txt'
    data_file = pd.read_csv(file_name, sep=' ', names=["users", "items", "ratings", "time"])

    user_per_rating = data_file['items'].value_counts()
    bottom_items = np.array(user_per_rating.reset_index().values.tolist())
    bottom_items = bottom_items[:, 0]

    length = len(bottom_items)
    q3 = int(length / 4 * 3)
    # q3 = int(length / 2)
    bottom_items = bottom_items[q3:]
    bottom_items = set(bottom_items)
    return bottom_items


class Interaction(object):

    def __init__(self, args, hyperparameters):
        """
        :param args:
        :param args.pretrain_model_type: 0 for no pre-train model, 'both', 'critic', 'actor'
        """

        self.dataset = args.dataset
        self.train_file, self.item_set = load_file_dict(args, type=1)
        self.validate_file, _ = load_file_dict(args, type=2)
        self.test_file, _ = load_file_dict(args, type=0)

        self.add_regularization = args.add_regularization
        self.u_reg = float(args.u_reg)
        self.i_reg = float(args.i_reg)
        self.ifrevnet = args.ifrevnet

        self.args = args
        self.optimizer = args.optimizer
        print('item num = ', len(self.item_set))
        print('user num = ', len(self.train_file))

        self.lr = hyperparameters['lr']
        self.mul_store_size = args.mul_store_size

        self.state_num = int(args.state_num)
        self.batch_size = args.batch_size
        self.dim = hyperparameters['dim']

        self.user_set = list(self.train_file.keys())
        user_num = len(self.user_set)
        self.user_set1 = self.user_set[:int(user_num / 2)]
        self.user_set2 = self.user_set[int(user_num / 2):]

        # for the bottom 25% testing
        self.bottom_user = load_bottom_users(args)
        self.bottom_items = load_bottom_items(args)
        self.bottom_ui = []
        for u in self.bottom_user:
            intersect = list(set(self.train_file[u]) & self.bottom_items)
            if len(intersect) > 0:
                self.bottom_ui.append(u)
        print('self.bottom_ui = ', len(self.bottom_ui))

        self.recommended = []
        self.cur_train_id = -1
        self.cur_state = []
        self.cur_dict = []

        self.TopK = args.TopK
        self.test_rec_num = max(self.TopK)

        # For validation
        self.val_neg_num = args.val_neg_num
        self.val_memory_size = args.val_memory_size
        self.validate_p_memory = Memory(self.val_memory_size)
        self.validate_n_memory = Memory(self.val_memory_size)
        self.store_validation()

        # For train
        self.sample = []
        self.n = 0
        self.hinge_m = 0.5
        self.loss_actor = args.loss_actor
        self.loss_critic = args.loss_critic
        self.pretrain_model_type = args.pretrain_model_type

        self.p_memory = []
        self.n_memory = []

        # initialize networks and memory
        self.Feature_Gen = Feature(len(self.user_set), max(self.item_set) + 1, args, self.dim)
        self.Critic_Network = CriticModel(self.dim)
        self.Actor_Network = ActorModel(self.dim)

        if args.load_trained == 'critic':
            if USE_CUDA:
                print('using gpu')
                self.Feature_Gen = self.Feature_Gen.cuda()
                self.Critic_Network = self.Critic_Network.cuda()
                self.Actor_Network = self.Actor_Network.cuda()
            d = args.initial_type + str(args.lr) + '_bestCritic/'
            path = args.data_folder + d
            self.Feature_Gen.load_state_dict(torch.load(path + 'best_Feature_Gen.pth')['model'])
            self.Critic_Network.load_state_dict(torch.load(path + 'best_Critic.pth')['model'])

        else:
            # initialize embedding layer for Feature
            if args.initialization == 'random':
                # random initialization
                var = 0.01
                print('random initialization with var = ', var)
                self.Feature_Gen.item_embeddings.weight.data.normal_(0, var)
                self.Feature_Gen.user_embeddings.weight.data.normal_(0, var)
            else:
                self.user_emb = load_emb_file(args, hyperparameters['spectopk'], hyperparameters['dis'],
                                              hyperparameters['norm'], hyperparameters['dim'], type=0)
                self.item_emb = load_emb_file(args, hyperparameters['spectopk'], hyperparameters['dis'],
                                              hyperparameters['norm'], hyperparameters['dim'], type=1)
                self.Feature_Gen.item_embeddings = nn.Embedding.from_pretrained(
                    to_tensor(self.item_emb, dtype='float', requires_grad=True))
                self.Feature_Gen.user_embeddings = nn.Embedding.from_pretrained(
                    to_tensor(self.user_emb, dtype='float', requires_grad=True))

            if USE_CUDA:
                print('using gpu')
                self.Feature_Gen = self.Feature_Gen.cuda()
                self.Critic_Network = self.Critic_Network.cuda()
                self.Actor_Network = self.Actor_Network.cuda()
            else:
                print('no gpu')

            # initial code
            self.Feature_Gen_optim = Adam(self.Feature_Gen.parameters(), lr=self.lr)
            self.Critic_Network_optim = Adam(self.Critic_Network.parameters(), lr=self.lr)
            self.Actor_Network_optim = Adam(self.Actor_Network.parameters(), lr=self.lr)

    # Pre-Train Part
    def init(self, user, num):
        cur_state = self.train_file[user][num: num + self.state_num]
        next_rec = self.train_file[user][num + self.state_num]
        return cur_state, next_rec

    def multi_store(self, stored_neg):
        tempt = random.randint(0, stored_neg-1)
        random.shuffle(self.sample)
        for i in range(self.mul_store_size):
            self.p_memory.append(
                [self.sample[i][0], self.sample[i][1], [self.sample[i][2]], [1]])
            self.n_memory.append(
                [self.sample[i][0], self.sample[i][1], [self.sample[i][3][tempt]], [-1]])

    def update(self):
        self.Feature_Gen.train()
        self.Feature_Gen_optim.zero_grad()
        critic_loss = []
        actor_loss = []
        loss_critic, loss_actor = None, None

        if (self.n+1) * int(self.batch_size / 2) > len(self.p_memory):
            be = self.n * int(self.batch_size / 2)
            af = len(self.p_memory)
            self.n = 0
        else:
            be = self.n * int(self.batch_size / 2)
            af = (self.n + 1) * int(self.batch_size / 2)
            self.n += 1

        # positive
        p_samples = self.p_memory[be:af]
        p_user = np.array([x[0] for x in p_samples])
        p_state = np.array([x[1] for x in p_samples])
        p_action = np.array([x[2] for x in p_samples])

        # negative
        n_samples = self.n_memory[be:af]
        n_user = np.array([x[0] for x in n_samples])
        n_state = np.array([x[1] for x in n_samples])
        n_action = np.array([x[2] for x in n_samples])

        # for regularization
        reg_loss = 0
        if self.add_regularization:
            user = np.concatenate((p_user, n_user), axis=None)
            action = np.concatenate((p_action, n_action), axis=None)
            try:
                init_user_emb = self.user_emb[user]
                init_item_emb = self.item_emb[action]
                init_user_emb = to_tensor(init_user_emb, dtype='float')
                init_item_emb = to_tensor(init_item_emb, dtype='float')

                user = to_tensor(user)
                action = to_tensor(action)
                cur_user_emb = self.Feature_Gen.user_embeddings(user)
                cur_item_emb = self.Feature_Gen.item_embeddings(action)
                reg_loss = self.u_reg * (torch.mean(init_user_emb - cur_user_emb) ** 2) + self.i_reg * (
                            torch.mean(init_item_emb - cur_item_emb) ** 2)
            except:
                print('WARNING!')
                print('user = ', user)
                print('action = ', action)
                print('user = ', len(user))
                print('action = ', len(action))
        p_user = to_tensor(p_user)
        p_state = to_tensor(p_state)
        p_action = to_tensor(p_action)

        p_action = self.Feature_Gen.item_embeddings(p_action)
        p_action = torch.squeeze(p_action)
        p_s = self.Feature_Gen(p_user, p_state)

        n_user = to_tensor(n_user)
        n_state = to_tensor(n_state)
        n_action = to_tensor(n_action)

        n_action = self.Feature_Gen.item_embeddings(n_action)
        n_action = torch.squeeze(n_action)
        n_s = self.Feature_Gen(n_user, n_state)

        if self.pretrain_model_type == 'both' or self.pretrain_model_type == 'critic':
            # Critic
            self.Critic_Network.train()
            self.Critic_Network_optim.zero_grad()

            p_reward = self.Critic_Network(p_s, p_action)
            n_reward = self.Critic_Network(n_s, n_action)

            # Note that p_reward should be smaller while n_reward should be larger
            if self.loss_critic == 'hinge':
                # hinge loss
                temp = p_reward - n_reward
                loss_critic = torch.clamp(temp + self.hinge_m, min=0)
                loss_critic = torch.mean(loss_critic)

            elif self.loss_critic == 'log':
                # log loss
                temp = p_reward - n_reward
                loss_critic = torch.log(1 + torch.exp(temp))
                loss_critic = torch.mean(loss_critic)

            elif self.loss_critic == 'square_square':
                # square-square loss
                n_reward = torch.clamp(self.hinge_m - n_reward, min=0)
                loss_critic = p_reward ** 2 + n_reward ** 2
                loss_critic = torch.mean(loss_critic)
            else:
                # square_exp loss
                gama = 1
                n_reward = gama * torch.exp(-n_reward)
                loss_critic = p_reward ** 2 + n_reward
                loss_critic = torch.mean(loss_critic)

            critic_loss.append(float(to_numpy(loss_critic)))

        if self.pretrain_model_type == 'both' or self.pretrain_model_type == 'actor':
            # Actor
            self.Actor_Network.train()
            self.Actor_Network_optim.zero_grad()

            # positive
            p_pse_action = self.Actor_Network(p_s)

            # negative
            n_pse_action = self.Actor_Network(n_s)

            if self.loss_actor == 'hinge':
                # hinge loss
                temp = (p_pse_action - p_action) ** 2 - (n_pse_action - n_action) ** 2
                temp = torch.mean(temp, 1)
                loss_actor = torch.clamp(temp + self.hinge_m, min=0)
                loss_actor = torch.mean(loss_actor)

            elif self.loss_actor == 'log':
                # log loss
                temp = (p_pse_action - p_action) ** 2 - (n_pse_action - n_action) ** 2
                temp = torch.mean(temp, 1)
                loss_actor = torch.log(1 + torch.exp(temp))
                loss_actor = torch.mean(loss_actor)

            elif self.loss_actor == 'square_square':
                # square-square loss
                temp_pos = torch.mean((p_pse_action - p_action) ** 2, 1)
                temp_neg = torch.mean((n_pse_action - n_action) ** 2, 1)
                temp_neg = torch.clamp(self.hinge_m - temp_neg, min=0)
                loss_actor = temp_pos ** 2 + temp_neg ** 2
                loss_actor = torch.mean(loss_actor)
            else:
                # square_exp loss
                gama = 1
                temp_pos = torch.mean((p_pse_action - p_action) ** 2, 1)
                temp_neg = torch.mean((n_pse_action - n_action) ** 2, 1)
                temp_neg = gama * torch.exp(-temp_neg)
                loss_actor = temp_pos ** 2 + temp_neg
                loss_actor = torch.mean(loss_actor)

            actor_loss.append(float(to_numpy(loss_actor)))

        if self.pretrain_model_type == 'both':
            loss = loss_actor + loss_critic
        elif self.pretrain_model_type == 'critic':
            loss = loss_critic
        else:
            loss = loss_actor
        loss += reg_loss
        loss.retain_grad()
        loss.backward()
        self.Feature_Gen_optim.step()
        self.Critic_Network_optim.step()
        self.Actor_Network_optim.step()

        return critic_loss, actor_loss

    # Selection Part for validating and testing
    def select_action_critic(self, data, is_test):
        """
        :param data: data = [user, state]
        :param is_test: candidate set needs to remove validation when testing
        :return: recommend items
        """
        self.Feature_Gen.eval()
        batch_size = 512  # here set the batch=256, can be changed and will not change the results
        user = data[0]
        u = to_tensor(np.array([user]))
        state = to_tensor(data[1][np.newaxis:])
        state = self.Feature_Gen(u, state)

        if is_test:
            val_set = set()
            if user in self.validate_file.keys():
                val_set = set(self.validate_file[user])
            can_items = np.array(list(deepcopy(self.item_set - val_set - set(self.train_file[user]))))
        else:
            if self.dataset == 'ml_1m':
                can_items = np.array(list(deepcopy(self.item_set - set(self.train_file[user]))))
            else:
                val_set = np.array(list(set(self.validate_file[user])))
                negtive = np.random.choice(np.array(list(self.item_set - set(self.train_file[user]))), 1000)
                can_items = np.unique(np.concatenate((val_set, negtive)))
        can_size = can_items.shape[0]

        with torch.no_grad():
            for i in range(math.ceil(can_size / batch_size)):
                if can_size >= (i + 1) * batch_size:
                    can_emb = to_tensor(can_items[i * batch_size:(i + 1) * batch_size])
                    can_emb = self.Feature_Gen.item_embeddings(can_emb)
                    can_state = state.repeat(batch_size, 1)
                else:
                    can_emb = to_tensor(can_items[i * batch_size:])
                    can_emb = self.Feature_Gen.item_embeddings(can_emb)
                    can_state = state.repeat(can_size - i * batch_size, 1)

                critic_reward = self.Critic_Network(can_state, can_emb)

                cur_scores = to_numpy(critic_reward)
                if i == 0:
                    scores = cur_scores
                else:
                    scores = np.append(scores, cur_scores)

        index = list(scores.argsort()[0:self.test_rec_num])
        rec_items = can_items[:, np.newaxis][index, :].T[0]
        return rec_items

    def select_action_actor(self, data, is_test):
        """
        :param data: data = [user, state]
        :param is_test: candidate set needs to remove validation when testing
        :return: recommend items
        """
        self.Feature_Gen.eval()
        user = data[0]
        u = to_tensor(np.array([user]))
        state = to_tensor(data[1][np.newaxis:])
        state = self.Feature_Gen(u, state)

        if is_test:
            val_set = set()
            if user in self.validate_file.keys():
                val_set = set(self.validate_file[user])
            can_items = np.array(list(deepcopy(self.item_set - val_set - set(self.train_file[user]))))
        else:
            if self.dataset == 'ml_1m':
                can_items = np.array(list(deepcopy(self.item_set - set(self.train_file[user]))))
            else:
                val_set = np.array(list(set(self.validate_file[user])))
                negtive = np.random.choice(np.array(list(self.item_set - set(self.train_file[user]))), 1000)
                can_items = np.unique(np.concatenate((val_set, negtive)))
        with torch.no_grad():
            can_emb = to_tensor(can_items)
            can_emb = self.Feature_Gen.item_embeddings(can_emb)

            self.Actor_Network.eval()
            action = self.Actor_Network(state)

        if USE_CUDA:
            action = action.cuda()

        action = action.repeat(can_items.shape[0], 1)
        scores = torch.mean(torch.abs(can_emb - action), dim=1)
        scores = to_numpy(scores)

        rec_items = list(scores.argsort()[0:self.test_rec_num])
        rec_items = can_items[:, np.newaxis][rec_items, :].T[0]

        return rec_items

    # Validation Part
    def store_validation(self):
        for user in self.validate_file.keys():
            cur_state = self.train_file[user][-self.state_num:]
            for num in range(len(self.validate_file[user])):
                next_rec = self.validate_file[user][num]
                self.validate_p_memory.append(user, cur_state, next_rec, 1)

                # add negative samples
                val_neg = np.random.choice(
                    list(self.item_set - set(self.train_file[user]) - set(self.validate_file[user])), self.val_neg_num)
                for i in range(self.val_neg_num):
                    self.validate_n_memory.append(user, cur_state, val_neg[i], -1)

                cur_state = np.delete(cur_state, 0, 0)
                cur_state = np.insert(cur_state, self.state_num - 1, next_rec, 0)

    def loss_validation(self):
        self.Feature_Gen.eval()
        p_loss_critic = []
        n_loss_critic = []

        p_loss_actor = []
        n_loss_actor = []

        # for positive samples
        p_user, p_state, p_action, p_reward = self.validate_p_memory.sample_and_split(int(self.batch_size))
        user = to_tensor(p_user)
        state = to_tensor(p_state)
        action = to_tensor(p_action)
        action = self.Feature_Gen.item_embeddings(action)
        s = self.Feature_Gen(user, state)

        self.Critic_Network.eval()
        with torch.no_grad():
            pse_reward = self.Critic_Network(s, action)
            pse_reward = torch.mean(pse_reward)
            p_loss_critic.append(float(to_numpy(pse_reward)))

            self.Actor_Network.eval()
            pse_action = self.Actor_Network(s)

        loss = criterion(pse_action, action)
        p_loss_actor.append(float(to_numpy(loss)))

        # for negative samples
        n_user, n_state, n_action, n_reward = self.validate_n_memory.sample_and_split(int(self.batch_size))

        user = to_tensor(n_user)
        state = to_tensor(n_state)
        action = to_tensor(n_action)
        action = self.Feature_Gen.item_embeddings(action)
        s = self.Feature_Gen(user, state)

        with torch.no_grad():
            pse_reward = self.Critic_Network(s, action)
            pse_reward = torch.mean(pse_reward)
            n_loss_critic.append(float(to_numpy(pse_reward)))

            pse_action = self.Actor_Network(s)

        loss = criterion(pse_action, action)
        n_loss_actor.append(float(to_numpy(loss)))

        return p_loss_critic, n_loss_critic, p_loss_actor, n_loss_actor

    def validating(self, critic_actor):
        """
        :param critic_actor: 1 for validating critic networks, 0 for actor networks
        :return:
        """
        if critic_actor:
            print('validate critic')
        else:
            print('validate actor')
        prec = np.zeros(len(self.TopK))
        rec = np.zeros(len(self.TopK))
        hr = np.zeros(len(self.TopK))

        recommend = {}
        for user in (self.validate_file.keys()):
            cur_state = self.train_file[user][-self.state_num:]
            if critic_actor:
                rec_items = self.select_action_critic([user, cur_state], is_test=0)
            else:
                rec_items = self.select_action_actor([user, cur_state], is_test=0)
            recommend[user] = rec_items

        for user in recommend.keys():
            actual = self.validate_file[user]
            predict = recommend[user]
            prec += PRECISION(actual, predict, self.TopK)
            rec += RECALL(actual, predict, self.TopK)
            hr += HR(actual, predict, self.TopK)

        num = len(self.validate_file.keys())
        prec = prec / num
        rec = rec / num
        hr = hr / num

        return prec, rec, hr, recommend

    # Test Part
    def testing(self, critic_actor):
        """
        :param critic_actor: 1 for testing critic networks, 0 for actor networks
        :return:
        """
        if critic_actor:
            print('test critic')
        else:
            print('test actor')
        prec = np.zeros(len(self.TopK))
        rec = np.zeros(len(self.TopK))
        hr = np.zeros(len(self.TopK))

        recommend = {}
        for user in (self.test_file.keys()):
            if user in self.validate_file.keys():
                length = len(self.validate_file[user])
            else:
                length = 0
            if length >= self.state_num:
                cur_state = self.validate_file[user][-self.state_num:]
            elif length == 0:
                cur_state = self.train_file[user][-(self.state_num - length):]
            else:
                cur_state1 = self.train_file[user][-(self.state_num - length):]
                cur_state2 = self.validate_file[user]
                cur_state = np.concatenate((cur_state1, cur_state2), axis=0)

            if critic_actor:
                rec_items = self.select_action_critic([user, cur_state], is_test=1)
            else:
                rec_items = self.select_action_actor([user, cur_state], is_test=1)
            recommend[user] = rec_items

        for user in recommend.keys():
            actual = self.test_file[user]
            predict = recommend[user]
            prec += PRECISION(actual, predict, self.TopK)
            rec += RECALL(actual, predict, self.TopK)
            hr += HR(actual, predict, self.TopK)

        num = len(self.test_file.keys())
        prec = prec / num
        rec = rec / num
        hr = hr / num

        return prec, rec, hr, recommend

    def testing_bottom(self, critic_actor):
        """
        :param critic_actor: 1 for testing critic networks, 0 for actor networks
        :return:
        """
        if critic_actor:
            print('test critic')
        else:
            print('test actor')
        prec = np.zeros(len(self.TopK))
        rec = np.zeros(len(self.TopK))
        hr = np.zeros(len(self.TopK))

        recommend = {}
        for user in tqdm(self.bottom_user):
            if user not in self.test_file.keys():
                continue
            if user in self.validate_file.keys():
                length = len(self.validate_file[user])
            else:
                length = 0
            if length >= self.state_num:
                cur_state = self.validate_file[user][-self.state_num:]
            elif length == 0:
                cur_state = self.train_file[user][-(self.state_num - length):]
            else:
                cur_state1 = self.train_file[user][-(self.state_num - length):]
                cur_state2 = self.validate_file[user]
                cur_state = np.concatenate((cur_state1, cur_state2), axis=0)

            if critic_actor:
                rec_items = self.select_action_critic([user, cur_state], is_test=1)
            else:
                rec_items = self.select_action_actor([user, cur_state], is_test=1)
            recommend[user] = rec_items

        for user in recommend.keys():
            actual = self.test_file[user]
            predict = recommend[user]
            prec += PRECISION(actual, predict, self.TopK)
            rec += RECALL(actual, predict, self.TopK)
            hr += HR(actual, predict, self.TopK)

        num = len(self.test_file.keys())
        prec = prec / num
        rec = rec / num
        hr = hr / num

        return prec, rec, hr, recommend

    def testing_bottom_items(self, critic_actor):
        """
        :param critic_actor: 1 for testing critic networks, 0 for actor networks
        :return:
        """
        if critic_actor:
            print('test critic')
        else:
            print('test actor')
        prec = np.zeros(len(self.TopK))
        rec = np.zeros(len(self.TopK))
        hr = np.zeros(len(self.TopK))

        recommend = {}
        for user in tqdm(self.bottom_ui):
            if user not in self.test_file.keys():
                continue
            if user in self.validate_file.keys():
                length = len(self.validate_file[user])
            else:
                length = 0
            if length >= self.state_num:
                cur_state = self.validate_file[user][-self.state_num:]
            elif length == 0:
                cur_state = self.train_file[user][-(self.state_num - length):]
            else:
                cur_state1 = self.train_file[user][-(self.state_num - length):]
                cur_state2 = self.validate_file[user]
                cur_state = np.concatenate((cur_state1, cur_state2), axis=0)

            if critic_actor:
                rec_items = self.select_action_critic([user, cur_state], is_test=1)
            else:
                rec_items = self.select_action_actor([user, cur_state], is_test=1)
            recommend[user] = rec_items

        for user in recommend.keys():
            actual = self.test_file[user]
            predict = recommend[user]
            prec += PRECISION(actual, predict, self.TopK)
            rec += RECALL(actual, predict, self.TopK)
            hr += HR(actual, predict, self.TopK)

        num = len(self.test_file.keys())
        prec = prec / num
        rec = rec / num
        hr = hr / num

        return prec, rec, hr, recommend


