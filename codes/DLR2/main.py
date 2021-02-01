import argparse
import psutil
import os
import cProfile
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
import torch
from torch.optim import lr_scheduler
from copy import deepcopy
from scipy import stats
from util import to_tensor, to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xlwt
import math

from interaction import Interaction

np.random.seed(123)


def train(all_data, test_model, save_performance, draw_loss, add_sch):
    # parameter settings
    store_interval = 50
    bestval_test_actor = 0.0
    bestval_test_critic = 0.0
    best_test_actor = 0.0
    best_test_critic = 0.0

    for curdis in args.spec_distance:
        for curtopk in args.spec_topk:
            for curnorm in args.norm:
                for curdim in args.dim:
                    for curlr in args.pre_lr:

                        hyperparameters = {'dim': curdim, 'norm': curnorm, 'lr': curlr, 'spectopk': curtopk,
                                           'dis': curdis}
                        print('current setting = ', hyperparameters)
                        cur_bestval_test_actor = 0.0
                        cur_bestval_test_critic = 0.0
                        cur_best_test_actor = 0.0
                        cur_best_test_critic = 0.0

                        interact = Interaction(args, hyperparameters)
                        if all_data:
                            user_set = interact.user_set
                        else:
                            user_set = interact.user_set1

                        stored_neg = 100
                        neg_num = stored_neg
                        neg = {}

                        # create negative samples
                        can_neg = {}
                        can_neg_num = []
                        for user in user_set:
                            can_neg[user] = np.array(list(interact.item_set - set(interact.train_file[user])))
                            can_neg_num.append(len(can_neg[user]))
                        tmp = np.ones((len(user_set), neg_num))
                        for i in range(len(user_set)):
                            tmp[i] *= can_neg_num[i]
                        ttmp = np.random.rand(len(user_set), neg_num)
                        tmp = tmp * ttmp
                        tmp = np.floor(tmp).astype(np.int32)
                        tmpi = 0
                        for user in user_set:
                            neg[user] = np.squeeze(can_neg[user][:, np.newaxis][list(tmp[tmpi]), :])
                            tmpi += 1
                        del can_neg
                        del tmp
                        del ttmp
                        del can_neg_num
                        # create negative samples

                        # store in memory

                        for user in tqdm(user_set):
                            for num in range(len(interact.train_file[user]) - args.state_num):
                                cur_state, next_rec = interact.pre_init(user, num)
                                interact.pre_sample.append([user, cur_state, next_rec, neg[user][:stored_neg]])
                                random.shuffle(neg[user])

                        # load some pre-trained samples
                        interact.pre_multi_store(stored_neg)

                        if args.pre_trained == 0:
                            # start pre-training
                            critic_loss = []
                            actor_loss = []
                            p_val_cri = []
                            p_val_act = []
                            n_val_cri = []
                            n_val_act = []

                            val_critic = []
                            val_actor = []
                            te_critic = []
                            te_actor = []

                            if add_sch:
                                f_sch = lr_scheduler.CosineAnnealingLR(interact.Feature_Gen_optim, T_max=args.max_pre_iteration)
                                c_sch = lr_scheduler.CosineAnnealingLR(interact.Critic_Network_optim, T_max=args.max_pre_iteration)
                                a_sch = lr_scheduler.CosineAnnealingLR(interact.Actor_Network_optim, T_max=args.max_pre_iteration)

                            for iteration in tqdm(range(args.max_pre_iteration)):
                                if (iteration + 1) % store_interval == 0:
                                    interact.pre_multi_store(stored_neg)

                                c_critic, c_actor = interact.pre_update()
                                critic_loss.append(c_critic)
                                actor_loss.append(c_actor)
                                if add_sch:
                                    f_sch.step()
                                    c_sch.step()
                                    a_sch.step()
                                # print('c_critic = ', c_critic)

                                pc, nc, pa, na = interact.loss_validation()
                                p_val_cri.append(pc)
                                n_val_cri.append(nc)
                                p_val_act.append(pa)
                                n_val_act.append(na)

                                if draw_loss and (iteration + 1) % 500 == 0:  # For test
                                    # critic
                                    plt.figure()
                                    draw_critic_loss = np.array(critic_loss)
                                    mean_critic_loss = np.mean(draw_critic_loss, axis=1)

                                    draw_p_val_cri = np.array(p_val_cri)
                                    mean_p_val_cri = np.mean(draw_p_val_cri, axis=1)

                                    draw_n_val_cri = np.array(n_val_cri)
                                    mean_n_val_cri = np.mean(draw_n_val_cri, axis=1)

                                    x = range(mean_critic_loss.shape[0])
                                    plt.plot(x, mean_critic_loss, color="r", linestyle="-", linewidth=0.5, label='tr_cri')
                                    plt.plot(x, mean_p_val_cri, color="b", linestyle="-", linewidth=0.5, label='pos_val_cri')
                                    plt.plot(x, mean_n_val_cri, color="g", linestyle="-", linewidth=0.5, label='neg_val_cri')

                                    plt.title('critic iteration = ' + str(iteration))
                                    plt.legend()
                                    plt.savefig(args.initialization + '_critic.png')
                                    plt.show()

                                    # actor
                                    plt.figure()
                                    draw_actor_loss = np.array(actor_loss)
                                    mean_actor_loss = np.mean(draw_actor_loss, axis=1)

                                    draw_p_val_act = np.array(p_val_act)
                                    mean_p_val_act = np.mean(draw_p_val_act, axis=1)

                                    draw_n_val_act = np.array(n_val_act)
                                    mean_n_val_act = np.mean(draw_n_val_act, axis=1)

                                    x = range(mean_actor_loss.shape[0])
                                    plt.plot(x, mean_actor_loss, color="r", linestyle="-", linewidth=0.5, label='tr_act')
                                    plt.plot(x, mean_p_val_act, color="b", linestyle="-", linewidth=0.5, label='pos_val_act')
                                    plt.plot(x, mean_n_val_act, color="g", linestyle="-", linewidth=0.5, label='neg_val_act')

                                    plt.title('actor iteration = ' + str(iteration))
                                    plt.legend()
                                    plt.savefig(args.initialization + '_actor.png')
                                    plt.show()

                                if test_model and (iteration + 1) % 100 == 0:
                                    if args.pretrain_model_type == 'critic':
                                        vcprec, vcrec, vchr, _ = interact.validating(1)

                                        if cur_bestval_test_critic <= vcprec[0]:
                                            cur_bestval_test_critic = vcprec[0]
                                            tcprec, tcrec, tchr, _ = interact.testing(1)
                                            val_critic.append([vcprec, vcrec, vchr])
                                            te_critic.append([tcprec, tcrec, tchr])
                                            print('tcprec, tcrec, tchr = ', tcprec, tcrec, tchr)
                                            if cur_best_test_critic <= tcprec[0]:
                                                cur_best_test_critic = tcprec[0]
                                    elif args.pretrain_model_type == 'actor':
                                        vaprec, varec, vahr, _ = interact.validating(0)
                                        taprec, tarec, tahr, _ = interact.testing(0)

                                        val_actor.append([vaprec, varec, vahr])
                                        te_actor.append([taprec, tarec, tahr])

                                        if cur_bestval_test_actor <= vaprec[0]:
                                            cur_bestval_test_actor = vaprec[0]
                                            taprec, tarec, tahr, _ = interact.testing(0)
                                            val_actor.append([vaprec, varec, vahr])
                                            te_actor.append([taprec, tarec, tahr])
                                            # print('taprec, tarec, tahr = ', taprec, tarec, tahr)
                                            if cur_best_test_actor <= taprec[0]:
                                                cur_best_test_actor = taprec[0]
                                    else:
                                        vaprec, varec, vahr, _ = interact.validating(0)
                                        vcprec, vcrec, vchr, _ = interact.validating(1)

                                        if cur_bestval_test_actor <= vaprec[0]:
                                            cur_bestval_test_actor = vaprec[0]
                                            taprec, tarec, tahr, _ = interact.testing(0)
                                            val_actor.append([vaprec, varec, vahr])
                                            te_actor.append([taprec, tarec, tahr])
                                            print('taprec, tarec, tahr = ', taprec, tarec, tahr)
                                            if cur_best_test_actor <= taprec[0]:
                                                cur_best_test_actor = taprec[0]

                                        if cur_bestval_test_critic <= vcprec[0]:
                                            cur_bestval_test_critic = vcprec[0]
                                            tcprec, tcrec, tchr, _ = interact.testing(1)
                                            val_critic.append([vcprec, vcrec, vchr])
                                            te_critic.append([tcprec, tcrec, tchr])
                                            print('tcprec, tcrec, tchr = ', tcprec, tcrec, tchr)
                                            if cur_best_test_critic <= tcprec[0]:
                                                cur_best_test_critic = tcprec[0]

                                    if save_performance:
                                        wb = xlwt.Workbook()
                                        ws = wb.add_sheet('performance')

                                        for i in range(len(args.TopK)):
                                            # actor validation
                                            ws.write(0, i * 12, 'act_val_pre_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 1, 'act_val_rec_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 2, 'act_val_hr_' + str(args.TopK[i]))
                                            # actor test
                                            ws.write(0, i * 12 + 3, 'act_te_pre_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 4, 'act_te_rec_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 5, 'act_te_hr_' + str(args.TopK[i]))
                                            # critic validation
                                            ws.write(0, i * 12 + 6, 'cri_val_pre_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 7, 'cri__val_rec_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 8, 'cri__val_hr_' + str(args.TopK[i]))
                                            # critic test
                                            ws.write(0, i * 12 + 9, 'cri__te_pre_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 10, 'cri__te_rec_' + str(args.TopK[i]))
                                            ws.write(0, i * 12 + 11, 'cri__te_hr_' + str(args.TopK[i]))

                                        for i in range(len(args.TopK)):
                                            for j in range(len(val_actor)):
                                                # actor validation
                                                ws.write(j + 1, i * 12, val_actor[j][0][i])
                                                ws.write(j + 1, i * 12 + 1, val_actor[j][1][i])
                                                ws.write(j + 1, i * 12 + 2, val_actor[j][2][i])
                                            for j in range(len(te_actor)):
                                                # actor test
                                                ws.write(j + 1, i * 12 + 3, te_actor[j][0][i])
                                                ws.write(j + 1, i * 12 + 4, te_actor[j][1][i])
                                                ws.write(j + 1, i * 12 + 5, te_actor[j][2][i])

                                            for j in range(len(val_critic)):
                                                # critic validation
                                                ws.write(j + 1, i * 12 + 6, val_critic[j][0][i])
                                                ws.write(j + 1, i * 12 + 7, val_critic[j][1][i])
                                                ws.write(j + 1, i * 12 + 8, val_critic[j][2][i])
                                            for j in range(len(te_critic)):
                                                # critic test
                                                ws.write(j + 1, i * 12 + 9, te_critic[j][0][i])
                                                ws.write(j + 1, i * 12 + 10, te_critic[j][1][i])
                                                ws.write(j + 1, i * 12 + 11, te_critic[j][2][i])
                                        wb.save(
                                            result_path + args.dataset + '_' + str(curdim) + args.initialization + str(
                                                curlr) + str(curtopk) + str(curdis) + str(args.coefficient) + '.xls')

                                    if bestval_test_actor < cur_bestval_test_actor:
                                        bestval_test_actor = cur_bestval_test_actor
                                        print('new best val_test_actor')
                                        print('setting = ', hyperparameters)
                                        print('bestval_test_actor = ', bestval_test_actor)

                                    if bestval_test_critic < cur_bestval_test_critic:
                                        bestval_test_critic = cur_bestval_test_critic
                                        print('new best bestval_test_critic')
                                        print('setting = ', hyperparameters)
                                        print('bestval_test_critic = ', bestval_test_critic)

                                    if best_test_actor < cur_best_test_actor:
                                        best_test_actor = cur_best_test_actor
                                        print('new best best_test_actor')
                                        print('setting = ', hyperparameters)
                                        print('best_test_actor = ', best_test_actor)

                                        d = args.initial_type + str(curdim) + '_bestActor/'
                                        path = args.data_folder + d
                                        folder = os.path.exists(path)

                                        if not folder:
                                            os.makedirs(path)

                                        fg = {'model': interact.Feature_Gen.state_dict(),
                                              'optimizer': interact.Feature_Gen_optim.state_dict()}
                                        torch.save(fg, path + 'best_Feature_Gen.pth')
                                        actor_best = {'model': interact.Actor_Network.state_dict(),
                                                      'optimizer': interact.Actor_Network_optim.state_dict()}
                                        torch.save(actor_best, path + 'best_Actor.pth')

                                    if best_test_critic < cur_best_test_critic:
                                        best_test_critic = cur_best_test_critic
                                        print('new best best_test_critic')
                                        print('setting = ', hyperparameters)
                                        print('best_test_critic = ', best_test_critic)

                                        d = args.initial_type + str(curdim) + '_bestCritic/'
                                        path = args.data_folder + d
                                        folder = os.path.exists(path)

                                        if not folder:
                                            os.makedirs(path)

                                        fg = {'model': interact.Feature_Gen.state_dict(),
                                              'optimizer': interact.Feature_Gen_optim.state_dict()}
                                        torch.save(fg, path + 'best_Feature_Gen.pth')
                                        critic_best = {'model': interact.Critic_Network.state_dict(),
                                                       'optimizer': interact.Critic_Network_optim.state_dict()}
                                        torch.save(critic_best, path + 'best_Critic.pth')

    return 0


if __name__ == "__main__":
    # Initialize the parameters
    parser = argparse.ArgumentParser(description="Ensemble Learning for Recommendation")
    parser.add_argument('--state_num', default=5, help='the item number defined in the state')
    parser.add_argument('--critic_num', default=1, help='the number of critic networks')
    parser.add_argument('--actor_num', default=1, help='the number of actor networks')
    parser.add_argument('--mul_store_size', default=2000, help='number of samples store in single memory each time')
    parser.add_argument('--TopK', default=[1, 5, 10], help='top K items evaluated for validating and testing')
    parser.add_argument('--optimizer', default='adam', help='the optimizer strategy')
    parser.add_argument('--lars_coef', default=1e-3, help='eta in lars')
    parser.add_argument('--percentage', default=1, help='percentage of training data')
    parser.add_argument('--norm', default=['sym'], help='bpr, svd, rw, sym, ')
    parser.add_argument('--initialization', default='leporid', help='bpr, svd, leporid_nys')
    parser.add_argument('--spec_topk', default=[1000], help='all, 2000, 1000, 500, 100')
    parser.add_argument('--spec_distance', default=['jaccard'], help='cosine, jaccard')
    parser.add_argument('--dim', default=[64], help='embedding dimension')  #
    parser.add_argument('--initial_type', default='weak_reg', help='degree_centrality, spectral, strong_reg, weak_reg')
    parser.add_argument('--add_regularization', default=0, help='if add regularization')
    parser.add_argument('--coefficient', default=0.5, help='regularization coefficient_alpha')
    parser.add_argument('--u_reg', default=0.5, help='hyperparameter in regularize user emb')
    parser.add_argument('--i_reg', default=0.5, help='hyperparameter in regularize item emb')
    parser.add_argument('--ifrevnet', default=0, help='1: use revnet; 0: use resnet; 2: not use resnet or revnet')

    # Validation
    parser.add_argument('--val_neg_num', default=10, help='negative number for validation')
    parser.add_argument('--val_memory_size', default=100000, help='maximum size for memory for validation')

    # Train
    parser.add_argument('--pre_lr', default=[1e-3], help='the learning rate for pre-training')
    parser.add_argument('--pre_loss_actor', default='hinge', help='hinge, log, square_square, square_exp')
    parser.add_argument('--pre_loss_critic', default='square_square', help='hinge, log, square_square, square_exp')
    parser.add_argument('--pre_trained', default=0, help='both, critic, actor and 0')
    parser.add_argument('--load_trained', default=0, help='both, critic, actor and 0')
    parser.add_argument('--pretrain_model_type', default='critic', help='both, critic, actor')
    parser.add_argument('--max_pre_iteration', default=5000, help='max step for pre train process')  # init 1000
    parser.add_argument('--pre_memory_size', default=10000000, help='maximum size for memory')
    parser.add_argument('--pre_batch_size', default=64, help='batch_size')

    # Test
    parser.add_argument('--max_test_iteration', default=1, help='max step for single user test process')

    # Dataset
    parser.add_argument('--data_folder', default='../../dataset/ml_1m_all/', help='the data folder')
    parser.add_argument('--dataset', default='ml_1m_all', help='the dataset')

    args = parser.parse_args()
    print(args)
    print('time = ', time.asctime(time.localtime(time.time())))
    mem = psutil.virtual_memory()
    print('memory = ', int(round(mem.percent)))

    print('pretrain_model_type = ', args.pretrain_model_type)

    result_path = args.dataset + '/' + args.initialization + str(args.ifrevnet) + '_results/'
    folder = os.path.exists(result_path)
    if not folder:
        os.makedirs(result_path)
        print('create new folders')

    interaction = train(all_data=1, test_model=1, save_performance=1, draw_loss=1, add_sch=1)

