# LEPORID_DLR2
This is our implementation for the paper:

>Yinan Zhang, Boyang Li, Yong Liu, Hao Wang and Chunyan Miao (2021). Initialization Matters: Regularizing Manifold-informed Initialization for Neural Recommendation Systems. Accepeted by KDD 2021.


## Introduction

Proper initialization is crucial to the optimization and the generalization of neural networks. However, most existing neural recommendation systems initialize the user and item embeddings randomly. In this work, we propose a new initialization scheme for user and item embeddings called Laplacian Eigenmaps with Popularity-based Regularization for Isolated Datum (LEPORID). LEPORID endows the embeddings with information regarding multi-scale neighborhood structures on the data manifold and performs adaptive regularization to compensate for high embedding variance on the tail of the data distribution. Exploiting matrix sparsity, LEPORID embeddings can be computed efficiently. We evaluate LEPORID in a wide range of neural recommendation models. In contrast to the recent surprising finding that the simple K-nearest-neighbor (KNN) method often outperforms neural recommendation systems, we show that existing neural systems initialized with LEPORID often perform on par or better than KNN. To maximize effects of the initialization, we propose the Dual-Loss Residual Recommendation (DLR^2) network, which, when initialized with LEPORID, substantially outperforms both traditional and state-of-the-art neural recommender systems. 

## Citation

If you want to use our codes in your research, please cite:
```
@inproceedings{leporid_dlr2,
  author    = {Yinan Zhang and
               Boyang Li and
               Yong Liu and
               Hao Wang and
               Chunyan Miao,
  title     = {Initialization Matters: Regularizing Manifold-informed Initialization for Neural 
               Recommendation Systems},
  booktitle = {Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery
               and Data Mining, {SIGKDD} 2021, Singapore, August 14-18, 2021.},
  year      = {2021},
}
```

## Enviroment

Environment Requirement --> requirements.txt

## Datasets

The dataset should be placed in the folder:
> args.data_folder + args.dataset + '_train.txt'

with the following structure:
> users, items, ratings, time

Example dataset is placed in dataset/ml_1m_all/ml_1m_all_train.txt. Note that, both "ratings" and "time" are not used in the Leporid Initialization.

The initialization results will be saved in the folder:
> args.data_folder + '/emb/' + args.d_type + str(args.weak_lambda)

## Run the Codes

Run the simulator(Fig.2) in the paper:
> python codes/simulator_fig2.py

Run the le or leporid initialization:
> python codes/le_leporid.py

Run the our new proposed recommendation model, DLR2:
> python codes/DLR2/main.py




