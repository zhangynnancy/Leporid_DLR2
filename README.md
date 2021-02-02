# LEPORID

Environment Requirement --> requirements.txt

Simulation in Figure 1 --> codes/simulator.py

Generate leporid embeddings --> codes/spectral_emb_gen.py

Our proposed recommendation model --> codes/DLR2/main.py

### For Leporid Initialization

The dataset should be placed in the folder:
> args.data_folder + args.dataset + args.dataset + '_train.txt'

with the following structure:
> users, items, ratings, time

Example dataset is placed in dataset/ml_1m/ml_1m_train.txt. Note that, both "ratings" and "time" are not used in the Leporid Initialization.

The results will be saved in the folder:
> args.data_folder + args.dataset + '/emb/' + args.d_type

Run the codes:
> python spectral_emb_gen.py

### For our new proposed recommendation model, DLR2

Run the codes:
> python DLR2/main.py




