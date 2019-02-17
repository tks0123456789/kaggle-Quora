Change data_dir and embed_dir in exp_base_001.py to your file locations.

See exp_all.sh and log/*.json for experimental details.


It took 16-7 hours in one experiment on my PC(i7 4790k, 32M, GTX1070).



![AUC_n_embed](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/AUC_n_embed%3D1__Baseline__n_embed%3D3__n_embed%3D4.png)
![F1_n_embed](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/F1_n_embed%3D1__Baseline__n_embed%3D3__n_embed%3D4.png)

![AUC_EMA](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/AUC_Baseline__No%20EMA.png)
![F1_EMA](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/F1_Baseline__No%20EMA.png)

![AUC_PME_ReLU](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/AUC_Baseline__PME%20without%20ReLU.png)
![F1_PME_ReLU](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/F1_Baseline__PME%20without%20ReLU.png)

![AUC_MF](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/AUC_mf%3D80k__Baseline__mf%3D100k__mf%3D110k__mf%3D120k__mf%3D130k.png)
![F1_MF](https://raw.githubusercontent.com/tks0123456789/kaggle-Quora/master/F1_mf%3D80k__Baseline__mf%3D100k__mf%3D110k__mf%3D120k__mf%3D130k.png)
