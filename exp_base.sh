# n_embed = 1, 2, 3, 4
python exp_base_001.py --exp_name "n_embed=1" --exp_num 1 --clean_num 0 --drop_last 1 --n_embed 1
python exp_base_001.py --exp_name "Baseline" --exp_num 2 --clean_num 0 --drop_last 1 --n_embed 2
python exp_base_001.py --exp_name "n_embed=3" --exp_num 3 --clean_num 0 --drop_last 1 --n_embed 3
python exp_base_001.py --exp_name "n_embed=4" --exp_num 4 --clean_num 0 --drop_last 1 --n_embed 4

python create_plot.py --exp_num "1 2 3 4" --metric "AUC"
python create_plot.py --exp_num "1 2 3 4" --metric "F1"


# max_features = 80k, .., 130k
python exp_base_001.py --exp_name "mf=80k" --exp_num 5 --clean_num 0 --max_features 80000 --drop_last 1
python exp_base_001.py --exp_name "mf=100k" --exp_num 6 --clean_num 0 --max_features 100000 --drop_last 1
python exp_base_001.py --exp_name "mf=110k" --exp_num 7 --clean_num 0 --max_features 110000 --drop_last 1
python exp_base_001.py --exp_name "mf=120k" --exp_num 8 --clean_num 0 --max_features 120000 --drop_last 1
python exp_base_001.py --exp_name "mf=130k" --exp_num 9 --clean_num 0 --max_features 130000 --drop_last 1
python exp_base_001.py --exp_name "mf=140k" --exp_num 10 --clean_num 0 --max_features 140000 --drop_last 1
python exp_base_001.py --exp_name "mf=150k" --exp_num 11 --clean_num 0 --max_features 150000 --drop_last 1

python create_plot.py --exp_num "5 2 6 7 8 9" --metric "AUC"
python create_plot.py --exp_num "5 2 6 7 8 9" --metric "F1"


# PME w or wo ReLU
python exp_base_001.py --exp_name "PME without ReLU" --exp_num 12 --clean_num 0 --drop_last 1 --n_embed 2 --pme_relu 0

python create_plot.py --exp_num "2 12" --metric "AUC"
python create_plot.py --exp_num "2 12" --metric "F1"

# Use EMA or not
python exp_base_001.py --exp_name "No EMA" --exp_num 13 --clean_num 0 --drop_last 1 --n_embed 2 --mu 0

python create_plot.py --exp_num "2 13" --metric "AUC"
python create_plot.py --exp_num "2 13" --metric "F1"



