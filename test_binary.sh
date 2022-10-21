#! /bin/bash

#python main_kaggle.py -b 64 -lr 1e-5 -is 256 -cs 224 -train train_clahe_binary.csv -test validation_clahe_binary.csv
#python main_kaggle.py -b 64 -lr 1e-4 -is 256 -cs 224 -train train_clahe_binary.csv -test validation_clahe_binary.csv
python main_kaggle.py -b 16 -lr 1e-3 -is 512 -cs 448
python main_kaggle.py -b 16 -lr 1e-4 -is 512 -cs 448 
python main_kaggle.py -b 16 -lr 1e-5 -is 512 -cs 448 
