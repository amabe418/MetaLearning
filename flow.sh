#!/bin/bash

python benchmark/knn_l1.py   --target-csv data/target.csv   --pool-csv data/original_datasets.csv   --out-csv data/neighbors.csv   --k 5

python benchmark/portfolio_ranking.py   --neighbors-csv data/neighbors.csv   --performance-csv data/original_datasets.csv   --output-csv data/ranking.csv   --top-k 5


