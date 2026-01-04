# asker

- Aqui se define el `suite` del `benchmark`
- Por cada `suite`, extraimos, el conjunto de `tasks` y el de cada dataset
- Por cada `task` extraimos entonces la cantidad de `runs`

```bash
python benchmark/knn_l1.py   --target-csv benchmark/target.csv   --pool-csv original_datasets.csv   --out-csv benchmark/neighbors.csv   --k 5
```

```bash
python benchmark/portfolio_ranking.py   --neighbors-csv benchmark/neighbors.csv   --performance-csv original_datasets.csv   --output-csv benchmark/ranking.csv   --top-k 5
```