# asker

- Aqui se define el `suite` del `benchmark`
- Por cada `suite`, extraimos, el conjunto de `tasks` y el de cada dataset
- Por cada `task` extraimos entonces la cantidad de `runs`

```bash
python knn_l1.py   --target-csv target.csv   --pool-csv openml_cc18_metafeatures_selected_norm.csv   --out-csv neighbors.csv   --k 5
```