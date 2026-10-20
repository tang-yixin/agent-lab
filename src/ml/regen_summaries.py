# -*- coding: utf-8 -*-
"""
从已保存的群体归属（labels/*.npz）重算群体摘要（无需重新拟合聚类）。

用途：_summarize 口径修正后快速刷新 summaries/*.json。
单独执行：/opt/miniconda3/bin/python /opt/ml/regen_summaries.py
"""
import json
import os

import numpy as np
import scipy.sparse as sp

from common import ARTIFACTS
from train_cluster import (_load_all, _genre_cols, _summarize,
                           _global_genre_mean)

LABELS_DIR = os.path.join(ARTIFACTS, "labels")
SUMM_DIR = os.path.join(ARTIFACTS, "summaries")


def main():
    (train_mat, _val, _test, user_ids, movie_ids,
     users_table, movies_table, _meta) = _load_all()
    genre_cols, _missing = _genre_cols(movie_ids, movies_table)
    global_genre_mean = _global_genre_mean(train_mat, genre_cols)
    os.makedirs(SUMM_DIR, exist_ok=True)
    fixed = 0
    for fname in sorted(os.listdir(LABELS_DIR)):
        if not fname.endswith(".npz"):
            continue
        labels = np.load(os.path.join(LABELS_DIR, fname))["labels"]
        out_name = fname.replace("_train.npz", ".json")
        if fname.startswith("kmeans"):
            k = int(fname.split("_k")[1].split("_")[0])
            n_clusters = k
            payload = {"k": k}
        else:
            key = fname[len("dbscan_"):-len("_train.npz")]
            eps = float(key.split("_ms")[0].replace("eps", "").replace("p", "."))
            ms = int(key.split("_ms")[1])
            n_clusters = len(set(labels.tolist()) - {-1})
            payload = {"eps": eps, "min_samples": ms}
        payload["clusters"] = _summarize(labels, n_clusters, user_ids,
                                         train_mat, users_table, movies_table,
                                         movie_ids, genre_cols,
                                         global_genre_mean)
        out = os.path.join(SUMM_DIR, out_name)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        fixed += 1
    print(json.dumps({"status": "completed", "regenerated": fixed},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
