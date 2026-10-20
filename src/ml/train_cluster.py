# -*- coding: utf-8 -*-
"""
聚类任务训练流水线：根据用户历史观影偏好（训练期 user×movie 稀疏画像）划分群体。

  - 算法：K-Means（k 网格，余弦空间）+ DBSCAN（余弦距离，eps 由 5-近邻距离
    分位数确定，min_samples 网格），各登记一组参数网格与一个最优配置；
  - 口径：画像只用训练期(≤T1)构建；验证期(T1,T2] 画像用于参数选择，
    测试期(>T2) 画像用于最终评价（时间纪律与分类任务一致）；
  - 评价指标：Silhouette（余弦）、SSE（K-Means）、噪声点占比与簇数（DBSCAN）；
  - 群体摘要：人数、平均评分、Top 类型偏好、代表电影、人口属性构成（可解释性）；
  - 训练期内评分 <5 的用户标记"历史不足"，查询时单独说明。

输出：models/*.joblib + labels/*.npz + summaries/*.json + eval/*_eval.json + 注册表。
"""
import json
import os

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize

from common import (T1_TS, T2_TS, T1_LABEL, T2_LABEL, ARTIFACTS, DATA_VERSION,
                    VERSIONS, LOW_HISTORY_THRESHOLD, now_str, load_registry,
                    save_registry, fail, ok)

PROFILE_DIR = os.path.join(ARTIFACTS, "profiles")
MODELS_DIR = os.path.join(ARTIFACTS, "models")
LABELS_DIR = os.path.join(ARTIFACTS, "labels")
SUMM_DIR = os.path.join(ARTIFACTS, "summaries")
EVAL_DIR = os.path.join(ARTIFACTS, "eval")

K_GRID = [5, 6, 7, 8]
MS_GRID = [3, 5]
EPS_PERCENTILES = [50, 70, 90]


def _load_all():
    with open(os.path.join(PROFILE_DIR, "user_ids.json"), encoding="utf-8") as f:
        user_ids = json.load(f)
    with open(os.path.join(PROFILE_DIR, "movie_ids.json"), encoding="utf-8") as f:
        movie_ids = json.load(f)
    with open(os.path.join(PROFILE_DIR, "users_table.json"), encoding="utf-8") as f:
        users_table = json.load(f)
    with open(os.path.join(PROFILE_DIR, "movies_table.json"), encoding="utf-8") as f:
        movies_table = json.load(f)
    with open(os.path.join(PROFILE_DIR, "profile_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    train_mat = sp.load_npz(os.path.join(PROFILE_DIR, "train_matrix.npz"))
    val_mat = sp.load_npz(os.path.join(PROFILE_DIR, "val_matrix.npz"))
    test_mat = sp.load_npz(os.path.join(PROFILE_DIR, "test_matrix.npz"))
    return (train_mat, val_mat, test_mat, user_ids, movie_ids,
            users_table, movies_table, meta)


def _genre_cols(movie_ids, movies_table):
    """电影类型 -> 该类型电影在画像矩阵中的列下标。

    注：个别电影有训练期评分但其属性在迭代一被隔离（不在清洗后电影表），跳过并计数。
    """
    genres = {}
    missing = 0
    for j, mid in enumerate(movie_ids):
        info = movies_table.get(str(mid))
        if info is None:
            missing += 1
            continue
        for g in info["genres"]:
            genres.setdefault(g, []).append(j)
    return genres, missing


def _global_genre_mean(train_mat, genre_cols):
    """全量训练画像上每个类型的平均评分（用于计算群体相对偏好偏差）。"""
    out = {}
    for g, cols in genre_cols.items():
        sl = train_mat[:, cols]
        if sl.nnz:
            out[g] = float(sl.sum() / sl.nnz)
    return out


def _active_mask(mat):
    """行内至少一条评分的用户掩码（零向量用户不参与轮廓系数计算）。"""
    return np.asarray(mat.sum(axis=1)).ravel() > 0


def _silhouette(mat, labels, active):
    """余弦轮廓系数：只统计非噪声点且画像非空的样本。"""
    idx = active & (labels != -1)
    if idx.sum() < 2 or len(set(labels[idx])) < 2:
        return None
    sub = mat[idx]
    if sp.issparse(sub):
        sub = sub.tocsr()
    return float(silhouette_score(sub, labels[idx], metric="cosine"))


def _kmeans_fit_eval(mat, k):
    """在给定画像上训练 KMeans 并计算指标。"""
    active = _active_mask(mat)
    Xn = sk_normalize(mat, norm="l2", copy=True).tocsr()
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(Xn)
    sil = _silhouette(mat, labels, active)
    return {
        "model": km,
        "labels": labels,
        "silhouette": sil,
        "sse": round(float(km.inertia_), 2),
        "cluster_sizes": np.bincount(labels, minlength=k).tolist(),
        "n_clusters": k,
    }


def _dbscan_fit_eval(mat, eps, min_samples):
    """在给定画像上训练 DBSCAN 并计算指标。

    零向量行（该时期无评分）不参与拟合：否则零向量彼此距离为 0，
    会聚成一个无意义的"零画像伪簇"。指标口径为"活跃用户中"的簇数/噪声比。
    """
    active = _active_mask(mat)
    sub = mat[active].tocsr()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    sub_labels = db.fit_predict(sub)
    labels = np.full(mat.shape[0], -1, dtype=int)
    labels[active] = sub_labels
    n_clusters = len(set(sub_labels) - {-1})
    noise_ratio = round(float((sub_labels == -1).mean()), 4)
    sil = _silhouette(mat, labels, active)
    sizes = [int((sub_labels == c).sum()) for c in sorted(set(sub_labels) - {-1})]
    return {
        "labels": labels,
        "silhouette": sil,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "cluster_sizes": sizes,
    }


def _summarize(labels, n_clusters, user_ids, train_mat, users_table,
               movies_table, movie_ids, genre_cols, global_genre_mean):
    """生成每个群体的可解释摘要。"""
    summaries = []
    valid = sorted(set(labels) - {-1})
    for c in valid:
        rows = np.where(labels == c)[0]
        sub = train_mat[rows]
        n = len(rows)
        # 平均评分只统计实际评过分的条目（稀疏矩阵的隐含 0 不参与）
        avg_rating = float(sub.sum() / sub.nnz) if sub.nnz else None
        # Top 类型偏好：按"群体均分 - 全局该类型均分"排序（相对偏好，可区分群体）
        genre_means = {}
        for g, cols in genre_cols.items():
            sl = sub[:, cols]
            if sl.nnz:
                m = float(sl.sum() / sl.nnz)
                base = global_genre_mean.get(g)
                if base is not None:
                    genre_means[g] = {"mean": round(m, 3),
                                      "delta": round(m - base, 3)}
        top_genres = [{"genre": g, **v} for g, v in
                      sorted(genre_means.items(), key=lambda kv: -kv[1]["delta"])[:5]]
        # 代表电影（群内 ≥10 条评分、平均 ≥4 分，按均分×数量综合排序）
        col_cnt = np.asarray(sub.getnnz(axis=0)).ravel()
        col_sum = np.asarray(sub.sum(axis=0)).ravel()
        col_mean = np.divide(col_sum, col_cnt, out=np.zeros_like(col_sum),
                             where=col_cnt > 0)
        cand = [(j, col_mean[j], col_cnt[j]) for j in range(sub.shape[1])
                if col_cnt[j] >= 10 and col_mean[j] >= 4.0
                and str(movie_ids[j]) in movies_table]
        cand.sort(key=lambda t: (-t[1], -t[2]))
        top_movies = [{"movie_id": movie_ids[j],
                       "title": movies_table[str(movie_ids[j])]["title"],
                       "avg_rating": round(float(col_mean[j]), 2),
                       "n_ratings": int(col_cnt[j])} for j, _, _ in cand[:5]]
        # 人口属性构成
        demos = []
        for i in rows:
            uid = str(user_ids[i])
            if uid in users_table:
                demos.append(users_table[uid])
        if demos:
            male = sum(1 for d in demos if d["gender"] == "M")
            ages = [d["age"] for d in demos]
            occs = [d["occupation"] for d in demos]
            top_age = max(set(ages), key=ages.count)
            top_occ = max(set(occs), key=occs.count)
            demo_sum = {"gender_male_ratio": round(male / len(demos), 3),
                        "top_age": top_age, "top_occupation": top_occ}
        else:
            demo_sum = {}
        summaries.append({
            "cluster_id": int(c), "size": int(n),
            "avg_rating": round(avg_rating, 3) if avg_rating else None,
            "top_genres": top_genres, "top_movies": top_movies,
            "demographics": demo_sum,
        })
    return summaries


def train():
    (train_mat, val_mat, test_mat, user_ids, movie_ids,
     users_table, movies_table, meta) = _load_all()
    for d in (MODELS_DIR, LABELS_DIR, SUMM_DIR, EVAL_DIR):
        os.makedirs(d, exist_ok=True)
    genre_cols, movies_missing = _genre_cols(movie_ids, movies_table)
    global_genre_mean = _global_genre_mean(train_mat, genre_cols)

    time_cfg = {
        "data_version": DATA_VERSION,
        "time_boundary": {"T1": T1_LABEL, "T2": T2_LABEL,
                          "T1_timestamp": T1_TS, "T2_timestamp": T2_TS},
        "profile": "训练期 user×movie 稀疏画像（未评分=0），余弦空间聚类",
        "movies_missing_from_cleaned_table": movies_missing,
        "note": "{} 部电影有训练期评分但其属性在迭代一被隔离，"
                "画像列保留、类型偏好与代表电影跳过这些电影".format(movies_missing),
    }

    # ================= K-Means =================
    kmeans_grid = []
    for k in K_GRID:
        # 训练期画像上训练
        tr = _kmeans_fit_eval(train_mat, k)
        # 验证期画像上评价（用于选参）
        va = _kmeans_fit_eval(val_mat, k)
        # 测试期画像上最终评价
        te = _kmeans_fit_eval(test_mat, k)
        kmeans_grid.append({
            "k": k,
            "train": {"silhouette": tr["silhouette"], "sse": tr["sse"],
                      "cluster_sizes": tr["cluster_sizes"]},
            "val": {"silhouette": va["silhouette"], "sse": va["sse"]},
            "test": {"silhouette": te["silhouette"], "sse": te["sse"]},
        })
        # 保存模型、训练期归属、群体摘要（每个 k 均登记，供 Agent 按参数查询）
        joblib.dump(tr["model"], os.path.join(MODELS_DIR, "kmeans_k{}.joblib".format(k)))
        np.savez(os.path.join(LABELS_DIR, "kmeans_k{}_train.npz".format(k)),
                 labels=tr["labels"])
        summaries = _summarize(tr["labels"], k, user_ids, train_mat, users_table,
                               movies_table, movie_ids, genre_cols,
                               global_genre_mean)
        with open(os.path.join(SUMM_DIR, "kmeans_k{}.json".format(k)), "w",
                  encoding="utf-8") as f:
            json.dump({"k": k, "clusters": summaries}, f, ensure_ascii=False,
                      indent=2)

    # 按验证集轮廓系数选参；无有效轮廓系数时退化为 SSE 最小
    kmeans_best = max(kmeans_grid, key=lambda g: g["val"]["silhouette"]
                      if g["val"]["silhouette"] is not None else -1e9)
    best_k = kmeans_best["k"]

    # ================= DBSCAN =================
    # eps 候选：训练期画像 (min_samples+1)-近邻余弦距离的 50/70/90 分位。
    # 说明：分位数越高 eps 越大 -> 核心点越多、噪声越少；与 min_samples 对齐，
    # 保证"p% 的用户在其 (ms+1) 邻域半径 eps 内至少有 ms 个邻居"。
    dbscan_grid = []
    for ms in MS_GRID:
        nn = NearestNeighbors(n_neighbors=ms + 1, metric="cosine").fit(train_mat.tocsr())
        dists, _ = nn.kneighbors(train_mat.tocsr(), n_neighbors=ms + 1)
        knn = dists[:, -1]
        for p in EPS_PERCENTILES:
            eps = round(float(np.percentile(knn, p)), 4)
            tr = _dbscan_fit_eval(train_mat, eps, ms)
            va = _dbscan_fit_eval(val_mat, eps, ms)
            te = _dbscan_fit_eval(test_mat, eps, ms)
            key = "eps{}_ms{}".format(str(eps).replace(".", "p"), ms)
            dbscan_grid.append({
                "key": key, "eps": eps, "min_samples": ms,
                "train": {"silhouette": tr["silhouette"],
                          "n_clusters": tr["n_clusters"],
                          "noise_ratio": tr["noise_ratio"],
                          "cluster_sizes": tr["cluster_sizes"]},
                "val": {"silhouette": va["silhouette"],
                        "n_clusters": va["n_clusters"],
                        "noise_ratio": va["noise_ratio"]},
                "test": {"silhouette": te["silhouette"],
                         "n_clusters": te["n_clusters"],
                         "noise_ratio": te["noise_ratio"]},
            })
            np.savez(os.path.join(LABELS_DIR, "dbscan_{}_train.npz".format(key)),
                     labels=tr["labels"])
            summaries = _summarize(tr["labels"], tr["n_clusters"], user_ids,
                                   train_mat, users_table, movies_table,
                                   movie_ids, genre_cols, global_genre_mean)
            with open(os.path.join(SUMM_DIR, "dbscan_{}.json".format(key)), "w",
                      encoding="utf-8") as f:
                json.dump({"eps": eps, "min_samples": ms,
                           "clusters": summaries}, f, ensure_ascii=False,
                          indent=2)

    # 选参：验证集上 2≤簇数≤15 且噪声<50% 的配置中轮廓系数最高；否则取轮廓系数最高者
    def _dbscan_select(g):
        v = g["val"]
        sil = v["silhouette"] if v["silhouette"] is not None else -1
        good = 2 <= v["n_clusters"] <= 15 and v["noise_ratio"] < 0.5
        return (good, sil)
    dbscan_best = max(dbscan_grid, key=_dbscan_select)

    # ================= 落盘评价与注册 =================
    kmeans_eval = dict(time_cfg)
    kmeans_eval.update({
        "artifact_id": VERSIONS["clust_kmeans"], "algorithm": "kmeans",
        "version": VERSIONS["clust_kmeans"],
        "space": "余弦空间（画像行 L2 归一化后 KMeans）",
        "grid": kmeans_grid,
        "selected_params": {"k": best_k},
        "selected": {"train": kmeans_best["train"], "val": kmeans_best["val"],
                     "test": kmeans_best["test"]},
        "low_history_threshold": LOW_HISTORY_THRESHOLD,
        "low_history_users_train": meta["low_history_users_train"],
    })
    dbscan_eval = dict(time_cfg)
    dbscan_eval.update({
        "artifact_id": VERSIONS["clust_dbscan"], "algorithm": "dbscan",
        "version": VERSIONS["clust_dbscan"],
        "space": "余弦距离，eps 取训练期画像 (min_samples+1)-近邻距离的 {} 分位".format(EPS_PERCENTILES),
        "grid": dbscan_grid,
        "selected_params": {"eps": dbscan_best["eps"],
                            "min_samples": dbscan_best["min_samples"]},
        "selected": {"train": dbscan_best["train"], "val": dbscan_best["val"],
                     "test": dbscan_best["test"]},
    })
    with open(os.path.join(EVAL_DIR, "kmeans_eval.json"), "w", encoding="utf-8") as f:
        json.dump(kmeans_eval, f, ensure_ascii=False, indent=2)
    with open(os.path.join(EVAL_DIR, "dbscan_eval.json"), "w", encoding="utf-8") as f:
        json.dump(dbscan_eval, f, ensure_ascii=False, indent=2)

    reg = load_registry() or {"meta": {}, "entries": []}
    for eid, alg, params, metrics, files in [
        (VERSIONS["clust_kmeans"], "kmeans", {"k": best_k},
         {"val": kmeans_best["val"], "test": kmeans_best["test"]},
         ["models/kmeans_k{}.joblib".format(best_k),
          "labels/kmeans_k{}_train.npz".format(best_k),
          "summaries/kmeans_k{}.json".format(best_k),
          "eval/kmeans_eval.json"]),
        (VERSIONS["clust_dbscan"], "dbscan",
         {"eps": dbscan_best["eps"], "min_samples": dbscan_best["min_samples"]},
         {"val": dbscan_best["val"], "test": dbscan_best["test"]},
         ["labels/dbscan_{}_train.npz".format(dbscan_best["key"]),
          "summaries/dbscan_{}.json".format(dbscan_best["key"]),
          "eval/dbscan_eval.json"]),
    ]:
        entry = {
            "artifact_id": eid, "task": "clustering", "algorithm": alg,
            "version": eid, "data_version": DATA_VERSION,
            "selected_params": params,
            "param_grid": {"kmeans": K_GRID, "dbscan_min_samples": MS_GRID,
                           "dbscan_eps_percentiles": EPS_PERCENTILES},
            "metrics": metrics, "files": files, "created_at": now_str(),
        }
        reg["entries"] = [e for e in reg["entries"]
                          if e.get("artifact_id") != eid]
        reg["entries"].append(entry)
    save_registry(reg)

    return ok({
        "kmeans": {"best_k": best_k, "val_silhouette": kmeans_best["val"]["silhouette"],
                   "test_silhouette": kmeans_best["test"]["silhouette"]},
        "dbscan": {"best": dbscan_best["key"],
                   "eps": dbscan_best["eps"], "ms": dbscan_best["min_samples"],
                   "val_silhouette": dbscan_best["val"]["silhouette"],
                   "val_noise": dbscan_best["val"]["noise_ratio"],
                   "test_noise": dbscan_best["test"]["noise_ratio"]},
    })


if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=False))
