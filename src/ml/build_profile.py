# -*- coding: utf-8 -*-
"""
构建迭代二的基础数据层（运行一次，供分类/聚类/降维共用）：

  1. 按 T1/T2 将清洗后评分切分为 训练期(≤T1) / 验证期(T1,T2] / 测试期(>T2)；
  2. 构建用户偏好画像 = 训练期 user × movie 稀疏评分矩阵（未评分=0）；
     同时为聚类参数选择/最终评价构建验证期、测试期画像（同一用户/电影口径）；
  3. 计算分类所需的训练期统计量（用户均分、电影评分次数、全局均分）；
  4. 输出到 /opt/ml-artifacts/profiles/，元信息写入 profile_meta.json。

时间纪律：
  - 画像只用训练期数据构建；验证期画像仅用于算法选择，测试期画像仅用于最终评价；
  - 分类统计量只在训练期计算（满足“不得使用评分事件发生后的信息”）。
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import (T1_TS, T2_TS, T1_LABEL, T2_LABEL, STAGE_DIR, ARTIFACTS,
                    LOW_HISTORY_THRESHOLD, VERSIONS, now_str, fail, ok)

PROFILE_DIR = os.path.join(ARTIFACTS, "profiles")


def _load_table(name, encoding):
    path = os.path.join(STAGE_DIR, "{}.csv".format(name))
    if name == "ratings":
        df = pd.read_csv(path, sep="::", engine="python", header=None,
                         names=["UserID", "MovieID", "Rating", "Timestamp"],
                         encoding=encoding)
        df["UserID"] = pd.to_numeric(df["UserID"], errors="coerce")
        df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
        bad = df.isna().any(axis=1).sum()
        if bad:
            # 清洗后理论上不应出现；保留日志，剔除无法解析的行
            df = df.dropna().astype({"UserID": "int64", "MovieID": "int64",
                                     "Rating": "int64", "Timestamp": "int64"})
        return df, bad
    if name == "users":
        df = pd.read_csv(path, sep="::", engine="python", header=None,
                         names=["UserID", "Gender", "Age", "Occupation", "Zip"],
                         encoding=encoding)
        df["UserID"] = pd.to_numeric(df["UserID"], errors="coerce")
        df = df.dropna().astype({"UserID": "int64"})
        return df, 0
    if name == "movies":
        df = pd.read_csv(path, sep="::", engine="python", header=None,
                         names=["MovieID", "Title", "Genres"], encoding=encoding)
        df["MovieID"] = pd.to_numeric(df["MovieID"], errors="coerce")
        df = df.dropna().astype({"MovieID": "int64"})
        df["Title"] = df["Title"].fillna("").str.strip()
        df["Genres"] = df["Genres"].fillna("").astype(str)
        return df, 0
    raise ValueError(name)


def _to_matrix(df, user_ids, movie_ids):
    """(user, movie) 评分 -> csr 矩阵，行=user_ids 顺序，列=movie_ids 顺序。"""
    ui = {u: i for i, u in enumerate(user_ids)}
    mi = {m: j for j, m in enumerate(movie_ids)}
    df = df[df["UserID"].isin(ui) & df["MovieID"].isin(mi)]
    rows = df["UserID"].map(ui).to_numpy()
    cols = df["MovieID"].map(mi).to_numpy()
    vals = df["Rating"].to_numpy(dtype=np.float32)
    return sp.csr_matrix((vals, (rows, cols)),
                         shape=(len(user_ids), len(movie_ids)), dtype=np.float32), len(df)


def build():
    os.makedirs(PROFILE_DIR, exist_ok=True)

    # 1. 读取清洗后数据（编码在 fetch 阶段已探测）
    with open(os.path.join(STAGE_DIR, "encodings.json"), encoding="utf-8") as f:
        encodings = json.load(f)
    ratings, bad_rows = _load_table("ratings", encodings["ratings"])
    users, _ = _load_table("users", encodings["users"])
    movies, _ = _load_table("movies", encodings["movies"])

    n_ratings = len(ratings)
    n_users = len(users)
    n_movies = len(movies)

    # 2. 时间切分
    train_r = ratings[ratings["Timestamp"] <= T1_TS]
    val_r = ratings[(ratings["Timestamp"] > T1_TS) & (ratings["Timestamp"] <= T2_TS)]
    test_r = ratings[ratings["Timestamp"] > T2_TS]

    # 3. 画像口径：训练期出现过的用户与电影
    user_ids = sorted(train_r["UserID"].unique().tolist())
    movie_ids = sorted(train_r["MovieID"].unique().tolist())

    train_mat, train_nnz = _to_matrix(train_r, user_ids, movie_ids)
    val_mat, val_nnz = _to_matrix(val_r, user_ids, movie_ids)
    test_mat, test_nnz = _to_matrix(test_r, user_ids, movie_ids)

    val_only_movies = len(set(val_r["MovieID"]) - set(movie_ids))
    test_only_movies = len(set(test_r["MovieID"]) - set(movie_ids))

    # 4. 分类训练期统计量（只用 ≤T1 数据）
    user_mean = {k: float(v) for k, v in
                 train_r.groupby("UserID")["Rating"].mean().round(6).to_dict().items()}
    user_count = {k: int(v) for k, v in
                  train_r.groupby("UserID")["Rating"].count().to_dict().items()}
    movie_count = {k: int(v) for k, v in
                   train_r.groupby("MovieID")["Rating"].count().to_dict().items()}
    global_mean = round(float(train_r["Rating"].mean()), 6)
    low_history_users = [u for u in user_ids if user_count.get(u, 0) < LOW_HISTORY_THRESHOLD]

    # 5. 落盘
    sp.save_npz(os.path.join(PROFILE_DIR, "train_matrix.npz"), train_mat)
    sp.save_npz(os.path.join(PROFILE_DIR, "val_matrix.npz"), val_mat)
    sp.save_npz(os.path.join(PROFILE_DIR, "test_matrix.npz"), test_mat)
    with open(os.path.join(PROFILE_DIR, "user_ids.json"), "w", encoding="utf-8") as f:
        json.dump(user_ids, f)
    with open(os.path.join(PROFILE_DIR, "movie_ids.json"), "w", encoding="utf-8") as f:
        json.dump(movie_ids, f)

    # 用户/电影属性表（供查询与迭代三复用）
    users_table = {
        str(r.UserID): {"gender": r.Gender, "age": int(r.Age),
                        "occupation": int(r.Occupation), "zip": str(r.Zip)}
        for r in users.itertuples()
    }
    movies_table = {
        str(r.MovieID): {"title": r.Title,
                         "genres": [g for g in r.Genres.split("|") if g]}
        for r in movies.itertuples()
    }
    with open(os.path.join(PROFILE_DIR, "users_table.json"), "w", encoding="utf-8") as f:
        json.dump(users_table, f, ensure_ascii=False)
    with open(os.path.join(PROFILE_DIR, "movies_table.json"), "w", encoding="utf-8") as f:
        json.dump(movies_table, f, ensure_ascii=False)

    class_stats = {
        "user_mean": user_mean,
        "user_count": user_count,
        "movie_count": movie_count,
        "global_mean": global_mean,
    }
    with open(os.path.join(PROFILE_DIR, "class_stats.json"), "w", encoding="utf-8") as f:
        json.dump(class_stats, f, ensure_ascii=False)

    meta = {
        "profile_version": VERSIONS["profile"],
        "data_version": "cleaned-v1.0",
        "time_boundary": {"T1": T1_LABEL, "T2": T2_LABEL,
                          "T1_timestamp": T1_TS, "T2_timestamp": T2_TS},
        "split_sizes": {
            "ratings_total": n_ratings,
            "ratings_train": len(train_r),
            "ratings_val": len(val_r),
            "ratings_test": len(test_r),
            "unparseable_rows_dropped": int(bad_rows),
        },
        "users_cleaned": n_users,
        "movies_cleaned": n_movies,
        "profile_users": len(user_ids),
        "profile_movies": len(movie_ids),
        "train_nnz": int(train_nnz),
        "val_nnz": int(val_nnz),
        "test_nnz": int(test_nnz),
        "val_only_movies": int(val_only_movies),
        "test_only_movies": int(test_only_movies),
        "low_history_threshold": LOW_HISTORY_THRESHOLD,
        "low_history_users_train": len(low_history_users),
        "positive_ratio_train": round(float((train_r["Rating"] >= 4).mean()), 6),
        "created_at": now_str(),
    }
    with open(os.path.join(PROFILE_DIR, "profile_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return ok({"profile_dir": PROFILE_DIR, "meta": meta})


if __name__ == "__main__":
    import sys
    print(json.dumps(build(), ensure_ascii=False))
