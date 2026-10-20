# -*- coding: utf-8 -*-
"""
分类任务训练流水线：预测评分事件是否 ≥4 分（正类）。

方案沿用作业四定稿：
  - 算法：逻辑回归（LR）+ 决策树 CART（DT），各训练一个注册模型；
  - 特征 6 个（F1-F6），F1/F2 统计量只在训练期(≤T1)计算，验证/测试复用
    训练期统计量（满足“不得使用评分事件发生后的信息”）；
  - 超参在验证集(T1,T2] 上按 PR-AUC 选择，测试集(>T2) 只做最终评价；
  - 评价指标：PR-AUC、F1（阈值 0.5）、准确率、混淆矩阵。

输出：models/class_*.joblib + eval/class_*_eval.json + 注册表条目。
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             accuracy_score, confusion_matrix)
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   OrdinalEncoder, MultiLabelBinarizer)

from common import (T1_TS, T2_TS, T1_LABEL, T2_LABEL, STAGE_DIR, ARTIFACTS,
                    DATA_VERSION, VERSIONS, now_str, load_registry,
                    save_registry, fail, ok)

MODELS_DIR = os.path.join(ARTIFACTS, "models")
EVAL_DIR = os.path.join(ARTIFACTS, "eval")
PROFILE_DIR = os.path.join(ARTIFACTS, "profiles")

LR_C_GRID = [0.01, 0.1, 1.0, 10.0]
DT_DEPTH_GRID = [4, 6, 8, None]
DT_LEAF_GRID = [10, 50]
AGE_ORDER = [1, 18, 25, 35, 45, 50, 56]


def _load_stage():
    with open(os.path.join(STAGE_DIR, "encodings.json"), encoding="utf-8") as f:
        encodings = json.load(f)
    ratings = pd.read_csv(os.path.join(STAGE_DIR, "ratings.csv"), sep="::",
                          engine="python", header=None,
                          names=["UserID", "MovieID", "Rating", "Timestamp"],
                          encoding=encodings["ratings"])
    for col in ("UserID", "MovieID", "Rating", "Timestamp"):
        ratings[col] = pd.to_numeric(ratings[col], errors="coerce")
    ratings = ratings.dropna().astype(
        {"UserID": "int64", "MovieID": "int64", "Rating": "int64",
         "Timestamp": "int64"})
    users = pd.read_csv(os.path.join(STAGE_DIR, "users.csv"), sep="::",
                        engine="python", header=None,
                        names=["UserID", "Gender", "Age", "Occupation", "Zip"],
                        encoding=encodings["users"])
    movies = pd.read_csv(os.path.join(STAGE_DIR, "movies.csv"), sep="::",
                         engine="python", header=None,
                         names=["MovieID", "Title", "Genres"],
                         encoding=encodings["movies"])
    return ratings, users, movies


def _load_stats():
    with open(os.path.join(PROFILE_DIR, "class_stats.json"), encoding="utf-8") as f:
        return json.load(f)


def build_features(events, stats, encoders, users, movies):
    """
    构造 F1-F6 特征矩阵（编码器须已拟合）。
    :param events: DataFrame(UserID, MovieID, Rating, Timestamp)
    :return: (X_lr, X_dt, flags)  flags 为冷启动标记统计
    """
    user_mean = stats["user_mean"]
    movie_count = stats["movie_count"]
    global_mean = stats["global_mean"]

    user_demo = users.set_index("UserID")
    movie_genres = movies.set_index("MovieID")["Genres"].fillna("").astype(str)

    u = events["UserID"].to_numpy()
    m = events["MovieID"].to_numpy()

    # F1 用户历史均分（训练期统计量；缺失 -> 全局均分 + 冷启动标记）
    known_users = np.fromiter(map(int, user_mean.keys()), dtype=np.int64)
    cold_user = ~np.isin(u, known_users)
    f1 = np.array([user_mean.get(str(x), global_mean) for x in u], dtype=np.float32)
    # F2 电影历史评分次数（训练期统计量；缺失 -> 0 + 冷启动标记）
    known_movies = np.fromiter(map(int, movie_count.keys()), dtype=np.int64)
    cold_movie = ~np.isin(m, known_movies)
    f2 = np.log1p(np.array([movie_count.get(str(x), 0) for x in m], dtype=np.float32))

    # F3 电影类型 multi-hot（编码器基于全量 cleaned 电影表拟合——类型为静态属性）
    genre_lists = [movie_genres.get(x, "").split("|") if movie_genres.get(x, "") else []
                   for x in m]
    f3 = encoders["mlb"].transform(genre_lists).astype(np.float32)

    # F4 年龄（有序类别）、F5 性别、F6 职业（用户属性为静态信息）
    demos = user_demo.reindex(u).reset_index(drop=True)
    missing_demos = int(demos["Age"].isna().sum())
    age_vals = pd.to_numeric(demos["Age"], errors="coerce").fillna(35).astype(int) \
        .to_numpy().reshape(-1, 1)
    occ_vals = pd.to_numeric(demos["Occupation"], errors="coerce").fillna(0) \
        .astype(int).to_numpy().reshape(-1, 1)
    f4 = encoders["age_enc"].transform(age_vals)
    f5 = encoders["gender_enc"].transform(demos["Gender"].fillna("M")
                                          .to_numpy().reshape(-1, 1))
    f6 = encoders["occ_enc"].transform(occ_vals)

    x_cat = np.hstack([f3, f4, f5, f6]).astype(np.float32)
    x_num = np.column_stack([f1, f2]).astype(np.float32)
    x_lr = np.hstack([encoders["scaler"].transform(x_num), x_cat])
    x_dt = np.hstack([x_num, x_cat])
    return x_lr, x_dt, {"cold_user": int(cold_user.sum()),
                        "cold_movie": int(cold_movie.sum()),
                        "missing_demographics": missing_demos}


def _metric_dict(y, proba, pred):
    return {
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "n": int(len(y)),
        "positive_ratio": round(float(y.mean()), 4),
    }


def _feature_names(encoders):
    names = ["F1_user_mean", "F2_log1p_movie_count"]
    names += ["genre_{}".format(g) for g in encoders["mlb"].classes_]
    names += ["age_{}".format(a) for a in encoders["age_enc"].categories_[0]]
    names += ["gender_{}".format(g) for g in encoders["gender_enc"].categories_[0]]
    names += ["occupation_{}".format(o) for o in encoders["occ_enc"].categories_[0]]
    return names


def train():
    ratings, users, movies = _load_stage()
    stats = _load_stats()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    train_r = ratings[ratings["Timestamp"] <= T1_TS]
    val_r = ratings[(ratings["Timestamp"] > T1_TS) & (ratings["Timestamp"] <= T2_TS)]
    test_r = ratings[ratings["Timestamp"] > T2_TS]

    y_train = (train_r["Rating"] >= 4).to_numpy()
    y_val = (val_r["Rating"] >= 4).to_numpy()
    y_test = (test_r["Rating"] >= 4).to_numpy()

    # 编码器与标准化器：拟合在静态属性（电影类型、用户人口属性）上，
    # 数值标准化器只拟合训练期特征
    encoders = {
        "mlb": MultiLabelBinarizer().fit(
            [g.split("|") for g in movies["Genres"].fillna("").astype(str) if g]),
        "age_enc": OrdinalEncoder(categories=[AGE_ORDER],
                                  handle_unknown="use_encoded_value",
                                  unknown_value=-1),
        "gender_enc": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        "occ_enc": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        "scaler": StandardScaler(),
    }
    encoders["age_enc"].fit(users[["Age"]])
    encoders["gender_enc"].fit(users[["Gender"]])
    encoders["occ_enc"].fit(users[["Occupation"]])
    # 在训练期数值特征上拟合 scaler（先粗算 F1/F2）
    u_tr = train_r["UserID"].to_numpy()
    m_tr = train_r["MovieID"].to_numpy()
    user_mean = stats["user_mean"]
    movie_count = stats["movie_count"]
    f1_tr = np.array([user_mean.get(str(x), stats["global_mean"]) for x in u_tr],
                     dtype=np.float32)
    f2_tr = np.log1p(np.array([movie_count.get(str(x), 0) for x in m_tr],
                              dtype=np.float32))
    encoders["scaler"].fit(np.column_stack([f1_tr, f2_tr]).astype(np.float64))

    # 训练集/验证集/测试集特征（编码器统一在训练期数据上拟合）
    X_train_lr, X_train_dt, fl_tr = build_features(train_r, stats, encoders,
                                                   users, movies)
    X_val_lr, X_val_dt, fl_val = build_features(val_r, stats, encoders,
                                                users, movies)
    X_test_lr, X_test_dt, fl_test = build_features(test_r, stats, encoders,
                                                   users, movies)

    split_sizes = {"ratings_train": int(len(train_r)), "ratings_val": int(len(val_r)),
                   "ratings_test": int(len(test_r))}

    # ---- 逻辑回归：C 网格，验证集 PR-AUC 选参 ----
    lr_grid = []
    for c in LR_C_GRID:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                 random_state=42)
        clf.fit(X_train_lr, y_train)
        proba_val = clf.predict_proba(X_val_lr)[:, 1]
        m_val = _metric_dict(y_val, proba_val, (proba_val >= 0.5).astype(int))
        lr_grid.append({"C": c, "val": m_val})
    lr_best = max(lr_grid, key=lambda g: g["val"]["pr_auc"])
    lr = LogisticRegression(C=lr_best["C"], class_weight="balanced",
                            max_iter=2000, random_state=42)
    lr.fit(X_train_lr, y_train)

    lr_metrics = {
        "train": _metric_dict(y_train, lr.predict_proba(X_train_lr)[:, 1],
                              lr.predict(X_train_lr)),
        "val": lr_best["val"],
        "test": _metric_dict(y_test, lr.predict_proba(X_test_lr)[:, 1],
                             lr.predict(X_test_lr)),
    }
    fnames = _feature_names(encoders)
    top_idx = np.argsort(np.abs(lr.coef_[0]))[::-1][:15]
    lr_top = [{"feature": fnames[i], "weight": round(float(lr.coef_[0][i]), 4)}
              for i in top_idx]

    # ---- 决策树：max_depth × min_samples_leaf 网格，验证集 PR-AUC 选参 ----
    dt_grid = []
    for depth in DT_DEPTH_GRID:
        for leaf in DT_LEAF_GRID:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                         class_weight="balanced", random_state=42)
            clf.fit(X_train_dt, y_train)
            proba_val = clf.predict_proba(X_val_dt)[:, 1]
            m_val = _metric_dict(y_val, proba_val, (proba_val >= 0.5).astype(int))
            dt_grid.append({"max_depth": depth, "min_samples_leaf": leaf,
                            "val": m_val})
    dt_best = max(dt_grid, key=lambda g: g["val"]["pr_auc"])
    dt = DecisionTreeClassifier(max_depth=dt_best["max_depth"],
                                min_samples_leaf=dt_best["min_samples_leaf"],
                                class_weight="balanced", random_state=42)
    dt.fit(X_train_dt, y_train)
    dt_metrics = {
        "train": _metric_dict(y_train, dt.predict_proba(X_train_dt)[:, 1],
                              dt.predict(X_train_dt)),
        "val": dt_best["val"],
        "test": _metric_dict(y_test, dt.predict_proba(X_test_dt)[:, 1],
                             dt.predict(X_test_dt)),
    }
    dt_top = [{"feature": fnames[i], "importance": round(float(v), 4)}
              for i, v in sorted(enumerate(dt.feature_importances_),
                                 key=lambda kv: -kv[1])[:15]]

    # ---- 落盘模型与编码器 ----
    joblib.dump(lr, os.path.join(MODELS_DIR, "class_lr.joblib"))
    joblib.dump(dt, os.path.join(MODELS_DIR, "class_dt.joblib"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "class_encoders.joblib"))

    common_cfg = {
        "data_version": DATA_VERSION,
        "time_boundary": {"T1": T1_LABEL, "T2": T2_LABEL,
                          "T1_timestamp": T1_TS, "T2_timestamp": T2_TS},
        "split_sizes": split_sizes,
        "label": "rating >= 4 为正类（1-3 为负类）",
        "features": {
            "F1": "用户历史均分（训练期统计量，缺失用全局均分并标注冷启动）",
            "F2": "电影历史评分次数 log1p（训练期统计量，缺失为 0 并标注冷启动）",
            "F3": "电影类型 multi-hot",
            "F4": "年龄有序编码",
            "F5": "性别 one-hot",
            "F6": "职业 one-hot",
        },
        "evaluation": "验证集按 PR-AUC 选参；测试集最终评价。PR-AUC 对不平衡更敏感，F1 为调和平均。",
    }

    lr_eval = dict(common_cfg)
    lr_eval.update({
        "artifact_id": VERSIONS["class_lr"], "algorithm": "logistic_regression",
        "version": VERSIONS["class_lr"],
        "grid": lr_grid, "selected_params": {"C": lr_best["C"],
                                             "class_weight": "balanced"},
        "metrics": lr_metrics, "top_features": lr_top,
        "cold_start_flags": {"train": fl_tr, "val": fl_val, "test": fl_test},
    })
    dt_eval = dict(common_cfg)
    dt_eval.update({
        "artifact_id": VERSIONS["class_dt"], "algorithm": "decision_tree",
        "version": VERSIONS["class_dt"],
        "grid": dt_grid, "selected_params": {"max_depth": dt_best["max_depth"],
                                             "min_samples_leaf": dt_best["min_samples_leaf"],
                                             "class_weight": "balanced"},
        "metrics": dt_metrics, "top_features": dt_top,
        "cold_start_flags": {"train": fl_tr, "val": fl_val, "test": fl_test},
    })
    with open(os.path.join(EVAL_DIR, "class_lr_eval.json"), "w", encoding="utf-8") as f:
        json.dump(lr_eval, f, ensure_ascii=False, indent=2)
    with open(os.path.join(EVAL_DIR, "class_dt_eval.json"), "w", encoding="utf-8") as f:
        json.dump(dt_eval, f, ensure_ascii=False, indent=2)

    # ---- 注册 ----
    reg = load_registry() or {"meta": {}, "entries": []}
    for eid, alg, params, metrics, files in [
        (VERSIONS["class_lr"], "logreg", {"C": lr_best["C"],
                                          "class_weight": "balanced"},
         {"val": lr_best["val"], "test": lr_metrics["test"]},
         ["models/class_lr.joblib", "models/class_encoders.joblib",
          "eval/class_lr_eval.json"]),
        (VERSIONS["class_dt"], "dtree", {"max_depth": dt_best["max_depth"],
                                         "min_samples_leaf": dt_best["min_samples_leaf"],
                                         "class_weight": "balanced"},
         {"val": dt_best["val"], "test": dt_metrics["test"]},
         ["models/class_dt.joblib", "models/class_encoders.joblib",
          "eval/class_dt_eval.json"]),
    ]:
        entry = {
            "artifact_id": eid, "task": "classification", "algorithm": alg,
            "version": eid, "data_version": DATA_VERSION,
            "selected_params": params,
            "metrics": metrics,
            "files": files, "created_at": now_str(),
        }
        reg["entries"] = [e for e in reg["entries"]
                          if e.get("artifact_id") != eid]
        reg["entries"].append(entry)
    save_registry(reg)

    return ok({
        "lr": {"params": lr_best, "test_pr_auc": lr_metrics["test"]["pr_auc"],
               "test_f1": lr_metrics["test"]["f1"]},
        "dt": {"params": dt_best, "test_pr_auc": dt_metrics["test"]["pr_auc"],
               "test_f1": dt_metrics["test"]["f1"]},
    })


if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=False))
