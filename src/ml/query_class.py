# -*- coding: utf-8 -*-
"""
分类查询工具（容器内执行）：预测 (UserID, MovieID) 是否会给出 ≥4 分。

用法：
  /opt/miniconda3/bin/python /opt/ml/query_class.py \
      --algo logreg|dtree|both --user 42 --movie 100 [--at T2]

口径：
  - 特征统计量（F1/F2）来自登记的训练期(≤T1)统计量，预测时点仅支持 T1/T2；
  - 未知用户/电影、模型未登记、前置数据缺失 → 返回明确状态，不生成占位结果；
  - 训练期无评分用户/电影 → 使用全局估计并附加"历史不足/冷启动"说明。
"""
import argparse
import json
import os

import joblib
import numpy as np

from common import (ARTIFACTS, T1_TS, T2_TS, VERSIONS, check_prereqs,
                    load_registry, fail, ok)

PROFILE_DIR = os.path.join(ARTIFACTS, "profiles")
MODELS_DIR = os.path.join(ARTIFACTS, "models")

ALGO_ALIAS = {"logreg": "class-lr-v1.0", "dtree": "class-dt-v1.0",
              "both": "both"}


def _load_tables():
    with open(os.path.join(PROFILE_DIR, "users_table.json"), encoding="utf-8") as f:
        users_table = json.load(f)
    with open(os.path.join(PROFILE_DIR, "movies_table.json"), encoding="utf-8") as f:
        movies_table = json.load(f)
    with open(os.path.join(PROFILE_DIR, "class_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    return users_table, movies_table, stats


def _build_row(user_id, movie_id, encoders, users_table, movies_table, stats):
    """构造单条事件的 F1-F6 特征行（列顺序与训练时完全一致）。"""
    user_mean = stats["user_mean"]
    movie_count = stats["movie_count"]
    global_mean = stats["global_mean"]

    notes = []
    if str(user_id) in user_mean:
        f1 = float(user_mean[str(user_id)])
    else:
        f1 = float(global_mean)
        notes.append("用户 {} 训练期无评分记录(历史不足),F1 使用训练集全局均分 {} 估计".format(
            user_id, global_mean))
    if str(movie_id) in movie_count:
        f2 = float(np.log1p(movie_count[str(movie_id)]))
    else:
        f2 = 0.0
        notes.append("电影 {} 训练期无评分记录(冷启动),F2 按 0 处理".format(movie_id))

    movie_info = movies_table.get(str(movie_id))
    if movie_info is None:
        genres = []
        notes.append("电影 {} 有评分记录但其属性在迭代一被隔离,F3 类型特征按全 0 处理".format(movie_id))
    else:
        genres = movie_info["genres"]
    f3 = encoders["mlb"].transform([genres]).astype(np.float32)

    demo = users_table.get(str(user_id))
    if demo is None:
        # 评分保留但人口属性在迭代一被隔离的用户：用训练默认值，并明确标注
        demo = {"gender": "M", "age": 35, "occupation": 0}
        notes.append("用户 {} 不在清洗后用户表(其人口属性在迭代一被隔离),"
                     "F4-F6 使用训练默认值(性别 M/年龄 35/职业 0)并降低置信度".format(user_id))
    f4 = encoders["age_enc"].transform(np.array([[demo["age"]]], dtype=int))
    f5 = encoders["gender_enc"].transform([[demo["gender"]]])
    f6 = encoders["occ_enc"].transform(np.array([[demo["occupation"]]], dtype=int))

    x_cat = np.hstack([f3, f4, f5, f6]).astype(np.float32)
    x_num = np.array([[f1, f2]], dtype=np.float32)
    x_lr = np.hstack([encoders["scaler"].transform(x_num), x_cat])
    x_dt = np.hstack([x_num, x_cat])
    return (x_lr, x_dt), notes


def _predict_one(algo, user_id, movie_id, at_ts, reg):
    entry = next((e for e in reg["entries"] if e["artifact_id"] == algo), None)
    if entry is None:
        return fail("算法 {} 未登记:模型缺失,请先运行训练流水线".format(algo))
    users_table, movies_table, stats = _load_tables()
    with open(os.path.join(PROFILE_DIR, "user_ids.json"), encoding="utf-8") as f:
        profile_users = set(json.load(f))
    with open(os.path.join(PROFILE_DIR, "movie_ids.json"), encoding="utf-8") as f:
        profile_movies = set(json.load(f))
    # 真正未知 = 不在对应表且训练期无评分；评分保留但属性被迭代一隔离的可预测并标注
    if str(movie_id) not in movies_table and movie_id not in profile_movies:
        return fail("未知 MovieID {}:不在清洗后数据(cleaned-v1.0)中".format(movie_id))
    if str(user_id) not in users_table and user_id not in profile_users:
        return fail("未知 UserID {}:不在清洗后数据(cleaned-v1.0)中".format(user_id))

    encoders = joblib.load(os.path.join(ARTIFACTS, "models", "class_encoders.joblib"))
    model = joblib.load(os.path.join(ARTIFACTS, entry["files"][0]))
    (x_lr, x_dt), notes = _build_row(user_id, movie_id, encoders, users_table,
                                     movies_table, stats)
    if x_lr is None:
        return fail(notes[0])

    x = x_lr if algo == VERSIONS["class_lr"] else x_dt
    proba = float(model.predict_proba(x)[0][1])
    pred = int(proba >= 0.5)

    movie = movies_table.get(str(movie_id),
                             {"title": "(该电影属性在迭代一被隔离)", "genres": []})
    demo = users_table.get(str(user_id),
                           {"gender": "M", "age": 35, "occupation": 0})
    return ok({
        "algorithm": "logistic_regression" if algo == VERSIONS["class_lr"] else "decision_tree",
        "target": {"user_id": user_id, "movie_id": movie_id,
                   "movie_title": movie["title"], "movie_genres": movie["genres"],
                   "user_gender": demo["gender"], "user_age": demo["age"],
                   "user_occupation": demo["occupation"]},
        "at_time": "T2" if at_ts == T2_TS else "T1",
        "prediction": pred,
        "prob_positive": round(proba, 4),
        "threshold": 0.5,
        "model_version": entry["version"],
        "params": entry["selected_params"],
        "eval_metrics": entry["metrics"],
        "notes": notes,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["logreg", "dtree", "both"])
    ap.add_argument("--user", required=True, type=int)
    ap.add_argument("--movie", required=True, type=int)
    ap.add_argument("--at", default="T2", choices=["T1", "T2"])
    args = ap.parse_args()

    okp, msg = check_prereqs()
    if not okp:
        print(json.dumps(fail(msg), ensure_ascii=False))
        return
    reg = load_registry()

    at_ts = T2_TS if args.at == "T2" else T1_TS
    algos = ["class-lr-v1.0", "class-dt-v1.0"] if args.algo == "both" \
        else [ALGO_ALIAS[args.algo]]
    results = [_predict_one(a, args.user, args.movie, at_ts, reg) for a in algos]
    if len(results) == 1:
        print(json.dumps(results[0], ensure_ascii=False))
        return
    # both：两算法对照
    failed = [r for r in results if r["status"] != "completed"]
    done = [r for r in results if r["status"] == "completed"]
    out = {
        "status": "completed" if done else "failed",
        "comparison": done,
        "failures": failed,
        "target": {"user_id": args.user, "movie_id": args.movie},
        "at_time": args.at,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
