# -*- coding: utf-8 -*-
"""
聚类查询工具（容器内执行）：查询用户（或用户集合）的群体归属与群体摘要。

用法：
  /opt/miniconda3/bin/python /opt/ml/query_cluster.py \
      --algo kmeans|dbscan --users "1,2,3" 或 "1-100" \
      [--k 6] [--eps 0.4 --min_samples 5]

口径：
  - 只返回已登记的参数网格内的结果；未登记的参数 → 明确状态，不现场重训；
  - DBSCAN 的 -1 为噪声点标记；训练期评分 <5 的用户附加"历史不足"提示；
  - 训练期无评分（不在画像中）的用户 → "未纳入群体画像"状态说明。
"""
import argparse
import json
import os
import re

import numpy as np

from common import (ARTIFACTS, VERSIONS, LOW_HISTORY_THRESHOLD, check_prereqs,
                    load_registry, fail, ok)

LABELS_DIR = os.path.join(ARTIFACTS, "labels")
SUMM_DIR = os.path.join(ARTIFACTS, "summaries")
EVAL_DIR = os.path.join(ARTIFACTS, "eval")
PROFILE_DIR = os.path.join(ARTIFACTS, "profiles")

KMEANS_K_GRID = [5, 6, 7, 8]


def _parse_users(spec):
    """解析用户集合："1,2,3" / "1-100" / "42"。"""
    users = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            if b - a > 2000:
                return None, "区间过大（最多 2000 个用户），请缩小范围"
            users.extend(range(a, b + 1))
        elif part.isdigit():
            users.append(int(part))
        else:
            return None, "无法解析用户集合: {}".format(part)
    if not users:
        return None, "未提供有效用户"
    return sorted(set(users)), None


def _pick_variant(entry, args):
    """根据用户指定参数选择已登记变体；未指定用登记最优参数。"""
    if entry["algorithm"] == "kmeans":
        k = args.k or entry["selected_params"]["k"]
        if k not in KMEANS_K_GRID:
            return None, ("指定的 k={} 无已登记模型（已登记 k={}），"
                          "系统不现场训练".format(k, KMEANS_K_GRID))
        return {"k": k, "labels_file": "kmeans_k{}_train.npz".format(k),
                "summary_file": "kmeans_k{}.json".format(k),
                "metrics": entry["metrics"]}, None
    # dbscan
    eps = args.eps or entry["selected_params"]["eps"]
    ms = args.min_samples or entry["selected_params"]["min_samples"]
    key = "eps{}_ms{}".format(str(eps).replace(".", "p"), ms)
    grid_keys = [g["key"] for g in
                 json.load(open(os.path.join(EVAL_DIR, "dbscan_eval.json"),
                                encoding="utf-8"))["grid"]]
    if key not in grid_keys:
        registered = ", ".join("eps={} min_samples={}".format(
            g["eps"], g["min_samples"]) for g in
            json.load(open(os.path.join(EVAL_DIR, "dbscan_eval.json"),
                           encoding="utf-8"))["grid"])
        return None, ("指定的 eps={}, min_samples={} 无已登记模型（已登记参数: {}），"
                      "系统不现场训练".format(eps, ms, registered))
    return {"eps": eps, "min_samples": ms,
            "labels_file": "dbscan_{}_train.npz".format(key),
            "summary_file": "dbscan_{}.json".format(key),
            "metrics": entry["metrics"]}, None


def _load_misc():
    with open(os.path.join(PROFILE_DIR, "user_ids.json"), encoding="utf-8") as f:
        user_ids = json.load(f)
    with open(os.path.join(PROFILE_DIR, "users_table.json"), encoding="utf-8") as f:
        users_table = json.load(f)
    with open(os.path.join(PROFILE_DIR, "class_stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    return user_ids, users_table, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["kmeans", "dbscan"])
    ap.add_argument("--users", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--min_samples", type=int, default=None)
    args = ap.parse_args()

    okp, msg = check_prereqs()
    if not okp:
        print(json.dumps(fail(msg), ensure_ascii=False))
        return
    reg = load_registry()

    eid = VERSIONS["clust_kmeans"] if args.algo == "kmeans" \
        else VERSIONS["clust_dbscan"]
    entry = next((e for e in reg["entries"] if e["artifact_id"] == eid), None)
    if entry is None:
        print(json.dumps(fail("算法 {} 未登记:模型缺失,请先运行训练流水线".format(args.algo)),
                         ensure_ascii=False))
        return

    users, perr = _parse_users(args.users)
    if perr:
        print(json.dumps(fail(perr), ensure_ascii=False))
        return

    variant, verr = _pick_variant(entry, args)
    if verr:
        print(json.dumps(fail(verr), ensure_ascii=False))
        return

    user_ids, users_table, stats = _load_misc()
    uid_index = {u: i for i, u in enumerate(user_ids)}
    labels = np.load(os.path.join(LABELS_DIR, variant["labels_file"]))["labels"]
    with open(os.path.join(SUMM_DIR, variant["summary_file"]), encoding="utf-8") as f:
        summaries = {c["cluster_id"]: c for c in json.load(f)["clusters"]}

    results = []
    unknown_users = []
    touched_clusters = set()
    for u in users:
        row = {"user_id": u}
        if u in uid_index:
            lab = int(labels[uid_index[u]])
            row["status"] = "clustered"
            row["cluster"] = lab
            if lab == -1:
                row["noise"] = True
            else:
                row["noise"] = False
                touched_clusters.add(lab)
            cnt = stats["user_count"].get(str(u), 0)
            if cnt < LOW_HISTORY_THRESHOLD:
                row["low_history"] = True
                row["low_history_note"] = ("训练期内仅 {} 条评分（<{}），画像可信度较低"
                                           .format(cnt, LOW_HISTORY_THRESHOLD))
            else:
                row["low_history"] = False
            if str(u) not in users_table:
                row["no_demographics"] = True
                row["message"] = "该用户有训练期评分但不在清洗后用户表（人口属性在迭代一被隔离），群体归属仅基于评分画像"
            results.append(row)
            continue
        if str(u) in users_table:
            row["status"] = "no_profile"
            row["message"] = "该用户训练期无评分记录，未纳入群体画像（不属于任何群体）"
            results.append(row)
            continue
        row["status"] = "unknown"
        unknown_users.append(u)
        results.append(row)

    def _fmt_genres(top_genres):
        return [g["genre"] + "(" + ("+" if g["delta"] >= 0 else "") +
                str(g["delta"]) + ")" for g in top_genres[:3]]

    # 群体摘要：仅返回涉及的群体（用户集合较大时全部摘要会过于冗长）
    groups = []
    for c in sorted(touched_clusters):
        s = summaries[c]
        groups.append({
            "cluster_id": c,
            "size": s["size"],
            "avg_rating": s["avg_rating"],
            "top_genres": _fmt_genres(s["top_genres"]),
            "top_movies": [{"title": m["title"], "avg_rating": m["avg_rating"]}
                           for m in s["top_movies"][:3]],
            "demographics": s["demographics"],
        })
    if not touched_clusters and len(results) == 1 and results[0]["status"] == "clustered":
        # 单用户查询且命中群体时也附上摘要
        c = results[0]["cluster"]
        if c != -1:
            s = summaries[c]
            groups.append({
                "cluster_id": c, "size": s["size"], "avg_rating": s["avg_rating"],
                "top_genres": _fmt_genres(s["top_genres"]),
                "top_movies": [{"title": m["title"], "avg_rating": m["avg_rating"]}
                               for m in s["top_movies"][:3]],
                "demographics": s["demographics"],
            })

    print(json.dumps(ok({
        "algorithm": "kmeans" if args.algo == "kmeans" else "dbscan",
        "version": entry["version"],
        "params": variant,
        "eval_metrics": entry["metrics"],
        "noise_support": args.algo == "dbscan",
        "queried_users": len(results),
        "users": results,
        "groups": groups,
        "unknown_users": unknown_users,
    }), ensure_ascii=False))


if __name__ == "__main__":
    main()
