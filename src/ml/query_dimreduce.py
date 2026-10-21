# -*- coding: utf-8 -*-
"""
降维查询工具（容器内执行）：返回降维模型的信息保留、资源开销、
降维前后聚类对照与主要变化方向。

用法：
  /opt/miniconda3/bin/python /opt/ml/query_dimreduce.py --algo pca|svd [--dim 50]

口径：
  - 只返回已登记维数网格 {2,10,20,50} 内的结果，未登记维数 → 明确状态；
  - 聚类对照为聚类任务登记的最优参数在原始画像与降维坐标上的同一口径比较；
  - 二维坐标与散点图文件路径随结果返回（前端经后端端点读取全量坐标）。
"""
import argparse
import json
import os

from common import ARTIFACTS, VERSIONS, check_prereqs, load_registry, fail, ok

EVAL_DIR = os.path.join(ARTIFACTS, "eval")
DIM_GRID = [2, 10, 20, 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["pca", "svd"])
    ap.add_argument("--dim", type=int, default=None)
    args = ap.parse_args()

    okp, msg = check_prereqs()
    if not okp:
        print(json.dumps(fail(msg), ensure_ascii=False))
        return
    reg = load_registry()

    eid = VERSIONS["dim_pca"] if args.algo == "pca" else VERSIONS["dim_svd"]
    entry = next((e for e in reg["entries"] if e["artifact_id"] == eid), None)
    if entry is None:
        print(json.dumps(fail("算法 {} 未登记:模型缺失,请先运行训练流水线".format(args.algo)),
                         ensure_ascii=False))
        return

    dim = args.dim or entry["selected_params"]["n_components"]
    if dim not in DIM_GRID:
        print(json.dumps(fail("指定的维数 {} 无已登记模型（已登记维数 {}），系统不现场训练".format(
            dim, DIM_GRID)), ensure_ascii=False))
        return

    with open(os.path.join(EVAL_DIR, "{}_eval.json".format(args.algo)),
              encoding="utf-8") as f:
        eval_obj = json.load(f)
    variant = next(v for v in eval_obj["variants"] if v["n_components"] == dim)
    with open(os.path.join(ARTIFACTS, "profiles", "profile_meta.json"),
              encoding="utf-8") as f:
        meta = json.load(f)

    print(json.dumps(ok({
        "algorithm": args.algo,
        "version": entry["version"],
        "params": {"n_components": dim},
        "original_dims": {
            "n_users": meta["profile_users"],
            "n_movies": meta["profile_movies"],
            "description": "训练期 user×movie 稀疏画像（未评分=0）",
        },
        "retention": variant["retention"],
        "resource": variant["resource"],
        "cluster_comparison": variant["cluster_comparison"],
        "principal_directions": eval_obj["principal_directions_top_movies"][:5],
        "files_2d": eval_obj["files_2d"],
        "all_registered_dims": DIM_GRID,
        "notes": [eval_obj["pca_note"], eval_obj["cluster_note"]],
    }), ensure_ascii=False))


if __name__ == "__main__":
    main()
