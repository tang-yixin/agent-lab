# -*- coding: utf-8 -*-
"""
一键训练流水线（容器内执行）：拉取数据 → 构建画像 → 分类 → 聚类 → 降维 → 注册。

由宿主机 scripts/train-ml.ps1 调用：
  docker exec -u root lab2-hadoop /opt/miniconda3/bin/python /opt/ml/train_all.py

运行时长参考：约 5-15 分钟（主要耗时在 DBSCAN 网格与轮廓系数计算）。
"""
import json
import traceback

import fetch_data
import build_profile
import train_class
import train_cluster
import train_dimreduce
from common import ARTIFACTS, DATA_VERSION, T1_TS, T2_TS, T1_LABEL, T2_LABEL, \
    VERSIONS, load_registry, save_registry, now_str


def step(name, fn):
    print("=" * 60)
    print("[STEP] {}".format(name))
    print("=" * 60, flush=True)
    try:
        result = fn()
        print("[OK] {} -> {}".format(name, json.dumps(result, ensure_ascii=False)))
        return result
    except Exception:  # noqa: BLE001
        print("[FAIL] {}".format(name))
        traceback.print_exc()
        raise


def main():
    step("1/5 从 HDFS 拉取清洗后数据", fetch_data.fetch)
    meta = step("2/5 构建画像与时间切分", build_profile.build)
    step("3/5 分类训练（LR + DT）", train_class.train)
    step("4/5 聚类训练（KMeans + DBSCAN）", train_cluster.train)
    step("5/5 降维训练（PCA + SVD）", train_dimreduce.train)

    # 汇总注册表 meta（供前置校验与版本对照）
    reg = load_registry() or {"meta": {}, "entries": []}
    reg["meta"] = {
        "data_version": DATA_VERSION,
        "time_boundary": {"T1": T1_LABEL, "T2": T2_LABEL,
                          "T1_timestamp": T1_TS, "T2_timestamp": T2_TS},
        "profile_version": VERSIONS["profile"],
        "split_sizes": meta["meta"]["split_sizes"],
        "created_at": now_str(),
    }
    save_registry(reg)
    print("[DONE] 训练流水线完成，注册表已更新: {}/registry.json".format(ARTIFACTS))


if __name__ == "__main__":
    main()
