# -*- coding: utf-8 -*-
"""
迭代二共享模块：时间边界、HDFS 路径、注册表读写、前置校验。

供容器内训练/查询脚本（train_*.py / query_*.py）共同使用。
所有脚本在 lab2-hadoop 容器内以 root 身份执行（docker exec -u root）。
"""
import json
import os
import subprocess
from datetime import datetime, timezone

# ---- 时间边界（迭代一定稿，与 config/version.json 一致）----
T1_TS = 978307199      # 2000-12-31 23:59:59 UTC，训练期截止
T2_TS = 1009843199     # 2001-12-31 23:59:59 UTC，验证期截止

T1_LABEL = "2000-12-31 23:59:59 UTC"
T2_LABEL = "2001-12-31 23:59:59 UTC"

# ---- 数据与路径 ----
HDFS_BIN = "/opt/hadoop/bin/hdfs"

CLEANED_RATINGS = "/user/hadoop/cleaned/ratings"
CLEANED_USERS = "/user/hadoop/cleaned/users"
CLEANED_MOVIES = "/user/hadoop/cleaned/movies"

DATA_VERSION = "cleaned-v1.0"

ARTIFACTS = "/opt/ml-artifacts"
REGISTRY_PATH = os.path.join(ARTIFACTS, "registry.json")
STAGE_DIR = "/tmp/ml-data"

# 训练期内评分低于该阈值 -> 标记"历史不足"
LOW_HISTORY_THRESHOLD = 5

# 各算法版本号（训练成功后写入注册表，供 Agent 工具与迭代三引用）
VERSIONS = {
    "profile": "profile-v1.0",
    "class_lr": "class-lr-v1.0",
    "class_dt": "class-dt-v1.0",
    "clust_kmeans": "clust-kmeans-v1.0",
    "clust_dbscan": "clust-dbscan-v1.0",
    "dim_pca": "dim-pca-v1.0",
    "dim_svd": "dim-svd-v1.0",
}


def hdfs_exists(path):
    """判断 HDFS 路径是否存在（目录）。"""
    p = subprocess.run([HDFS_BIN, "dfs", "-test", "-d", path],
                       capture_output=True, text=True)
    return p.returncode == 0


def load_registry():
    """读取注册表；不存在或损坏时返回 None。"""
    try:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_registry(reg):
    """写回注册表（覆盖）。"""
    os.makedirs(ARTIFACTS, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def check_prereqs(reg=None):
    """
    查询工具前置校验：
      1. HDFS 存在迭代一清洗后数据（cleaned-v1.0）；
      2. 注册表存在且 data_version 与 cleaned-v1.0 一致。
    返回 (ok, message)。不满足时查询工具应返回明确状态，不得生成占位结果。
    """
    missing = []
    if not hdfs_exists(CLEANED_RATINGS):
        missing.append("ratings")
    if not hdfs_exists(CLEANED_USERS):
        missing.append("users")
    if not hdfs_exists(CLEANED_MOVIES):
        missing.append("movies")
    if missing:
        return False, (
            "前置数据缺失：HDFS 上不存在清洗后数据 {}（期望版本 {}），"
            "请先运行迭代一清洗流程生成 cleaned 数据。".format(
                "、".join(missing), DATA_VERSION))
    if reg is None:
        reg = load_registry()
    if reg is None:
        return False, ("模型注册表不存在：机器学习流水线尚未完成训练与登记，"
                       "请先运行 scripts/train-ml.ps1 完成预训练。")
    meta = reg.get("meta") or {}
    if meta.get("data_version") != DATA_VERSION:
        return False, ("数据版本不一致：注册表 data_version={}，"
                       "期望 {}，存在混用不同数据版本的风险。".format(
                           meta.get("data_version"), DATA_VERSION))
    return True, "ok"


def fail(message):
    """构造查询工具的失败返回（统一状态格式）。"""
    return {"status": "failed", "message": message}


def ok(payload):
    """构造查询工具的成功返回。"""
    out = {"status": "completed"}
    out.update(payload)
    return out
