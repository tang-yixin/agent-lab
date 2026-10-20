# -*- coding: utf-8 -*-
"""
迭代二机器学习工具封装层（宿主机执行）：将容器内 sklearn 查询脚本封装为
Agent 可调用的工具。

每个工具 = docker exec -u root lab2-hadoop python /opt/ml/query_*.py ...
容器内脚本校验前置条件（cleaned-v1.0 数据 + 注册表版本一致），
只返回已登记的实际结果；缺失时返回明确状态，不生成占位结果。

参数一律以 argv 列表传入（不用 shell 拼接），防止注入。
"""
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

# 项目根目录（agent -> src -> agent-lab）
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTAINER = "lab2-hadoop"
PYTHON = "/opt/miniconda3/bin/python"
ML_DIR = "/opt/ml"
ARTIFACTS_HOST = os.path.join(_BASE_DIR, "ml-artifacts")

_USERS_RE = re.compile(r"^[\d,\-\s]+$")


def _run_ml(script: str, args: List[str], timeout: int = 300) -> Dict[str, Any]:
    """在容器内执行一个 ml 查询脚本，解析 stdout JSON。"""
    cmd = ["docker", "exec", "-u", "root", CONTAINER, PYTHON,
           "{}/{}".format(ML_DIR, script)] + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        return {"status": "failed", "message": "查询超时（{}s）".format(timeout)}
    if proc.returncode != 0:
        return {"status": "failed",
                "message": "容器内脚本执行失败",
                "stderr": proc.stderr[-1500:]}
    try:
        return json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"status": "failed", "message": "工具返回不可解析",
                "stdout": proc.stdout[-800:]}


def _check_user_movie(user_id: Any, movie_id: Any) -> Optional[str]:
    if not isinstance(user_id, int) or not isinstance(movie_id, int):
        return "UserID 与 MovieID 必须为整数"
    return None


def classify_predict_logreg(user_id: int, movie_id: int,
                            at_time: str = "T2") -> Dict[str, Any]:
    """分类预测（逻辑回归）：预测该用户对该电影给出 ≥4 分的概率。"""
    err = _check_user_movie(user_id, movie_id)
    if err:
        return {"status": "failed", "message": err}
    return _run_ml("query_class.py", ["--algo", "logreg", "--user", str(user_id),
                                      "--movie", str(movie_id), "--at", at_time])


def classify_predict_dtree(user_id: int, movie_id: int,
                           at_time: str = "T2") -> Dict[str, Any]:
    """分类预测（决策树）：预测该用户对该电影给出 ≥4 分的概率。"""
    err = _check_user_movie(user_id, movie_id)
    if err:
        return {"status": "failed", "message": err}
    return _run_ml("query_class.py", ["--algo", "dtree", "--user", str(user_id),
                                      "--movie", str(movie_id), "--at", at_time])


def cluster_query_kmeans(users: str, k: Optional[int] = None) -> Dict[str, Any]:
    """KMeans 群体查询：用户或用户集合（如 "42" / "1,2,3" / "1-100"）的群体归属与摘要。"""
    if not _USERS_RE.match(users or ""):
        return {"status": "failed",
                "message": "用户集合格式非法，仅支持数字、逗号与区间，如 \"1,2,3\" 或 \"1-100\""}
    args = ["--algo", "kmeans", "--users", users]
    if k is not None:
        args += ["--k", str(k)]
    return _run_ml("query_cluster.py", args)


def cluster_query_dbscan(users: str, eps: Optional[float] = None,
                         min_samples: Optional[int] = None) -> Dict[str, Any]:
    """DBSCAN 群体查询：用户或用户集合的群体归属（含噪声点标记）与摘要。"""
    if not _USERS_RE.match(users or ""):
        return {"status": "failed",
                "message": "用户集合格式非法，仅支持数字、逗号与区间，如 \"1,2,3\" 或 \"1-100\""}
    args = ["--algo", "dbscan", "--users", users]
    if eps is not None:
        args += ["--eps", str(eps)]
    if min_samples is not None:
        args += ["--min_samples", str(min_samples)]
    return _run_ml("query_cluster.py", args)


def reduce_query_pca(dim: Optional[int] = None) -> Dict[str, Any]:
    """PCA 降维查询：信息保留、资源开销、降维前后聚类对照与主要变化方向。"""
    args = ["--algo", "pca"]
    if dim is not None:
        args += ["--dim", str(dim)]
    return _run_ml("query_dimreduce.py", args)


def reduce_query_svd(dim: Optional[int] = None) -> Dict[str, Any]:
    """SVD 降维查询：信息保留、资源开销、降维前后聚类对照与主要变化方向。"""
    args = ["--algo", "svd"]
    if dim is not None:
        args += ["--dim", str(dim)]
    return _run_ml("query_dimreduce.py", args)


def _load_host_registry() -> Optional[Dict[str, Any]]:
    path = os.path.join(ARTIFACTS_HOST, "registry.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def ml_registry_status() -> Dict[str, Any]:
    """查询机器学习注册表状态：已登记模型、版本、参数与关键指标。"""
    reg = _load_host_registry()
    if reg is None:
        return {
            "status": "unavailable",
            "message": "模型注册表不存在：机器学习流水线尚未完成训练与登记"
                       "（请先运行 scripts/train-ml.ps1）",
        }
    entries = []
    for e in reg.get("entries", []):
        entries.append({
            "artifact_id": e.get("artifact_id"),
            "task": e.get("task"),
            "algorithm": e.get("algorithm"),
            "version": e.get("version"),
            "selected_params": e.get("selected_params"),
            "metrics": e.get("metrics"),
        })
    return {
        "status": "completed",
        "meta": reg.get("meta", {}),
        "entries": entries,
    }
