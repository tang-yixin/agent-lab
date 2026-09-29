# -*- coding: utf-8 -*-
"""
Hadoop 工具封装层：将 MapReduce 作业封装为 Agent 可调用的工具。

每个工具函数内部通过 subprocess 调用 `docker exec` 触发容器内的
`hadoop jar` 作业，并从 HDFS 读回结构化结果（JSON）。

Agent（DeepSeek function calling）通过调用这些函数完成任务，
不直接接触 Hadoop，保证"清洗与评分实际通过 Hadoop 执行"。
"""
import json
import subprocess
from typing import Any, Dict

# ---- HDFS 路径约定（与 docker-compose 挂载、启动脚本一致）----
CONTAINER = "lab2-hadoop"
HADOOP_BIN = "/opt/hadoop/bin/hadoop"
HDFS_BIN = "/opt/hadoop/bin/hdfs"
JAR = "/opt/hadoop-jobs.jar"

RAW_INPUT = "/user/hadoop/input"
USERS_REF = "/user/hadoop/input/users.dat"
MOVIES_REF = "/user/hadoop/input/movies.dat"

SCORE_BEFORE = "/user/hadoop/score_before"
CLEANED = "/user/hadoop/cleaned"
SCORE_AFTER = "/user/hadoop/score_after"


def _run(cmd: str, timeout: int = 600) -> Dict[str, Any]:
    """执行一条 shell 命令，返回 {returncode, stdout, stderr}。"""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "命令超时"}


def _hdfs_cat(path: str) -> str:
    """读取 HDFS 上某个路径的文本内容（文件直接读，目录读其下所有文件）。"""
    # 先判断是文件还是目录
    test_cmd = f'docker exec {CONTAINER} sh -c "{HDFS_BIN} dfs -test -d {path}"'
    is_dir = _run(test_cmd)["returncode"] == 0
    cat_path = f"{path}/*" if is_dir else path
    cmd = f'docker exec {CONTAINER} sh -c "{HDFS_BIN} dfs -cat {cat_path} 2>/dev/null"'
    res = _run(cmd)
    return res["stdout"]


def _read_json_from_hdfs(path: str) -> Dict[str, Any]:
    """从 HDFS 读回 JSON 文件（如 report.json / summary.json）。"""
    text = _hdfs_cat(path).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}


def _run_hadoop_job(main_class: str, input_path: str, output_path: str) -> Dict[str, Any]:
    """触发一个 MapReduce 作业，返回 {success, stdout, stderr}。"""
    cmd = (
        f'docker exec {CONTAINER} sh -c '
        f'"{HADOOP_BIN} jar {JAR} {main_class} {input_path} {output_path} '
        f'{USERS_REF} {MOVIES_REF}"'
    )
    res = _run(cmd)
    return {
        "success": res["returncode"] == 0,
        "returncode": res["returncode"],
        "stdout": res["stdout"],
        "stderr": res["stderr"],
    }


def score_data(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    对指定路径的数据执行五维质量评分。
    :param input_path: HDFS 输入路径（如 /user/hadoop/input）
    :param output_path: HDFS 输出路径（如 /user/hadoop/score_before）
    :return: 评分报告 dict，含 scores / counts
    """
    result = _run_hadoop_job("lab2.ScoreJob", input_path, output_path)
    if not result["success"]:
        return {
            "status": "failed",
            "message": "评分作业执行失败",
            "stderr": result["stderr"][-2000:],
        }
    report = _read_json_from_hdfs(f"{output_path}/report.json")
    if not report:
        return {"status": "failed", "message": "未能读取评分报告"}
    report["status"] = "completed"
    report["input"] = input_path
    report["output"] = output_path
    return report


def clean_data(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    对指定路径的数据执行清洗（修复/去重/隔离）。
    :param input_path: HDFS 输入路径
    :param output_path: HDFS 输出路径（如 /user/hadoop/cleaned）
    :return: 清洗统计 dict，含 fixed / dedup / quarantine / conflict
    """
    result = _run_hadoop_job("lab2.CleanJob", input_path, output_path)
    if not result["success"]:
        return {
            "status": "failed",
            "message": "清洗作业执行失败",
            "stderr": result["stderr"][-2000:],
        }
    summary = _read_json_from_hdfs(f"{output_path}/summary.json")
    if not summary:
        return {"status": "failed", "message": "未能读取清洗统计"}
    summary["status"] = "completed"
    summary["input"] = input_path
    summary["output"] = output_path
    return summary


def read_quarantine_samples(output_path: str, limit: int = 10) -> list:
    """读取隔离记录样例，用于 Agent 展示代表性异常。"""
    text = _hdfs_cat(f"{output_path}/quarantine")
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[:limit]


def read_clean_samples(output_path: str, limit: int = 10) -> list:
    """读取清洗后数据样例（按类型各取若干）。"""
    samples = []
    for sub in ["ratings", "users", "movies"]:
        text = _hdfs_cat(f"{output_path}/{sub}")
        lines = [l for l in text.splitlines() if l.strip()]
        samples.extend(lines[: max(1, limit // 3)])
    return samples[:limit]


def read_clean_counts(output_path: str) -> Dict[str, int]:
    """统计清洗后各类型记录数。"""
    counts = {}
    for sub in ["ratings", "users", "movies"]:
        text = _hdfs_cat(f"{output_path}/{sub}")
        counts[sub] = len([l for l in text.splitlines() if l.strip()])
    counts["total"] = sum(counts.values())
    return counts


def build_report(score_before_path: str, cleaned_path: str,
                 score_after_path: str) -> Dict[str, Any]:
    """
    组合生成完整评估报告，包含：
    - 清洗前/后五维评分
    - 清洗统计（修复/去重/隔离/冲突）
    - 清洗后各类型记录数
    - 版本信息与 T1/T2
    用于前端"结果获取"与报告下载。
    """
    before = _read_json_from_hdfs(f"{score_before_path}/report.json")
    after = _read_json_from_hdfs(f"{score_after_path}/report.json")
    summary = _read_json_from_hdfs(f"{cleaned_path}/summary.json")
    counts = read_clean_counts(cleaned_path)

    # 版本信息（与 docs 定稿一致）
    version = {
        "raw": "raw-v1.0",
        "cleaned": "cleaned-v1.0",
        "rules": "rules-v1.0",
        "scoring": "scoring-v1.0",
        "timeboundary": "timeboundary-v1.0",
    }
    time_boundary = {
        "T1": "2000-12-31 23:59:59 UTC",
        "T2": "2001-12-31 23:59:59 UTC",
        "T1_timestamp": 978307199,
        "T2_timestamp": 1009843199,
    }

    return {
        "version": version,
        "time_boundary": time_boundary,
        "score_before": before.get("scores", {}),
        "score_after": after.get("scores", {}),
        "counts_before": before.get("counts", {}),
        "counts_after": after.get("counts", {}),
        "clean_summary": summary,
        "clean_counts": counts,
    }
