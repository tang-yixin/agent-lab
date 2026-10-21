# -*- coding: utf-8 -*-
"""
FastAPI 后端：接收前端自然语言请求，调用 Agent 完成数据治理任务。

端点：
  POST /chat          用户输入一句话，Agent 编排 Hadoop 工具并返回结果
  GET  /             前端页面（后续阶段接入）
  GET  /health       健康检查
"""
import json
import os
import re
import sys
import uuid
from datetime import datetime

# 使 backend 能导入 agent 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional

from agent import Agent
import hadoop_tools
import ml_tools

# 加载配置（优先 config/secrets.env）
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "secrets.env")
load_dotenv(_ENV_PATH)

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

app = FastAPI(title="MovieLens 数据分析 Agent")

# 机器学习产物目录（与 docker-compose 挂载一致）
_ML_ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "..", "ml-artifacts")

# 涉及机器学习分析的工具名（用于判断 /chat 归档时是否附注册表快照）
_ML_TOOL_NAMES = {
    "classify_predict_logreg", "classify_predict_dtree",
    "cluster_query_kmeans", "cluster_query_dbscan",
    "reduce_query_pca", "reduce_query_svd",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 惰性初始化 Agent（首次请求时创建，避免无 key 时启动即报错）
_agent = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        if not API_KEY:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 config/secrets.env 中填写")
        _agent = Agent(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    return _agent


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    steps: list
    status: str
    task_id: Optional[str] = None


# 报告归档目录（项目根目录 reports/）
_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")


def _archive_report(task_id: str, report_data: dict) -> str:
    """将完整报告写入本地 reports/ 目录，返回文件名。"""
    os.makedirs(_REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"task-{task_id[:8]}-{ts}.json"
    payload = dict(report_data)
    payload["task_id"] = task_id
    payload["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(_REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return filename


@app.get("/health")
def health():
    return {"status": "ok", "api_key_configured": bool(API_KEY)}


@app.get("/report")
def report():
    """返回完整评估报告（清洗前后评分 + 清洗统计 + 版本 + T1/T2）。"""
    try:
        data = hadoop_tools.build_report(
            hadoop_tools.SCORE_BEFORE,
            hadoop_tools.CLEANED,
            hadoop_tools.SCORE_AFTER,
        )
        return {"status": "completed", "report": data}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "message": f"读取报告失败：{e}"}


@app.get("/samples")
def samples(limit: int = 30):
    """返回清洗后数据样例 + 各类型记录数。"""
    try:
        clean_samples = hadoop_tools.read_clean_samples(hadoop_tools.CLEANED, limit)
        quarantine_samples = hadoop_tools.read_quarantine_samples(hadoop_tools.CLEANED, 20)
        counts = hadoop_tools.read_clean_counts(hadoop_tools.CLEANED)
        return {
            "status": "completed",
            "clean_counts": counts,
            "clean_samples": clean_samples,
            "quarantine_samples": quarantine_samples,
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "message": f"读取样例失败：{e}"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    task_id = str(uuid.uuid4())
    try:
        agent = get_agent()
        result = agent.run(req.message, history=req.history)
        # 任务完成后自动归档报告（HDFS 有有效结果时才归档）
        try:
            report = hadoop_tools.build_report(
                hadoop_tools.SCORE_BEFORE,
                hadoop_tools.CLEANED,
                hadoop_tools.SCORE_AFTER,
            )
            if report.get("score_before") or report.get("clean_summary"):
                _archive_report(task_id, report)
        except Exception:
            pass  # 归档失败不影响主流程
        # 涉及机器学习分析时，把注册表快照附进归档（供追溯）
        try:
            called = {s.get("tool") for s in result["steps"]}
            if called & _ML_TOOL_NAMES:
                registry = ml_tools._load_host_registry()
                if registry:
                    _archive_report(task_id, {"ml_registry": registry})
        except Exception:
            pass
        return ChatResponse(
            answer=result["answer"],
            steps=result["steps"],
            status="completed",
            task_id=task_id,
        )
    except RuntimeError as e:
        return ChatResponse(answer=str(e), steps=[], status="failed", task_id=task_id)
    except Exception as e:  # noqa: BLE001
        return ChatResponse(answer=f"执行失败：{e}", steps=[], status="failed", task_id=task_id)


@app.get("/reports")
def list_reports():
    """列出 reports/ 目录下的历史报告文件名（倒序）。"""
    os.makedirs(_REPORTS_DIR, exist_ok=True)
    try:
        files = sorted(
            [f for f in os.listdir(_REPORTS_DIR) if f.endswith(".json")],
            reverse=True,
        )
        return {"status": "completed", "reports": files}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "message": f"读取报告列表失败：{e}"}


@app.get("/report-file/{filename}")
def get_report_file(filename: str):
    """读取单个历史报告的完整内容。"""
    if not filename.endswith(".json"):
        return {"status": "failed", "message": "非法文件名"}
    path = os.path.join(_REPORTS_DIR, filename)
    if not os.path.exists(path):
        return {"status": "failed", "message": "文件不存在"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {"status": "completed", "report": json.load(f)}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "message": f"读取报告失败：{e}"}


# ================= 迭代二：机器学习分析端点 =================

def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/ml/registry")
def ml_registry():
    """机器学习注册表：已登记模型、版本、参数与指标。"""
    reg = ml_tools._load_host_registry()
    if reg is None:
        return {"status": "unavailable",
                "message": "模型注册表不存在：机器学习流水线尚未完成训练与登记"
                           "（请先运行 scripts/train-ml.ps1）"}
    return {"status": "completed", "registry": reg}


@app.get("/ml/overview")
def ml_overview():
    """三类任务两算法对照总览（供前端实验对照卡渲染，全部来自已登记结果）。"""
    reg = ml_tools._load_host_registry()
    if reg is None:
        return {"status": "unavailable",
                "message": "模型注册表不存在：机器学习流水线尚未完成训练与登记"}
    entries = {e["artifact_id"]: e for e in reg.get("entries", [])}

    def _entry(eid):
        return entries.get(eid)

    # 分类
    classification = {}
    for eid, algo_key in (("class-lr-v1.0", "logreg"), ("class-dt-v1.0", "dtree")):
        e = _entry(eid)
        if not e:
            continue
        eval_obj = _read_json(os.path.join(_ML_ARTIFACTS, "eval",
                                           "class_{}_eval.json".format(
                                               "lr" if algo_key == "logreg" else "dt")))
        classification[algo_key] = {
            "version": e["version"],
            "params": e["selected_params"],
            "metrics": e["metrics"],
            "grid": eval_obj.get("grid") if eval_obj else None,
            "split_sizes": eval_obj.get("split_sizes") if eval_obj else None,
            "label": eval_obj.get("label") if eval_obj else None,
        }

    # 聚类
    clustering = {}
    for eid, algo_key in (("clust-kmeans-v1.0", "kmeans"),
                          ("clust-dbscan-v1.0", "dbscan")):
        e = _entry(eid)
        if not e:
            continue
        eval_obj = _read_json(os.path.join(_ML_ARTIFACTS, "eval",
                                           "{}_eval.json".format(algo_key)))
        clustering[algo_key] = {
            "version": e["version"],
            "params": e["selected_params"],
            "param_grid": e.get("param_grid"),
            "metrics": e["metrics"],
            "grid": eval_obj.get("grid") if eval_obj else None,
            "selected": eval_obj.get("selected") if eval_obj else None,
        }

    # 降维
    dimension_reduction = {}
    for eid, algo_key in (("dim-pca-v1.0", "pca"), ("dim-svd-v1.0", "svd")):
        e = _entry(eid)
        if not e:
            continue
        eval_obj = _read_json(os.path.join(_ML_ARTIFACTS, "eval",
                                           "{}_eval.json".format(algo_key)))
        variants = []
        if eval_obj:
            for v in eval_obj.get("variants", []):
                variants.append({
                    "n_components": v["n_components"],
                    "retention": v["retention"],
                    "resource": v["resource"],
                    "cluster_comparison": v["cluster_comparison"],
                })
        dimension_reduction[algo_key] = {
            "version": e["version"],
            "params": e["selected_params"],
            "dims": e.get("dim_grid"),
            "metrics": e["metrics"],
            "variants": variants,
            "principal_directions": eval_obj.get("principal_directions_top_movies")
            if eval_obj else None,
            "files_2d": eval_obj.get("files_2d") if eval_obj else None,
        }

    return {
        "status": "completed",
        "meta": reg.get("meta", {}),
        "classification": classification,
        "clustering": clustering,
        "dimension_reduction": dimension_reduction,
    }


_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
_ALLOWED_ARTIFACT_EXT = {".json": "application/json", ".png": "image/png"}


@app.get("/ml/artifacts/{name:path}")
def ml_artifact(name: str):
    """读取已登记产物文件（坐标/摘要/评价报告/散点图等），仅限 ml-artifacts 内白名单类型。"""
    if not _ARTIFACT_NAME_RE.match(name) or ".." in name.split("/"):
        return JSONResponse({"status": "failed", "message": "非法文件名"},
                            status_code=400)
    ext = os.path.splitext(name)[1].lower()
    if ext not in _ALLOWED_ARTIFACT_EXT:
        return JSONResponse(
            {"status": "failed", "message": "仅支持 .json / .png 产物文件"},
            status_code=400)
    path = os.path.abspath(os.path.join(_ML_ARTIFACTS, name))
    base = os.path.abspath(_ML_ARTIFACTS)
    if not path.startswith(base + os.sep):
        return JSONResponse({"status": "failed", "message": "路径越界"},
                            status_code=400)
    if not os.path.isfile(path):
        return JSONResponse({"status": "failed", "message": "文件不存在"},
                            status_code=404)
    return FileResponse(path, media_type=_ALLOWED_ARTIFACT_EXT[ext],
                        filename=os.path.basename(name))


# 托管前端静态页面（放在路由之后，作为兜底）
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
