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
import sys
import uuid
from datetime import datetime

# 使 backend 能导入 agent 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional

from agent import Agent
import hadoop_tools

# 加载配置（优先 config/secrets.env）
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "secrets.env")
load_dotenv(_ENV_PATH)

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

app = FastAPI(title="MovieLens 数据治理 Agent")

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


# 托管前端静态页面（放在路由之后，作为兜底）
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
