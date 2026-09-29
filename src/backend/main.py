# -*- coding: utf-8 -*-
"""
FastAPI 后端：接收前端自然语言请求，调用 Agent 完成数据治理任务。

端点：
  POST /chat          用户输入一句话，Agent 编排 Hadoop 工具并返回结果
  GET  /             前端页面（后续阶段接入）
  GET  /health       健康检查
"""
import os
import sys

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
    try:
        agent = get_agent()
        result = agent.run(req.message, history=req.history)
        return ChatResponse(answer=result["answer"], steps=result["steps"], status="completed")
    except RuntimeError as e:
        return ChatResponse(answer=str(e), steps=[], status="failed")
    except Exception as e:  # noqa: BLE001
        return ChatResponse(answer=f"执行失败：{e}", steps=[], status="failed")


# 托管前端静态页面（放在路由之后，作为兜底）
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
