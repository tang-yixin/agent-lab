# -*- coding: utf-8 -*-
"""
Agent 核心：基于 DeepSeek（OpenAI 兼容接口）的 function calling 编排。

Agent 理解用户自然语言意图，决定调用哪些 Hadoop 工具、按什么顺序，
最终把工具返回的真实结果组织成自然语言回答。

不编造分数、不生成占位结果：所有数字都来自工具的真实返回值。
"""
import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from hadoop_tools import (
    score_data,
    clean_data,
    read_quarantine_samples,
    read_clean_samples,
    SCORE_BEFORE,
    CLEANED,
    SCORE_AFTER,
    RAW_INPUT,
)

# ---- 工具定义（供 function calling 使用）----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score_data",
            "description": (
                "对 HDFS 指定路径的数据执行五维质量评分（Accurate/Complete/Unique/"
                "Up-to-date/Consistent），返回各维度得分与计数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "HDFS 输入数据路径，如 /user/hadoop/input",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "HDFS 评分结果输出路径，如 /user/hadoop/score_before",
                    },
                },
                "required": ["input_path", "output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clean_data",
            "description": (
                "对 HDFS 指定路径的数据执行清洗（修复/去重/隔离），"
                "返回修复、去重、隔离、冲突的条数统计。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "HDFS 输入数据路径，如 /user/hadoop/input",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "HDFS 清洗结果输出路径，如 /user/hadoop/cleaned",
                    },
                },
                "required": ["input_path", "output_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_quarantine_samples",
            "description": "读取隔离记录的样例，用于解释处理了哪些异常。",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "清洗输出路径，如 /user/hadoop/cleaned",
                    },
                    "limit": {"type": "integer", "description": "样例条数，默认 10"},
                },
                "required": ["output_path"],
            },
        },
    },
]

# 工具名 -> 可调用函数
TOOL_FUNCS = {
    "score_data": score_data,
    "clean_data": clean_data,
    "read_quarantine_samples": read_quarantine_samples,
}


class Agent:
    """MovieLens 数据治理 Agent。"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def run(self, user_message: str, history: List[Dict[str, str]] = None,
            max_turns: int = 6) -> Dict[str, Any]:
        """
        处理一条用户消息，返回 {answer, steps}。
        steps 记录每个工具调用的名称、参数、返回结果，用于前端展示执行情况。
        history 为历史对话（[{role, content}]），用于支持追问。
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if history:
            for h in history:
                role = h.get("role")
                content = h.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        steps: List[Dict[str, Any]] = []

        for _ in range(max_turns):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = resp.choices[0].message
            finish = resp.choices[0].finish_reason

            # 追加 assistant 消息（含 tool_calls）
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # 若无工具调用，说明已给出最终回答
            if not msg.tool_calls:
                return {"answer": msg.content or "", "steps": steps}

            # 执行工具
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                func = TOOL_FUNCS.get(name)
                if func is None:
                    result = {"status": "error", "message": f"未知工具 {name}"}
                else:
                    try:
                        result = func(**args)
                    except Exception as e:  # noqa: BLE001
                        result = {"status": "error", "message": str(e)}

                steps.append({"tool": name, "args": args, "result": result})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # 超过最大轮次仍未结束
        return {
            "answer": "任务执行超过最大轮次，未获得最终结果。",
            "steps": steps,
        }


SYSTEM_PROMPT = """你是 MovieLens 数据治理 Agent。用户会用自然语言提出数据清洗与质量评估需求。

你有以下工具：
1. score_data(input_path, output_path)：对数据做五维质量评分
2. clean_data(input_path, output_path)：对数据做清洗（修复/去重/隔离）
3. read_quarantine_samples(output_path, limit)：读取隔离记录样例

标准数据治理流程（当用户要求"清洗并评估"时按此执行）：
1. 先用 score_data 对原始数据 /user/hadoop/input 评分，输出到 /user/hadoop/score_before
2. 再用 clean_data 清洗 /user/hadoop/input，输出到 /user/hadoop/cleaned
3. 再用 score_data 对清洗后数据 /user/hadoop/cleaned 评分，输出到 /user/hadoop/score_after
4. 可选：用 read_quarantine_samples 读取隔离样例
5. 最后汇总对比，向用户解释

必须遵守：
- 所有分数和统计必须来自工具真实返回值，绝不编造或估算。
- 工具执行失败时，如实说明失败环节，不得生成占位结果。
- 区分"修复""去重""隔离"，隔离不等于删除，不要把隔离表述为"问题已修复"。
- 说明仍未解决的问题和评价局限（如格式正确不代表内容真实、历史数据时效性受年代限制）。
- 回答用中文，条理清晰。"""
