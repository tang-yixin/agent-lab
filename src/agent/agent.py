# -*- coding: utf-8 -*-
"""
Agent 核心：基于 DeepSeek（OpenAI 兼容接口）的 function calling 编排。

Agent 理解用户自然语言意图，决定调用哪些 Hadoop/机器学习工具、按什么顺序，
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
from ml_tools import (
    classify_predict_logreg,
    classify_predict_dtree,
    cluster_query_kmeans,
    cluster_query_dbscan,
    reduce_query_pca,
    reduce_query_svd,
    ml_registry_status,
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
    # ================= 迭代二：分类 =================
    {
        "type": "function",
        "function": {
            "name": "classify_predict_logreg",
            "description": (
                "用逻辑回归模型预测：用户 UserID 对电影 MovieID 给出 ≥4 分评分的概率。"
                "返回预测类别、正类概率、模型版本、参数与评价指标（PR-AUC/F1）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "目标 UserID"},
                    "movie_id": {"type": "integer", "description": "目标 MovieID"},
                    "at_time": {
                        "type": "string",
                        "enum": ["T1", "T2"],
                        "description": "预测时点（默认 T2）。特征统计量均为训练期(≤T1)登记口径。",
                    },
                },
                "required": ["user_id", "movie_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_predict_dtree",
            "description": (
                "用决策树模型预测：用户 UserID 对电影 MovieID 给出 ≥4 分评分的概率。"
                "返回预测类别、正类概率、模型版本、参数与评价指标（PR-AUC/F1）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "目标 UserID"},
                    "movie_id": {"type": "integer", "description": "目标 MovieID"},
                    "at_time": {
                        "type": "string",
                        "enum": ["T1", "T2"],
                        "description": "预测时点（默认 T2）。特征统计量均为训练期(≤T1)登记口径。",
                    },
                },
                "required": ["user_id", "movie_id"],
            },
        },
    },
    # ================= 迭代二：聚类 =================
    {
        "type": "function",
        "function": {
            "name": "cluster_query_kmeans",
            "description": (
                "KMeans 用户群体查询：返回目标用户（或用户集合）的群体归属、"
                "群体摘要（人数/平均评分/类型偏好/代表电影/人口属性）、"
                "模型版本、参数与评价指标。users 支持单个 ID、逗号列表或区间，如 42 / 1,2,3 / 1-100。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "users": {
                        "type": "string",
                        "description": "目标用户集合，如 \"42\"、\"1,2,3\" 或 \"1-100\"",
                    },
                    "k": {
                        "type": "integer",
                        "description": "指定已登记的簇数 k（5/6/7/8 之一）；未登记值会返回明确状态，不现场训练",
                    },
                },
                "required": ["users"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cluster_query_dbscan",
            "description": (
                "DBSCAN 用户群体查询：返回目标用户（或用户集合）的群体归属（含噪声点标记）、"
                "群体摘要、模型版本、参数与评价指标。users 支持单个 ID、逗号列表或区间，如 42 / 1,2,3 / 1-100。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "users": {
                        "type": "string",
                        "description": "目标用户集合，如 \"42\"、\"1,2,3\" 或 \"1-100\"",
                    },
                    "eps": {
                        "type": "number",
                        "description": "指定已登记的 eps；未登记值会返回明确状态，不现场训练",
                    },
                    "min_samples": {
                        "type": "integer",
                        "description": "指定已登记的 min_samples；未登记值会返回明确状态，不现场训练",
                    },
                },
                "required": ["users"],
            },
        },
    },
    # ================= 迭代二：降维 =================
    {
        "type": "function",
        "function": {
            "name": "reduce_query_pca",
            "description": (
                "PCA 降维查询：返回信息保留（方差解释率/重构误差）、资源开销（耗时/内存/存储）、"
                "降维前后聚类对照（原始画像 vs 降维坐标上的 KMeans/DBSCAN 指标）"
                "与主要变化方向（各主成分高载荷电影）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dim": {
                        "type": "integer",
                        "description": "目标维数（已登记 2/10/20/50 之一，默认 50）；未登记值会返回明确状态",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reduce_query_svd",
            "description": (
                "SVD 降维查询：返回信息保留（方差解释率/重构误差）、资源开销（耗时/内存/存储）、"
                "降维前后聚类对照（原始画像 vs 降维坐标上的 KMeans/DBSCAN 指标）"
                "与主要变化方向（各奇异向量高载荷电影）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dim": {
                        "type": "integer",
                        "description": "目标维数（已登记 2/10/20/50 之一，默认 50）；未登记值会返回明确状态",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ml_registry_status",
            "description": (
                "查询机器学习注册表：已登记的分类/聚类/降维模型、版本、参数与评价指标。"
                "用于回答\"有哪些模型/是否已训练\"类问题或校验前置状态。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# 工具名 -> 可调用函数
TOOL_FUNCS = {
    "score_data": score_data,
    "clean_data": clean_data,
    "read_quarantine_samples": read_quarantine_samples,
    "classify_predict_logreg": classify_predict_logreg,
    "classify_predict_dtree": classify_predict_dtree,
    "cluster_query_kmeans": cluster_query_kmeans,
    "cluster_query_dbscan": cluster_query_dbscan,
    "reduce_query_pca": reduce_query_pca,
    "reduce_query_svd": reduce_query_svd,
    "ml_registry_status": ml_registry_status,
}


class Agent:
    """MovieLens 数据治理 Agent。"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def run(self, user_message: str, history: List[Dict[str, str]] = None,
            max_turns: int = 8) -> Dict[str, Any]:
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


SYSTEM_PROMPT = """你是 MovieLens 数据分析 Agent。系统分两部分能力：

【一、数据治理（迭代一）】
工具：score_data、clean_data、read_quarantine_samples
标准流程（用户要求"清洗并评估"时）：
1. score_data 对 /user/hadoop/input 评分 -> /user/hadoop/score_before
2. clean_data 清洗 /user/hadoop/input -> /user/hadoop/cleaned
3. score_data 对 /user/hadoop/cleaned 评分 -> /user/hadoop/score_after
4. 可选 read_quarantine_samples 读取隔离样例
5. 汇总对比并解释

【二、机器学习分析（迭代二，基于迭代一清洗数据 cleaned-v1.0 与时间边界 T1=2000-12-31、T2=2001-12-31）】
三类任务，每类两种已登记算法：

1) 分类（预测用户对电影是否给出 ≥4 分）：
   classify_predict_logreg(user_id, movie_id, at_time)、classify_predict_dtree(...)
   - 用户给出 UserID/MovieID；用户指定某算法时只调该算法，未指定时两个都调并做对照。
   - at_time 默认 T2；统计特征均为训练期(≤T1)登记口径。
   - 输出要点：预测类别、正类概率、模型版本、参数、评价指标(PR-AUC/F1)、
     以及"历史不足/冷启动"等说明。

2) 聚类（用户群体归属）：
   cluster_query_kmeans(users, k)、cluster_query_dbscan(users, eps, min_samples)
   - users 为单个 ID、逗号列表或区间（如 42 / 1,2,3 / 1-100）。
   - 用户未指定算法时两个都调并做对照；未登记的参数会得到明确状态，直接转述，不要补造结果。
   - 输出要点：群体归属、群体摘要（人数/平均评分/类型偏好/代表电影/人口属性）、
     噪声点标记(DBSCAN 的 cluster=-1)、历史不足提示、版本、参数、指标。

3) 降维（用户偏好画像压缩）：
   reduce_query_pca(dim)、reduce_query_svd(dim)
   - dim 为已登记维数 2/10/20/50 之一，默认 50；未登记维数会返回明确状态。
   - 用户未指定算法时两个都调并做对照。
   - 输出要点：信息保留（方差解释率/重构误差）、资源开销（耗时/内存/存储）、
     降维前后聚类对照（KMeans/DBSCAN 在原始画像与降维坐标上的指标）、
     主要变化方向（高载荷电影）。二维散点图坐标由前端页面自动展示。

其他工具：ml_registry_status 查询已登记模型清单（回答"有哪些模型/是否已训练"）。

必须遵守：
- 所有数字必须来自工具真实返回值，绝不编造、估算或使用示例数据。
- 工具返回 failed/unavailable 状态时，如实转述失败原因（如"模型未登记""前置数据缺失"
  "指定参数无已登记模型"），不得生成占位结果。
- 只使用已登记的实际结果；系统不现场训练模型。
- 迭代一规则同样适用：区分修复/去重/隔离；说明评价局限（如格式正确不代表内容真实、
  聚类为统计分组不一定是真实社交群体、降维信息有损等）。
- 回答用中文，条理清晰；对照两个算法时给出实际差异与依据。"""
