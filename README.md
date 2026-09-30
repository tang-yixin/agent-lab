# MovieLens 1M 数据分析 Agent 系统（迭代一）

> 项目以 MovieLens 1M 数据集为基础，分三次迭代建设一个由 Agent 驱动的大数据处理与分析系统。
> 本仓库当前为**迭代一：Hadoop 数据清洗与五维质量评估**。

---

## 1. 项目简介

用户在简易前端输入自然语言请求，Agent 自动调用 Hadoop 完成：
**数据检查 → 清洗前评分 → 清洗 → 清洗后评分 → 结果对照与解释**。

全程无需手动跑 Hadoop 命令、上传中间文件或逐阶段点击按钮。

---

## 2. 技术栈

| 组件 | 选型 |
| --- | --- |
| 大数据框架 | Hadoop 3.4.1（Docker 单节点伪分布式） |
| 计算引擎 | Java MapReduce（Maven 构建） |
| Agent | DeepSeek API（function calling） |
| 后端 | FastAPI（Python） |
| 前端 | 原生 HTML/JS（由 FastAPI 托管） |
| 数据 | ml-1m（movies/ratings/users） |

---

## 3. 目录结构

```
agent-lab/
├── ml-1m/               # 原始数据（只读，含脏数据注入）
├── src/
│   ├── hadoop/          # MapReduce 作业（Maven 工程）
│   │   ├── pom.xml
│   │   └── src/main/java/lab2/
│   │       ├── DataRules.java       # 共享规则（编码集/时间戳/T1T2/字段判定）
│   │       ├── ReferenceTables.java # 合法 ID 集合加载（跨表检查）
│   │       ├── ScoreJob.java        # 五维评分作业
│   │       └── CleanJob.java        # 数据清洗作业
│   ├── agent/           # Agent 核心（DeepSeek function calling）
│   │   ├── __init__.py       # 包标记（空文件，勿删）
│   │   ├── hadoop_tools.py   # Hadoop 工具封装（docker exec 触发作业）
│   │   └── agent.py          # Agent 编排（function calling + 工具定义）
│   ├── backend/         # FastAPI 后端
│   │   ├── __init__.py       # 包标记（空文件，勿删）
│   │   └── main.py          # /chat /health /report /samples /reports 端点 + 静态页面托管
│   └── frontend/        # 前端页面
│       ├── index.html       # 单页应用（输入/状态/图表/解释/追问/结果获取/历史报告）
│       └── vendor/          # 本地化的 Chart.js + marked（免 CDN）
├── config/
│   ├── hadoop/          # 4 个 Hadoop 配置（core/hdfs/yarn/mapred-site.xml）
│   ├── version.json     # 版本信息（数据/规则/评分/时间边界）
│   └── secrets.env      # DeepSeek API key（本地配置，勿提交 git）
├── reports/             # 任务自动归档的报告（运行产物，不入 git）
├── docs/                # 文档（项目详解、演示引导、交付文档、问题日志、方案定稿）
├── scripts/
│   ├── probe_data.py     # 数据探查
│   ├── probe_samples.py  # 脏数据抽样
│   ├── probe_times.py    # 时间戳分布（定 T1/T2 用）
│   ├── start-hadoop.sh   # Hadoop 容器启动脚本（容器内执行）
│   └── start-backend.ps1 # 后端启动脚本（自动定位 Python/项目目录）
├── docker-compose.yml   # Hadoop 容器编排
├── requirements.txt     # Python 依赖清单
├── .gitignore           # 忽略密钥/构建产物/虚拟环境
└── README.md
```

---

## 4. 快速开始

### 4.0 新机器初始化（首次运行 / 换电脑）

前置要求：已安装 **Docker Desktop** 和 **Python 3.10+**（Java/Maven 仅改动 MapReduce 时才需要）。

```powershell
# 1. 克隆仓库
cd 你的工作目录
git clone <仓库地址>
cd agent-lab

# 2. 创建虚拟环境并安装依赖（可跳过：也可直接用系统 python 执行第 3 步）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. 配置 API key
copy config\secrets.env.example config\secrets.env   # 然后用编辑器填入 DEEPSEEK_API_KEY

# 4. 编译 MapReduce jar（首次必须，生成 target\hadoop-jobs.jar）
cd src\hadoop
mvn -DskipTests package
cd ..\..

# 5. 启动 Hadoop（首次会自动拉取 apache/hadoop:3.4.1 镜像）
docker-compose up -d

# 6. 上传数据到 HDFS（首次）
docker exec lab2-hadoop sh -c "/opt/hadoop/bin/hdfs dfs -put /data/raw/ratings.dat /data/raw/users.dat /data/raw/movies.dat /user/hadoop/input/"

# 7. 启动后端
powershell -ExecutionPolicy Bypass -File scripts\start-backend.ps1

# 8. 浏览器打开 http://127.0.0.1:8000/
```

> 说明：
> - 第 2 步创建 venv 是**推荐**做法，依赖隔离更干净；若你机器上 Python 环境简单，也可跳过 venv，直接 `pip install -r requirements.txt`，启动脚本会自动回退到系统 python。
> - 第 4 步生成的 jar 会被 docker-compose 挂载进容器；若不编译，容器里没有 jar，运行作业会失败。
> - `config/secrets.env` 已在 `.gitignore` 中，不会随仓库上传，需自行配置。

### 4.1 启动 Hadoop 集群

```powershell
# 1. 双击打开 Docker Desktop，等其就绪

# 2. 在项目根目录 agent-lab 执行：
docker-compose up -d

# 3. 约 20 秒后验证（应看到 NameNode/DataNode/ResourceManager/NodeManager 四个 java 进程）：
docker exec lab2-hadoop sh -c "ps -ef | grep -E 'NameNode|DataNode|ResourceManager|NodeManager' | grep -v grep"
```

- NameNode Web UI: http://localhost:9870
- ResourceManager Web UI: http://localhost:8088
- 停止：`docker-compose down`（数据在 hadoop-data 卷中，不会丢）

### 4.1.1 下次启动三步走（演示常用）

已经完整跑过一遍后，下次只需：

```powershell
# ① 打开 Docker Desktop，等就绪
# ② 项目根目录启动 Hadoop：
docker-compose up -d
# ③ 启动后端（自动定位 Python 与项目目录，无需记长命令）：
powershell -ExecutionPolicy Bypass -File scripts\start-backend.ps1
```

然后浏览器打开 http://127.0.0.1:8000/  
在输入框输入自然语言即可触发完整流程。

> 页面演示时的自然语言示例（可直接粘贴）：
>
> - **完整流程（首选）**：`请使用默认规则清洗 MovieLens 1M，评估清洗前后的 Accurate、Complete、Unique、Up-to-date、Consistent 五个维度，并说明处理了哪些问题、还有哪些问题无法解决。`
> - **简洁版**：`请清洗 MovieLens 1M 数据，并评估清洗前后的五维质量。`
> - **只评分不清洗**：`只对当前数据做五维质量评分，不要清洗。`
> - **追问（任务完成后继续问）**：`隔离了哪些类型的异常？分别有多少？` / `为什么清洗后 Unique 变成满分了？`
>
> 也可直接点输入框下方的「默认清洗评估」和「追问数据量变化」快捷按钮。


### 4.2 编译 MapReduce 作业（仅当改了 Java 代码时才需要）

```powershell
cd src\hadoop
mvn -DskipTests package          # 产出 target\hadoop-jobs.jar
```

> jar 已通过 docker-compose 挂载进容器（`/opt/hadoop-jobs.jar`），
> **无需再 `docker cp`**。改代码后重新 `mvn package` + 重启容器即可生效。

### 4.3 上传数据（首次；数据在卷里，通常无需重做）

```powershell
# 上传原始数据到 HDFS（首次）
docker exec lab2-hadoop sh -c "/opt/hadoop/bin/hdfs dfs -put /data/raw/ratings.dat /data/raw/users.dat /data/raw/movies.dat /user/hadoop/input/"
```

> 数据存在 `hadoop-data` 命名卷里，`docker-compose down` 不会丢。
> 若下次启动后数据仍在，跳过此步；验证命令：
> `docker exec lab2-hadoop sh -c "/opt/hadoop/bin/hdfs dfs -ls /user/hadoop/input/"`

### 4.4 启动系统（前端 + Agent）

```powershell
# 1. 在 config/secrets.env 中填入 DEEPSEEK_API_KEY

# 2. 启动后端（托管前端页面 + /chat 接口；脚本自动定位 Python 与项目目录）
powershell -ExecutionPolicy Bypass -File scripts\start-backend.ps1

# 3. 浏览器打开 http://127.0.0.1:8000/
```

> 前端页面输入自然语言即可触发完整流程：Agent 通过 function calling
> 自动调用 Hadoop 作业（评分→清洗→评分），全程无需手动干预。
> 若不使用启动脚本，手动命令为：`cd src\backend; python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir <项目根目录>\src\backend`。

### 4.5 手动运行单个作业（调试用）

```powershell
# 1) 清洗前评分
docker exec lab2-hadoop sh -c "/opt/hadoop/bin/hadoop jar /opt/hadoop-jobs.jar lab2.ScoreJob /user/hadoop/input /user/hadoop/score_before /user/hadoop/input/users.dat /user/hadoop/input/movies.dat"

# 2) 清洗
docker exec lab2-hadoop sh -c "/opt/hadoop/bin/hadoop jar /opt/hadoop-jobs.jar lab2.CleanJob /user/hadoop/input /user/hadoop/cleaned /user/hadoop/input/users.dat /user/hadoop/input/movies.dat"

# 3) 清洗后评分
docker exec lab2-hadoop sh -c "/opt/hadoop/bin/hadoop jar /opt/hadoop-jobs.jar lab2.ScoreJob /user/hadoop/cleaned /user/hadoop/score_after /user/hadoop/input/users.dat /user/hadoop/input/movies.dat"
```

---

## 5. 五维评分方案（scoring-v1.0）

每维 0–100 分，清洗前后同一口径：

| 维度 | 公式 | 权重 |
| --- | --- | --- |
| Accurate | 合法记录数 / 总记录数 | 0.25 |
| Complete | 非空字段数 / 应存在字段总数 | 0.25 |
| Unique | 1 − 重复记录数 / 总记录数 | 0.20 |
| Up-to-date | 合法时间戳记录 / 应有时效性记录 | 0.10 |
| Consistent | 一致记录数 / 总记录数 | 0.20 |

综合分 = 加权平均。详见 `docs/阶段1_评分方案与清洗规则.md`。

---

## 6. 时间边界 T1/T2（timeboundary-v1.0）

| 边界 | 取值 | 用途 |
| --- | --- | --- |
| T1 | 2000-12-31 23:59:59 UTC | 训练期截止 |
| T2 | 2001-12-31 23:59:59 UTC | 验证期截止 |

T2 之后为测试期，后续迭代不得混用。详见 `docs/阶段7_时间边界与版本管理.md`。

---

## 7. 进度跟踪

### ✅ 已完成
- [x] 阶段 0：数据探查（发现人为注入的脏数据，见问题日志）
- [x] 阶段 1：五维评分方案 + 清洗规则定稿
- [x] 阶段 2：Hadoop Docker 环境搭建
- [x] 阶段 3：MapReduce 作业（评分/清洗）开发并跑通
- [x] 阶段 4：Agent（DeepSeek function calling）实现并端到端联调
- [x] 阶段 5：FastAPI 后端（/chat /health + 静态托管）
- [x] 阶段 6：前端页面（输入/状态/雷达图/统计/解释/追问/结果获取/历史报告）
- [x] 阶段 7：T1/T2 与版本管理（含任务 ID、报告自动归档、版本配置化）

### ✅ 已完成
- [x] 阶段 8：最终文档（`docs/系统说明与交付文档.md`）—— 系统说明/运行方法/工具接口/版本/配置/结果/限制

---

## 8. 真实运行结果（2026-09-28）

| 维度 | 清洗前 | 清洗后 |
| --- | --- | --- |
| Accurate | 94.12 | 100.0 |
| Complete | 97.65 | 100.0 |
| Unique | 92.04 | 100.0 |
| Up-to-date | 96.32 | 100.0 |
| Consistent | 96.43 | 100.0 |
| **Overall** | **95.27** | **100.0** |

数据量：清洗前 1,161,652 = 清洗后 1,005,855 + 隔离 155,503 + 去重 294。
