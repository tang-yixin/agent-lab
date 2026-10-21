# MovieLens 1M 数据分析 Agent 系统（迭代二）

> 项目以 MovieLens 1M 数据集为基础，分三次迭代建设一个由 Agent 驱动的大数据处理与分析系统。
> 本仓库当前为**迭代二：分类、聚类与降维分析**（复用迭代一的清洗数据 cleaned-v1.0 与时间边界 T1/T2）。

---

## 1. 项目简介

用户在简易前端输入自然语言请求，Agent 自动调用工具完成：

- **迭代一（数据治理）**：Hadoop MapReduce 执行 **数据检查 → 清洗前评分 → 清洗 → 清洗后评分 → 结果对照与解释**；
- **迭代二（机器学习分析）**：容器内 sklearn 执行 **分类预测（逻辑回归/决策树）· 用户聚类（KMeans/DBSCAN）· 降维分析（PCA/SVD）**，模型全部预训练并登记，Agent 只返回已登记的实际结果。

全程无需手动跑 Hadoop 命令、上传中间文件或逐阶段点击按钮。

---

## 2. 技术栈

| 组件 | 选型 |
| --- | --- |
| 大数据框架 | Hadoop 3.4.1（Docker 单节点伪分布式） |
| 计算引擎 | Java MapReduce（Maven 构建，迭代一清洗/评分） |
| 机器学习 | Python 3.10 + scikit-learn（封装在 Hadoop 容器镜像内，Miniconda 安装，docker exec 触发） |
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
│   ├── hadoop/          # MapReduce 作业（Maven 工程，迭代一）
│   │   ├── pom.xml
│   │   └── src/main/java/lab2/
│   │       ├── DataRules.java       # 共享规则（编码集/时间戳/T1T2/字段判定）
│   │       ├── ReferenceTables.java # 合法 ID 集合加载（跨表检查）
│   │       ├── ScoreJob.java        # 五维评分作业
│   │       └── CleanJob.java        # 数据清洗作业
│   ├── ml/              # 迭代二：容器内 sklearn 脚本（挂载进 Hadoop 容器 /opt/ml）
│   │   ├── Dockerfile        # 镜像：apache/hadoop + Miniconda + sklearn
│   │   ├── common.py         # 共享常量/注册表读写/前置校验
│   │   ├── fetch_data.py     # HDFS cleaned -> 本地 CSV
│   │   ├── build_profile.py  # 时间切分 + 画像矩阵 + 训练期统计量
│   │   ├── train_class.py    # 分类训练（LR/DT 网格 + 验证选参 + 测试评价）
│   │   ├── train_cluster.py  # 聚类训练（KMeans/DBSCAN 网格 + 群体摘要）
│   │   ├── train_dimreduce.py# 降维训练（PCA/SVD 网格 + 聚类对照 + 资源开销）
│   │   ├── train_all.py      # 一键训练入口（容器内执行）
│   │   ├── query_class.py    # 分类预测查询（Agent 工具）
│   │   ├── query_cluster.py  # 群体归属查询（Agent 工具）
│   │   ├── query_dimreduce.py# 降维查询（Agent 工具）
│   │   ├── update_registry_meta.py  # 补写注册表 meta
│   │   └── regen_summaries.py       # 从已存归属重算群体摘要
│   ├── agent/           # Agent 核心（DeepSeek function calling）
│   │   ├── __init__.py       # 包标记（空文件，勿删）
│   │   ├── hadoop_tools.py   # Hadoop 工具封装（docker exec 触发作业）
│   │   ├── ml_tools.py       # 迭代二机器学习工具封装（docker exec 触发容器内查询）
│   │   └── agent.py          # Agent 编排（function calling + 工具定义 + 系统提示词）
│   ├── backend/         # FastAPI 后端
│   │   ├── __init__.py       # 包标记（空文件，勿删）
│   │   └── main.py          # /chat /health /report /samples /reports /ml/* 端点 + 静态页面托管
│   └── frontend/        # 前端页面
│       ├── index.html       # 单页应用（导航分区：清洗/分类/聚类/降维）
│       └── vendor/          # 本地化的 Chart.js + marked（免 CDN）
├── ml-artifacts/        # 迭代二机器学习产物（训练生成，不入 git）
│   ├── registry.json    # 注册表（模型/参数/指标/版本，供 Agent 与迭代三使用）
│   ├── profiles/        # 画像矩阵 + 用户/电影表 + 训练期统计量
│   ├── models/          # 模型文件（joblib）
│   ├── coords/          # 降维坐标 + 2D 散点 JSON
│   ├── labels/          # 群体归属
│   ├── summaries/       # 群体摘要
│   ├── eval/            # 各算法评价报告 JSON
│   └── plots/           # 散点图 PNG 归档
├── config/
│   ├── hadoop/          # 4 个 Hadoop 配置（core/hdfs/yarn/mapred-site.xml）
│   ├── version.json     # 版本信息（数据/规则/评分/时间边界/迭代二模型）
│   └── secrets.env      # DeepSeek API key（本地配置，勿提交 git）
├── reports/             # 任务自动归档的报告（运行产物，不入 git）
├── docs/                # 文档（系统说明与交付文档、迭代二方案定稿、问题日志）
├── scripts/
│   ├── probe_data.py     # 数据探查
│   ├── probe_samples.py  # 脏数据抽样
│   ├── probe_times.py    # 时间戳分布（定 T1/T2 用）
│   ├── start-hadoop.sh   # Hadoop 容器启动脚本（容器内执行）
│   ├── start-backend.ps1 # 后端启动脚本（自动定位 Python/项目目录）
│   └── train-ml.ps1      # 迭代二一键预训练（容器内执行训练流水线）
├── docker-compose.yml   # Hadoop 容器编排（构建 lab2-hadoop-ml:1.0 镜像）
├── requirements.txt     # Python 依赖清单
├── .gitignore           # 忽略密钥/构建产物/虚拟环境/运行产物
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

# 5. 启动 Hadoop（首次会自动构建 lab2-hadoop-ml:1.0 镜像 = apache/hadoop:3.4.1 + Miniconda/sklearn，约 5-10 分钟）
docker-compose up -d

# 6. 上传数据到 HDFS（首次）
docker exec lab2-hadoop sh -c "/opt/hadoop/bin/hdfs dfs -put /data/raw/ratings.dat /data/raw/users.dat /data/raw/movies.dat /user/hadoop/input/"

# 7. 启动后端
powershell -ExecutionPolicy Bypass -File scripts\start-backend.ps1

# 8. 浏览器打开 http://127.0.0.1:8000/，先跑一次清洗评估（生成 cleaned-v1.0）

# 9.（迭代二）预训练机器学习模型：生成 ml-artifacts/ 注册表与全部模型/坐标/指标
powershell -ExecutionPolicy Bypass -File scripts\train-ml.ps1
```

> 说明：
> - 第 2 步创建 venv 是**推荐**做法，依赖隔离更干净；若你机器上 Python 环境简单，也可跳过 venv，直接 `pip install -r requirements.txt`，启动脚本会自动回退到系统 python。
> - 第 4 步生成的 jar 会被 docker-compose 挂载进容器；若不编译，容器里没有 jar，运行作业会失败。
> - 第 8 步：迭代二训练与查询依赖 HDFS 上的 `cleaned-v1.0` 清洗数据，先在前端跑一次清洗评估即可生成。
> - 第 9 步的训练产物 `ml-artifacts/` 已在 `.gitignore` 中（不入 git），换机器后需重跑预训练。
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

然后浏览器打开 http://127.0.0.1:8000/。训练产物在 `ml-artifacts/`（宿主机目录），
容器重启不影响，**无需重训**。

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
> 自动调用 Hadoop 作业（评分→清洗→评分）与机器学习查询工具，全程无需手动干预。
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

### 4.6 迭代二：机器学习预训练与查询

```powershell
# 1) 一键预训练（约 5-15 分钟；前置：HDFS 已有 cleaned-v1.0 数据）
powershell -ExecutionPolicy Bypass -File scripts\train-ml.ps1

# 2) 训练产物（模型/坐标/指标/注册表）落在项目根目录 ml-artifacts/
# 3) 前端 http://127.0.0.1:8000/ 顶部导航切换到「分类预测 / 用户聚类 / 降维分析」
#    即可用自然语言查询；未训练时查询会返回明确"未登记/数据不可用"状态。
```

> 训练流水线五步：HDFS 拉取 cleaned 数据 → 时间切分(T1/T2)与画像构建 →
> 分类训练(LR/DT 网格) → 聚类训练(KMeans/DBSCAN 网格) → 降维训练(PCA/SVD 网格)。
> 分类统计特征只在训练期(≤T1)计算；验证期用于选参；测试期只做最终评价。

---

## 5. 迭代二功能与演示

### 5.1 前端四分区

页面顶部导航切换四个分区，各分区独立对话历史（追问不跨区串扰）：

| 分区 | 功能 | 结果展示 |
| --- | --- | --- |
| 🧹 数据清洗 | 迭代一清洗与五维评估 | 雷达图/柱状图、数据量变化、样例与报告下载 |
| 🎯 分类预测 | 预测 (UserID, MovieID) 是否 ≥4 分 | 两算法预测卡（类别/概率/版本/参数/指标/局限标注）+ 实验对照表 |
| 👥 用户聚类 | 用户或集合的群体归属 | 归属表（噪声点/历史不足标记）+ 群体摘要（人数/均分/类型偏好/代表电影/人口属性）+ 参数网格对照 |
| 📉 降维分析 | 画像压缩与降维前后对照 | 信息保留、资源开销、聚类对照表、主要变化方向、2D 散点图 |

### 5.2 Agent 工具（7 个机器学习工具）

| 工具 | 参数 | 说明 |
| --- | --- | --- |
| classify_predict_logreg | user_id, movie_id, at_time(T1/T2) | 逻辑回归预测 ≥4 分概率 |
| classify_predict_dtree | 同上 | 决策树预测 ≥4 分概率 |
| cluster_query_kmeans | users（"42"/"1,2,3"/"1-100"）, k(可选 5-8) | KMeans 群体归属与摘要 |
| cluster_query_dbscan | users, eps/min_samples(可选,已登记网格内) | DBSCAN 群体归属 + 噪声点标记 |
| reduce_query_pca | dim(可选 2/10/20/50) | PCA 信息保留/资源开销/聚类对照/变化方向 |
| reduce_query_svd | 同上 | SVD 同上 |
| ml_registry_status | 无 | 已登记模型清单 |

未指定算法时 Agent 自动调用同类两个工具做对照；未登记参数/未知 ID/模型缺失
均返回明确状态，**不现场训练、不生成占位结果**。

### 5.3 演示提示词（可直接粘贴）

| 场景 | 自然语言输入 |
| --- | --- |
| 分类对照 | `用逻辑回归和决策树对照，预测用户 42 对电影 200 是否会给出不低于 4 分的评分` |
| 仅单算法 | `用决策树预测用户 42 对电影 200 的高分概率` |
| 聚类对照 | `用 KMeans 和 DBSCAN 对照，查询用户 42 的群体归属` |
| 用户集合 | `查询用户集合 1-50 的群体归属，用两种聚类算法对照` |
| 噪声点识别 | `用 DBSCAN 查询用户 79 的群体归属，标记是否为噪声点`（79 为真实噪声点） |
| 历史不足 | `用 KMeans 查询用户 3038 的群体归属`（训练期评分 <5，有历史不足提示） |
| 降维对照 | `用 PCA 和 SVD 对照，压缩用户偏好画像到 50 维，比较信息保留情况、降维前后的聚类质量和资源开销` |
| 二维可视化 | `用 PCA 压缩用户画像到 2 维，并说明二维散点图` |
| 异常路径 | `预测用户 99999 对电影 200 的高分概率` → 未知 ID 状态；`用 KMeans k=12 查询用户 42` → 未登记参数状态 |

### 5.4 模型与版本

版本号存 `config/version.json`：profile-v1.0、class-lr-v1.0、class-dt-v1.0、
clust-kmeans-v1.0、clust-dbscan-v1.0、dim-pca-v1.0、dim-svd-v1.0。
训练产物与指标登记在 `ml-artifacts/registry.json`，每次 `/chat` 涉及机器学习分析时
注册表快照自动归档进 `reports/`，供迭代三（知识图谱）复用。

---

## 6. 五维评分方案（scoring-v1.0）

每维 0–100 分，清洗前后同一口径：

| 维度 | 公式 | 权重 |
| --- | --- | --- |
| Accurate | 合法记录数 / 总记录数 | 0.25 |
| Complete | 非空字段数 / 应存在字段总数 | 0.25 |
| Unique | 1 − 重复记录数 / 总记录数 | 0.20 |
| Up-to-date | 合法时间戳记录 / 应有时效性记录 | 0.10 |
| Consistent | 一致记录数 / 总记录数 | 0.20 |

综合分 = 加权平均。详见 `docs/系统说明与交付文档.md` 第 5 章。

---

## 7. 时间边界 T1/T2（timeboundary-v1.0）

| 边界 | 取值 | 用途 |
| --- | --- | --- |
| T1 | 2000-12-31 23:59:59 UTC | 训练期截止（迭代二：画像/统计特征/模型训练只用 ≤T1） |
| T2 | 2001-12-31 23:59:59 UTC | 验证期截止（迭代二：算法选择与超参调整） |

T2 之后为测试期（只做最终评价），后续迭代不得混用。详见
`docs/系统说明与交付文档.md` 第 4.3 节与 `docs/迭代二_方案定稿.md` 第 1.3 节。

---

## 8. 进度跟踪

### ✅ 迭代一（已完成）
- [x] 阶段 0：数据探查（发现人为注入的脏数据，见问题日志）
- [x] 阶段 1：五维评分方案 + 清洗规则定稿
- [x] 阶段 2：Hadoop Docker 环境搭建
- [x] 阶段 3：MapReduce 作业（评分/清洗）开发并跑通
- [x] 阶段 4：Agent（DeepSeek function calling）实现并端到端联调
- [x] 阶段 5：FastAPI 后端（/chat /health + 静态托管）
- [x] 阶段 6：前端页面（输入/状态/雷达图/统计/解释/追问/结果获取/历史报告）
- [x] 阶段 7：T1/T2 与版本管理（含任务 ID、报告自动归档、版本配置化）
- [x] 阶段 8：最终文档（`docs/系统说明与交付文档.md`）—— 系统说明/运行方法/工具接口/版本/配置/结果/限制

### ✅ 迭代二（已完成）
- [x] 阶段 9：容器镜像（apache/hadoop + Miniconda + sklearn）+ compose 改造
- [x] 阶段 10：画像与数据层（时间切分、画像矩阵、训练期统计量、注册表）
- [x] 阶段 11：分类训练流水线（LR/DT 网格 + 验证选参 + 测试评价）
- [x] 阶段 12：聚类训练流水线（KMeans/DBSCAN 网格 + 群体摘要 + 噪声点）
- [x] 阶段 13：降维训练流水线（PCA/SVD 网格 + 信息保留 + 资源开销 + 聚类对照）
- [x] 阶段 14：Agent 工具（7 个 ml 工具）+ 系统提示词 + 后端 /ml/* 端点
- [x] 阶段 15：前端四分区（分类/聚类/降维结果卡、实验对照、2D 散点图）
- [x] 阶段 16：端到端真实运行（分类/聚类/降维对话流 + 异常路径验证）
- [x] 阶段 17：迭代二文档（方案定稿/交付文档更新/问题日志）

---

## 9. 迭代一真实运行结果（2026-09-28）

| 维度 | 清洗前 | 清洗后 |
| --- | --- | --- |
| Accurate | 94.12 | 100.0 |
| Complete | 97.65 | 100.0 |
| Unique | 92.04 | 100.0 |
| Up-to-date | 96.32 | 100.0 |
| Consistent | 96.43 | 100.0 |
| **Overall** | **95.27** | **100.0** |

数据量：清洗前 1,161,652 = 清洗后 1,005,855 + 隔离 155,503 + 去重 294。

---

## 10. 迭代二真实运行结果（2026-10-19）

数据切分：训练 901,409 / 验证 67,310 / 测试 27,394 条评分事件；画像 6,004 用户 × 3,678 电影。

| 任务 | 算法 | 最优参数 | 关键指标（测试集） |
| --- | --- | --- | --- |
| 分类 | 逻辑回归 | C=0.1 | PR-AUC 0.6834 · F1 0.6666 |
| 分类 | 决策树 | depth=None, leaf=50 | PR-AUC 0.6950 · F1 0.6868 |
| 聚类 | K-Means | k=5 | 验证轮廓 0.0146 · 群体 482~2095 人 |
| 聚类 | DBSCAN | eps=0.7129, ms=5 | 主簇 5,788 + 噪声 216（训练期） |
| 降维 | PCA | 50 维 | 方差解释率 40.6% · 重构误差 0.7704 · 3.25s |
| 降维 | SVD | 50 维 | 方差解释率 40.6% · 重构误差 0.7174 · 2.74s |

降维前后聚类对照：KMeans 轮廓 0.0177（原始）→ 0.2515（PCA-50）；DBSCAN 噪声 3.6% → 0%。
完整网格与解读见 `docs/迭代二_方案定稿.md`。
