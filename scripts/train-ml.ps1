$ErrorActionPreference = "Stop"

# 迭代二：一键预训练脚本（宿主机执行）
# 依次在 Hadoop 容器内运行：拉取 HDFS 清洗数据 -> 构建画像 -> 分类 -> 聚类 -> 降维 -> 注册
# 前置：docker-compose up -d 已启动容器（镜像 lab2-hadoop-ml:1.0），HDFS 有 cleaned-v1.0 数据
# 用法：powershell -ExecutionPolicy Bypass -File scripts\train-ml.ps1

Write-Host "== 迭代二机器学习预训练流水线 ==" -ForegroundColor Cyan
Write-Host "容器: lab2-hadoop | 耗时约 5-15 分钟（DBSCAN 网格最慢）" -ForegroundColor DarkGray
Write-Host ""

docker exec -u root lab2-hadoop /opt/miniconda3/bin/python /opt/ml/train_all.py

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "== 训练完成，产物在 ml-artifacts/（registry.json + models/coords/eval/summaries/plots）==" -ForegroundColor Green
} else {
    Write-Host "== 训练失败（退出码 $LASTEXITCODE），请查看上方容器内报错 ==" -ForegroundColor Red
}
