#!/bin/bash
# 单节点伪分布式 Hadoop 启动脚本（在容器内执行）
set -e

export HADOOP_HOME=/opt/hadoop
export JAVA_HOME=/usr/lib/jvm/jre

echo "=== 0. 准备 /data 目录权限 ==="
sudo mkdir -p /data/hdfs/name /data/hdfs/data
sudo chown -R hadoop:users /data/hdfs
sudo chmod -R 755 /data/hdfs

echo "=== 1. 检查并格式化 NameNode（首次） ==="
if [ ! -d /data/hdfs/name/current ]; then
  echo "NameNode 元数据目录为空，执行格式化..."
  $HADOOP_HOME/bin/hdfs namenode -format -force -nonInteractive
else
  echo "NameNode 已格式化，跳过。"
fi

echo "=== 2. 启动 HDFS（NameNode + DataNode） ==="
$HADOOP_HOME/bin/hdfs --daemon start namenode
$HADOOP_HOME/bin/hdfs --daemon start datanode

echo "=== 3. 启动 YARN（ResourceManager + NodeManager） ==="
$HADOOP_HOME/bin/yarn --daemon start resourcemanager
$HADOOP_HOME/bin/yarn --daemon start nodemanager

echo "=== 4. 创建 HDFS 工作目录 ==="
$HADOOP_HOME/bin/hdfs dfs -mkdir -p /user/hadoop/input /user/hadoop/output

echo "=== Hadoop 启动完成 ==="
echo "NameNode Web UI: http://localhost:9870"
echo "ResourceManager Web UI: http://localhost:8088"

# 保持容器运行
tail -f /dev/null
