# 迭代二：在 Hadoop 容器镜像基础上加入 Python + 机器学习运行时
# 说明：
#   - apache/hadoop:3.4.1 基于 CentOS 7.6，自带 yum 但无 Python 3；
#   - 使用 Miniconda（Python 3.10，自带 OpenSSL，与 CentOS 7 的 glibc 2.17 兼容）
#     安装 numpy/pandas/scipy/scikit-learn 等，避免 EPEL 老版本 Python 3.6 的限制；
#   - 不改动 Hadoop 本体，entrypoint 仍为 start-hadoop.sh。
FROM apache/hadoop:3.4.1

# 基础镜像默认 USER 为 hadoop，无法写入 /opt；安装阶段切回 root
USER root

ENV MINICONDA_DIR=/opt/miniconda3 \
    MINICONDA_VER=py310_23.11.0-2

# 安装 Miniconda（静默安装到 /opt/miniconda3）
RUN curl -fsSL -o /tmp/miniconda.sh \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VER}-Linux-x86_64.sh" \
    && bash /tmp/miniconda.sh -b -p ${MINICONDA_DIR} \
    && rm -f /tmp/miniconda.sh \
    && ${MINICONDA_DIR}/bin/pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.3 \
        scipy==1.11.4 \
        scikit-learn==1.4.2 \
        joblib==1.4.2 \
        matplotlib==3.8.4 \
        pillow==9.5.0

ENV PATH=${MINICONDA_DIR}/bin:$PATH
ENV MPLBACKEND=Agg

# 机器学习脚本与产物目录（由 docker-compose 挂载）
RUN mkdir -p /opt/ml /opt/ml-artifacts

# 恢复默认用户，Hadoop 守护进程仍以 hadoop 用户运行（与迭代一行为一致）；
# 宿主机执行 ml 脚本时统一用 docker exec -u root，保证可写 /opt/ml-artifacts
USER hadoop

CMD ["/bin/bash", "/opt/start-hadoop.sh"]
