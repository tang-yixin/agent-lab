# -*- coding: utf-8 -*-
"""
从 HDFS 拉取迭代一清洗后数据（cleaned-v1.0）到本地临时目录。

由 train_all.py 调用；单独运行可调试：
  /opt/miniconda3/bin/python /opt/ml/fetch_data.py
"""
import json
import os
import subprocess

from common import HDFS_BIN, CLEANED_RATINGS, STAGE_DIR, DATA_VERSION, \
    hdfs_exists, fail, ok


def _getmerge(src_dir, dst_file):
    p = subprocess.run([HDFS_BIN, "dfs", "-getmerge", src_dir, dst_file],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("getmerge 失败 {}: {}".format(src_dir, p.stderr[-500:]))


def detect_encoding(path):
    """判断清洗后文本编码：优先 UTF-8，含替换符则回退 ISO-8859-1。"""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
        if "�" not in text:
            return "utf-8"
    except UnicodeDecodeError:
        pass
    return "iso-8859-1"


def fetch():
    if not hdfs_exists(CLEANED_RATINGS):
        return fail("HDFS 上不存在清洗后数据 {}（期望版本 {}），请先运行迭代一清洗流程。".format(
            CLEANED_RATINGS, DATA_VERSION))
    os.makedirs(STAGE_DIR, exist_ok=True)
    paths = {}
    for name in ("ratings", "users", "movies"):
        src = "/user/hadoop/cleaned/{}".format(name)
        dst = os.path.join(STAGE_DIR, "{}.csv".format(name))
        _getmerge(src, dst)
        paths[name] = dst
    encodings = {k: detect_encoding(v) for k, v in paths.items()}
    # 持久化编码探测结果，供 build_profile.py 读取
    with open(os.path.join(STAGE_DIR, "encodings.json"), "w", encoding="utf-8") as f:
        json.dump(encodings, f)
    return ok({
        "data_version": DATA_VERSION,
        "stage_dir": STAGE_DIR,
        "files": paths,
        "encodings": encodings,
    })


if __name__ == "__main__":
    import json
    print(json.dumps(fetch(), ensure_ascii=False))
