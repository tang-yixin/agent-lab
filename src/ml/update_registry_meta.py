# -*- coding: utf-8 -*-
"""
补写/刷新注册表 meta（数据版本、时间边界、切分规模）。
单独重跑某个训练阶段后调用，保证前置校验可通过：
  /opt/miniconda3/bin/python /opt/ml/update_registry_meta.py
"""
import json

from common import (ARTIFACTS, DATA_VERSION, T1_TS, T2_TS, T1_LABEL, T2_LABEL,
                    VERSIONS, load_registry, save_registry, now_str)


def main():
    reg = load_registry()
    if reg is None:
        print(json.dumps({"status": "failed", "message": "注册表不存在"}))
        return
    with open("{}/profiles/profile_meta.json".format(ARTIFACTS),
              encoding="utf-8") as f:
        pm = json.load(f)
    reg["meta"] = {
        "data_version": DATA_VERSION,
        "time_boundary": {"T1": T1_LABEL, "T2": T2_LABEL,
                          "T1_timestamp": T1_TS, "T2_timestamp": T2_TS},
        "profile_version": VERSIONS["profile"],
        "split_sizes": pm["split_sizes"],
        "created_at": now_str(),
    }
    save_registry(reg)
    print(json.dumps({"status": "completed", "meta": reg["meta"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
