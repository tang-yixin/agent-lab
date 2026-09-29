# -*- coding: utf-8 -*-
"""统计合法评分时间戳的分布，用于确定 T1/T2 时间边界。"""
import os
import collections

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml-1m")
ENC = "latin-1"

path = os.path.join(BASE, "ratings.dat")
ts_list = []
with open(path, "r", encoding=ENC) as f:
    for line in f:
        line = line.rstrip("\n")
        fields = line.split("::")
        if len(fields) != 4:
            continue
        u, m, r, t = fields
        if not (u.isdigit() and m.isdigit() and r.isdigit() and t.isdigit()):
            continue
        if not (1 <= int(r) <= 5):
            continue
        tv = int(t)
        # 秒级时间戳：2000-04 ~ 2003-02 对应约 956e6 ~ 1045e6
        if 950_000_000 <= tv <= 1_050_000_000:
            ts_list.append(tv)

ts_list.sort()
n = len(ts_list)
import datetime

def fmt(v):
    return datetime.datetime.utcfromtimestamp(v).strftime("%Y-%m-%d")

print(f"完全合法的秒级评分记录数: {n}")
print(f"最小时间: {ts_list[0]} -> {fmt(ts_list[0])}")
print(f"最大时间: {ts_list[-1]} -> {fmt(ts_list[-1])}")

# 按年份/月份统计
month_counter = collections.Counter()
for tv in ts_list:
    month_counter[fmt(tv)[:7]] += 1
print("\n按月分布:")
for k in sorted(month_counter):
    print(f"  {k}: {month_counter[k]}")

# 给出若干候选分位点
for p in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    idx = int(n * p)
    v = ts_list[idx]
    print(f"P{p*100:.0f} -> {fmt(v)} ({v})")
