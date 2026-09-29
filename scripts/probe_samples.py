# -*- coding: utf-8 -*-
"""抽样查看脏数据的具体形态，辅助设计清洗规则。"""
import os
import collections

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml-1m")
ENC = "latin-1"

def read(path):
    with open(path, "r", encoding=ENC) as f:
        return f.readlines()

def show(title, samples):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)
    for s in samples:
        print(repr(s))

# ---- ratings 各字段数的样本 ----
ratings = read(os.path.join(BASE, "ratings.dat"))
by_len = collections.defaultdict(list)
for line in ratings:
    line = line.rstrip("\n")
    by_len[len(line.split("::"))].append(line)

show("ratings 字段数=1 样本(前10)", by_len[1][:10])
show("ratings 字段数=3 样本(前10)", by_len[3][:10])
show("ratings 字段数=5 样本(前10)", by_len[5][:10])

# ---- 时间戳异常样本 ----
print("\n" + "-" * 60)
print("时间戳异常样本(评分记录里 timestamp 非纯秒级的行)")
cnt = 0
for line in ratings:
    f = line.rstrip("\n").split("::")
    if len(f) == 4 and f[3].isdigit():
        tv = int(f[3])
        if tv > 2_000_000_000:
            print(repr(line.rstrip("\n")))
            cnt += 1
            if cnt >= 10:
                break

# ---- 评分异常样本 ----
print("\n" + "-" * 60)
print("评分异常样本(rating 非 1-5 整数)")
cnt = 0
for line in ratings:
    f = line.rstrip("\n").split("::")
    if len(f) == 4 and (not f[2].isdigit() or not (1 <= int(f[2]) <= 5)):
        print(repr(line.rstrip("\n")))
        cnt += 1
        if cnt >= 10:
            break

# ---- users 异常样本 ----
users = read(os.path.join(BASE, "users.dat"))
show("users 字段数!=5 样本(前10)", [l.rstrip("\n") for l in users if len(l.rstrip("\n").split("::")) != 5][:10])
print("\n" + "-" * 60)
print("users Gender=X 样本(前5)")
print("\n".join(repr(l.rstrip("\n")) for l in users if len(l.rstrip("\n").split("::")) == 5 and l.rstrip("\n").split("::")[1] == "X")[:5])
print("\nusers Age=30 样本(前5)")
print("\n".join(repr(l.rstrip("\n")) for l in users if len(l.rstrip("\n").split("::")) == 5 and l.rstrip("\n").split("::")[2] == "30")[:5])
print("\nusers Occupation=99 样本(前5)")
print("\n".join(repr(l.rstrip("\n")) for l in users if len(l.rstrip("\n").split("::")) == 5 and l.rstrip("\n").split("::")[3] == "99")[:5])

# ---- movies 异常样本 ----
movies = read(os.path.join(BASE, "movies.dat"))
show("movies 字段数!=3 样本(前10)", [l.rstrip("\n") for l in movies if len(l.rstrip("\n").split("::")) != 3][:10])
print("\n" + "-" * 60)
print("movies 标题无年份 样本(前10)")
print("\n".join(repr(l.rstrip("\n")) for l in movies if len(l.rstrip("\n").split("::")) == 3 and "(" not in l.rstrip("\n").split("::")[1])[:10])
print("\nmovies UnknownGenre 样本(前5)")
print("\n".join(repr(l.rstrip("\n")) for l in movies if len(l.rstrip("\n").split("::")) == 3 and "UnknownGenre" in l.rstrip("\n"))[:5])
