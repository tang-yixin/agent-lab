# -*- coding: utf-8 -*-
"""
数据探查脚本：扫描 ml-1m 三个文件，找出真实存在的质量问题。
仅使用标准库，输出结构化问题清单，供后续设计清洗规则和评分方案。
"""
import os
import re
import collections

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml-1m")
ENC = "latin-1"  # ISO-8859-1

RATINGS = os.path.join(BASE, "ratings.dat")
USERS = os.path.join(BASE, "users.dat")
MOVIES = os.path.join(BASE, "movies.dat")


def read_lines(path):
    with open(path, "r", encoding=ENC) as f:
        return f.readlines()


def report(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------- ratings.dat ----------
def probe_ratings():
    report("ratings.dat")
    lines = read_lines(RATINGS)
    total = len(lines)
    print(f"总记录数: {total}")

    field_counts = collections.Counter()
    non_int = {"UserID": 0, "MovieID": 0, "Rating": 0, "Timestamp": 0}
    rating_out_of_range = 0
    rating_non_int = 0
    empty_fields = 0
    ts_min = ts_max = None
    ts_valid = 0
    ts_invalid = 0
    seen = set()          # 用于查完全重复
    dup_count = 0
    key_map = collections.defaultdict(list)  # (user,movie,timestamp) -> rating
    conflict = 0
    user_ids = set()
    movie_ids = set()

    for i, line in enumerate(lines, 1):
        line = line.rstrip("\n")
        fields = line.split("::")
        field_counts[len(fields)] += 1
        if len(fields) != 4:
            empty_fields += 1
            continue
        u, m, r, t = fields
        user_ids.add(u)
        movie_ids.add(m)
        # 类型检查
        for name, val in [("UserID", u), ("MovieID", m), ("Rating", r), ("Timestamp", t)]:
            if not val.isdigit():
                non_int[name] += 1
        if r.isdigit():
            rating_val = int(r)
            if rating_val < 1 or rating_val > 5:
                rating_out_of_range += 1
        else:
            rating_non_int += 1
        # 时间戳
        if t.isdigit():
            tv = int(t)
            if ts_min is None or tv < ts_min:
                ts_min = tv
            if ts_max is None or tv > ts_max:
                ts_max = tv
            # 合理范围：约 1970 之后，2030 之前（秒级）
            if 0 < tv < 2_000_000_000:
                ts_valid += 1
            else:
                ts_invalid += 1
        else:
            ts_invalid += 1
        # 完全重复
        key = (u, m, r, t)
        if key in seen:
            dup_count += 1
        seen.add(key)
        # 同一 (user, movie, timestamp) 不同 rating 冲突
        key_map[(u, m, t)].append(r)

    for k, v in key_map.items():
        if len(set(v)) > 1:
            conflict += 1

    print(f"字段数量分布(字段数->条数): {dict(field_counts)}")
    print(f"非整数字段统计: {dict(non_int)}")
    print(f"评分越界(不在1-5): {rating_out_of_range}")
    print(f"评分非整数: {rating_non_int}")
    print(f"字段数!=4 的记录(视为空/解析异常): {empty_fields}")
    print(f"时间戳范围: {ts_min} ~ {ts_max}")
    print(f"时间戳合法(秒级)条数: {ts_valid}, 非法: {ts_invalid}")
    if ts_min and ts_max:
        import datetime
        def safe_ts(v):
            try:
                return datetime.datetime.utcfromtimestamp(v)
            except (OSError, ValueError, OverflowError):
                return f"<无法换算:{v}>"
        print(f"时间戳换算: {safe_ts(ts_min)} ~ {safe_ts(ts_max)} (UTC)")
    print(f"完全重复记录数: {dup_count}")
    print(f"(user,movie,timestamp)相同但rating不同的冲突数: {conflict}")
    print(f"出现过的 UserID 去重数: {len(user_ids)}")
    print(f"出现过的 MovieID 去重数: {len(movie_ids)}")


# ---------- users.dat ----------
def probe_users():
    report("users.dat")
    lines = read_lines(USERS)
    total = len(lines)
    print(f"总记录数: {total}")

    field_counts = collections.Counter()
    gender_vals = collections.Counter()
    age_vals = collections.Counter()
    occ_vals = collections.Counter()
    empty_fields = 0
    zip_leading_zero = 0
    zip_not_numeric = 0
    dup_user = collections.defaultdict(list)
    user_ids = set()
    valid_age = {1, 18, 25, 35, 45, 50, 56}
    valid_gender = {"M", "F"}

    for i, line in enumerate(lines, 1):
        line = line.rstrip("\n")
        fields = line.split("::")
        field_counts[len(fields)] += 1
        if len(fields) != 5:
            empty_fields += 1
            continue
        uid, g, a, o, z = fields
        user_ids.add(uid)
        gender_vals[g] += 1
        age_vals[a] += 1
        occ_vals[o] += 1
        if z.isdigit() and len(z) < 5 and z.startswith("0"):
            zip_leading_zero += 1
        if not z.isdigit():
            zip_not_numeric += 1
        dup_user[uid].append((g, a, o, z))

    conflict = sum(1 for v in dup_user.values() if len(set(v)) > 1)
    print(f"字段数量分布: {dict(field_counts)}")
    print(f"字段数!=5 的记录: {empty_fields}")
    print(f"Gender 取值分布: {dict(gender_vals)}")
    print(f"Age 取值分布: {dict(age_vals)}")
    print(f"Occupation 取值分布: {dict(occ_vals)}")
    print(f"非法 Gender(非M/F): {sum(v for k,v in gender_vals.items() if k not in valid_gender)}")
    print(f"非法 Age: {sum(v for k,v in age_vals.items() if not (k.isdigit() and int(k) in valid_age))}")
    print(f"邮编含前导零(可能被误读为数值而丢失): {zip_leading_zero}")
    print(f"邮编非纯数字: {zip_not_numeric}")
    print(f"同一 UserID 属性冲突数: {conflict}")
    print(f"UserID 去重数: {len(user_ids)}")


# ---------- movies.dat ----------
def probe_movies():
    report("movies.dat")
    lines = read_lines(MOVIES)
    total = len(lines)
    print(f"总记录数: {total}")

    field_counts = collections.Counter()
    empty_fields = 0
    movie_ids = set()
    dup_movie = collections.defaultdict(list)
    title_missing = 0
    year_missing = 0
    year_bad = 0
    genre_counter = collections.Counter()
    movie_without_genre = 0
    same_title_diff_id = collections.defaultdict(set)

    year_re = re.compile(r"\((\d{4})\)\s*$")

    for i, line in enumerate(lines, 1):
        line = line.rstrip("\n")
        fields = line.split("::")
        field_counts[len(fields)] += 1
        if len(fields) != 3:
            empty_fields += 1
            continue
        mid, title, genres = fields
        movie_ids.add(mid)
        if not title.strip():
            title_missing += 1
        m = year_re.search(title)
        if not m:
            year_missing += 1
        else:
            yr = int(m.group(1))
            if yr < 1880 or yr > 2010:
                year_bad += 1
        if not genres.strip():
            movie_without_genre += 1
        else:
            for g in genres.split("|"):
                genre_counter[g.strip()] += 1
        dup_movie[mid].append((title, genres))
        same_title_diff_id[title].add(mid)

    conflict = sum(1 for v in dup_movie.values() if len(set(v)) > 1)
    same_title_multi_id = sum(1 for v in same_title_diff_id.values() if len(v) > 1)

    print(f"字段数量分布: {dict(field_counts)}")
    print(f"字段数!=3 的记录: {empty_fields}")
    print(f"标题缺失/空白: {title_missing}")
    print(f"标题无年份: {year_missing}")
    print(f"年份异常(不在1880-2010): {year_bad}")
    print(f"无类型信息: {movie_without_genre}")
    print(f"类型取值分布(前30): {dict(genre_counter.most_common(30))}")
    print(f"同一 MovieID 标题/类型冲突数: {conflict}")
    print(f"同名电影对应多个 MovieID 的情况数: {same_title_multi_id}")
    print(f"MovieID 去重数: {len(movie_ids)}")


# ---------- 跨表关联 ----------
def probe_cross():
    report("跨表关联检查")
    # 收集合法 ID
    user_ids = set()
    for line in read_lines(USERS):
        f = line.rstrip("\n").split("::")
        if len(f) == 5:
            user_ids.add(f[0])
    movie_ids = set()
    for line in read_lines(MOVIES):
        f = line.rstrip("\n").split("::")
        if len(f) == 3:
            movie_ids.add(f[0])

    rating_user_orphan = 0
    rating_movie_orphan = 0
    for line in read_lines(RATINGS):
        f = line.rstrip("\n").split("::")
        if len(f) != 4:
            continue
        if f[0] not in user_ids:
            rating_user_orphan += 1
        if f[1] not in movie_ids:
            rating_movie_orphan += 1

    print(f"users.dat 合法 UserID 数: {len(user_ids)}")
    print(f"movies.dat 合法 MovieID 数: {len(movie_ids)}")
    print(f"ratings 中引用了不存在 UserID 的记录数: {rating_user_orphan}")
    print(f"ratings 中引用了不存在 MovieID 的记录数: {rating_movie_orphan}")


if __name__ == "__main__":
    probe_ratings()
    probe_users()
    probe_movies()
    probe_cross()
    print("\n探查完成。")
