# -*- coding: utf-8 -*-
"""
降维任务训练流水线：压缩训练期用户偏好画像（user×movie 稀疏矩阵）。

  - 算法：PCA（中心化后主成分）+ SVD（TruncatedSVD，稀疏矩阵不中心化），
    维数网格 {2, 10, 20, 50}，每个维数各登记一个变体（2 维用于可视化）；
  - 信息保留：累计方差解释率 + Frobenius 相对重构误差；
  - 资源开销：拟合耗时、峰值内存增量、降维坐标存储体积 vs 原始矩阵体积；
  - 聚类对照：用聚类任务登记的最优 KMeans/DBSCAN 参数，分别在原始画像与
    降维坐标上运行，对比 Silhouette/SSE/簇数/噪声比（同一口径，严格可比）；
  - 主要变化方向：各主成分上 |载荷| 最高的电影（PCA/SVD 各 Top-8）。

输出：models/*.joblib + coords/*.npz + plots/*.png + eval/*_eval.json + 注册表。
"""
import json
import os
import resource
import time

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize as sk_normalize

from common import (T1_TS, T2_TS, T1_LABEL, T2_LABEL, ARTIFACTS, DATA_VERSION,
                    VERSIONS, now_str, load_registry, save_registry, fail, ok)

PROFILE_DIR = os.path.join(ARTIFACTS, "profiles")
MODELS_DIR = os.path.join(ARTIFACTS, "models")
COORDS_DIR = os.path.join(ARTIFACTS, "coords")
PLOTS_DIR = os.path.join(ARTIFACTS, "plots")
EVAL_DIR = os.path.join(ARTIFACTS, "eval")
LABELS_DIR = os.path.join(ARTIFACTS, "labels")

DIM_GRID = [2, 10, 20, 50]
TOP_COMPONENT_MOVIES = 8


def _peak_mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load_context():
    train_mat = sp.load_npz(os.path.join(PROFILE_DIR, "train_matrix.npz"))
    with open(os.path.join(PROFILE_DIR, "user_ids.json"), encoding="utf-8") as f:
        user_ids = json.load(f)
    with open(os.path.join(PROFILE_DIR, "movie_ids.json"), encoding="utf-8") as f:
        movie_ids = json.load(f)
    with open(os.path.join(PROFILE_DIR, "movies_table.json"), encoding="utf-8") as f:
        movies_table = json.load(f)
    with open(os.path.join(EVAL_DIR, "kmeans_eval.json"), encoding="utf-8") as f:
        kmeans_eval = json.load(f)
    with open(os.path.join(EVAL_DIR, "dbscan_eval.json"), encoding="utf-8") as f:
        dbscan_eval = json.load(f)
    return train_mat, user_ids, movie_ids, movies_table, kmeans_eval, dbscan_eval


def _recon_error_pca(Xc, model):
    """PCA：中心化矩阵的相对重构误差。"""
    X_hat = model.inverse_transform(model.transform(Xc))
    denom = np.linalg.norm(Xc, "fro")
    return float(np.linalg.norm(Xc - X_hat, "fro") / denom)


def _recon_error_svd(X, model):
    """SVD：原始稀疏矩阵的相对重构误差。"""
    X_hat = model.inverse_transform(model.transform(X))
    denom = sp.linalg.norm(X, "fro")
    err = np.linalg.norm(X.toarray() - X_hat, "fro")
    return float(err / denom)


def _fit_with_resource(fit_fn):
    """记录拟合耗时与进程峰值内存（ru_maxrss 为单调峰值，报告绝对峰值）。
    返回 (result, cost)。"""
    t0 = time.time()
    result = fit_fn()
    dt = time.time() - t0
    return result, {"fit_time_s": round(dt, 3),
                    "peak_memory_mb": round(_peak_mem_mb(), 1)}


def _kmeans_on(mat, k, sparse):
    """在给定表示上运行 KMeans（余弦空间），返回指标。"""
    Xn = sk_normalize(mat, norm="l2", copy=True)
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    labels, _ = _fit_with_resource(lambda: km.fit_predict(Xn))
    sil = float(silhouette_score(mat, labels, metric="cosine"))
    return {"labels": labels, "silhouette": round(sil, 4),
            "sse": round(float(km.inertia_), 2),
            "cluster_sizes": np.bincount(labels, minlength=k).tolist()}


def _dbscan_on(mat, eps, ms):
    """在给定表示上运行 DBSCAN（余弦距离），返回指标。"""
    db = DBSCAN(eps=eps, min_samples=ms, metric="cosine", n_jobs=-1)
    labels, _ = _fit_with_resource(lambda: db.fit_predict(mat))
    n_clusters = len(set(labels) - {-1})
    sil = None
    keep = labels != -1
    if keep.sum() >= 2 and n_clusters >= 2:
        sil = float(silhouette_score(mat[keep], labels[keep], metric="cosine"))
    return {"labels": labels, "silhouette": round(sil, 4) if sil else None,
            "n_clusters": n_clusters,
            "noise_ratio": round(float((labels == -1).mean()), 4)}


def _top_movies_per_component(components, movie_ids, movies_table, k):
    """各主成分 |载荷| 最高的电影（主要变化方向）。

    个别电影属性在迭代一被隔离（不在清洗后电影表），跳过并计数。
    """
    out = []
    skipped = 0
    for c in range(min(k, components.shape[0])):
        idx = np.argsort(np.abs(components[c]))[::-1]
        movies = []
        for j in idx:
            info = movies_table.get(str(movie_ids[j]))
            if info is None:
                skipped += 1
                continue
            movies.append({"movie_id": movie_ids[j], "title": info["title"],
                           "loading": round(float(components[c][j]), 4)})
            if len(movies) >= TOP_COMPONENT_MOVIES:
                break
        out.append({"component": c + 1, "movies": movies})
    return out, skipped


def train():
    (train_mat, user_ids, movie_ids, movies_table,
     kmeans_eval, dbscan_eval) = _load_context()
    for d in (MODELS_DIR, COORDS_DIR, PLOTS_DIR, EVAL_DIR):
        os.makedirs(d, exist_ok=True)

    k_best = kmeans_eval["selected_params"]["k"]
    db_eps = dbscan_eval["selected_params"]["eps"]
    db_ms = dbscan_eval["selected_params"]["min_samples"]

    # 原始画像上的聚类指标（复用聚类任务已登记的训练期结果，口径一致）
    kmeans_orig = {
        "silhouette": kmeans_eval["selected"]["train"]["silhouette"],
        "sse": kmeans_eval["selected"]["train"]["sse"],
        "cluster_sizes": kmeans_eval["selected"]["train"]["cluster_sizes"],
    }
    dbscan_orig = {
        "silhouette": dbscan_eval["selected"]["train"]["silhouette"],
        "n_clusters": dbscan_eval["selected"]["train"]["n_clusters"],
        "noise_ratio": dbscan_eval["selected"]["train"]["noise_ratio"],
    }

    # 中心化（PCA 用）+ 原始稀疏（SVD 用）
    Xd = train_mat.toarray().astype(np.float32)
    mean_vec = Xd.mean(axis=0)
    Xc = Xd - mean_vec
    original_npz = os.path.join(PROFILE_DIR, "train_matrix.npz")
    original_storage_mb = round(os.path.getsize(original_npz) / 1e6, 3)

    pca_variants, svd_variants = [], []
    scatter_extra = {}

    for dim in DIM_GRID:
        # ---------- PCA ----------
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=42)
        pca_coords, cost_pca = _fit_with_resource(lambda: pca.fit_transform(Xc))
        pca_retained = round(float(pca.explained_variance_ratio_.sum()), 4)
        pca_recon = round(_recon_error_pca(Xc, pca), 4)
        np.savez_compressed(os.path.join(COORDS_DIR, "pca_dim{}.npz".format(dim)),
                            coords=pca_coords.astype(np.float32))
        coords_storage_mb = round(
            os.path.getsize(os.path.join(COORDS_DIR, "pca_dim{}.npz".format(dim))) / 1e6, 3)

        # ---------- SVD ----------
        svd = TruncatedSVD(n_components=dim, random_state=42)
        svd_coords, cost_svd = _fit_with_resource(lambda: svd.fit_transform(train_mat))
        svd_retained = round(float(svd.explained_variance_ratio_.sum()), 4)
        svd_recon = round(_recon_error_svd(train_mat, svd), 4)
        np.savez_compressed(os.path.join(COORDS_DIR, "svd_dim{}.npz".format(dim)),
                            coords=svd_coords.astype(np.float32))
        svd_coords_storage_mb = round(
            os.path.getsize(os.path.join(COORDS_DIR, "svd_dim{}.npz".format(dim))) / 1e6, 3)

        joblib.dump(pca, os.path.join(MODELS_DIR, "pca_dim{}.joblib".format(dim)))
        joblib.dump(svd, os.path.join(MODELS_DIR, "svd_dim{}.joblib".format(dim)))

        # ---------- 降维坐标上的聚类对照（与原始画像同一参数口径） ----------
        km_pca = _kmeans_on(pca_coords, k_best, sparse=False)
        km_svd = _kmeans_on(svd_coords, k_best, sparse=False)
        db_pca = _dbscan_on(pca_coords, db_eps, db_ms)
        db_svd = _dbscan_on(svd_coords, db_eps, db_ms)

        def _cmp(km_red, db_red, km_orig, db_orig):
            return {
                "kmeans": {
                    "original": {"silhouette": km_orig["silhouette"],
                                 "sse": km_orig["sse"],
                                 "cluster_sizes": km_orig["cluster_sizes"]},
                    "reduced": {"silhouette": km_red["silhouette"],
                                "sse": km_red["sse"],
                                "cluster_sizes": km_red["cluster_sizes"]},
                },
                "dbscan": {
                    "original": {"silhouette": db_orig["silhouette"],
                                 "n_clusters": db_orig["n_clusters"],
                                 "noise_ratio": db_orig["noise_ratio"]},
                    "reduced": {"silhouette": db_red["silhouette"],
                                "n_clusters": db_red["n_clusters"],
                                "noise_ratio": db_red["noise_ratio"]},
                },
            }

        pca_variants.append({
            "n_components": dim,
            "retention": {"explained_variance_ratio": pca_retained,
                          "recon_error_fro": pca_recon},
            "resource": dict(cost_pca,
                             coords_storage_mb=coords_storage_mb,
                             original_storage_mb=original_storage_mb),
            "cluster_comparison": _cmp(km_pca, db_pca, kmeans_orig, dbscan_orig),
        })
        svd_variants.append({
            "n_components": dim,
            "retention": {"explained_variance_ratio": svd_retained,
                          "recon_error_fro": svd_recon},
            "resource": dict(cost_svd,
                             coords_storage_mb=svd_coords_storage_mb,
                             original_storage_mb=original_storage_mb),
            "cluster_comparison": _cmp(km_svd, db_svd, kmeans_orig, dbscan_orig),
        })

        if dim == 2:
            scatter_extra = {
                "pca": {"coords": pca_coords.astype(np.float32),
                        "kmeans_labels": km_pca["labels"],
                        "dbscan_labels": db_pca["labels"]},
                "svd": {"coords": svd_coords.astype(np.float32),
                        "kmeans_labels": km_svd["labels"],
                        "dbscan_labels": db_svd["labels"]},
            }

    # ---------- 2D 可视化：坐标 JSON（前端绘制）+ PNG 归档 ----------
    for algo in ("pca", "svd"):
        extra = scatter_extra[algo]
        pts = []
        for i, uid in enumerate(user_ids):
            pts.append({
                "user_id": int(uid),
                "x": round(float(extra["coords"][i][0]), 4),
                "y": round(float(extra["coords"][i][1]), 4),
                "kmeans_cluster": int(extra["kmeans_labels"][i]),
                "dbscan_cluster": int(extra["dbscan_labels"][i]),
            })
        with open(os.path.join(COORDS_DIR, "scatter_{}2d.json".format(algo)), "w",
                  encoding="utf-8") as f:
            json.dump(pts, f)
        _save_scatter_png(pts, os.path.join(PLOTS_DIR, "scatter_{}2d.png".format(algo)))

    # ---------- 主要变化方向 ----------
    pca50 = joblib.load(os.path.join(MODELS_DIR, "pca_dim50.joblib"))
    svd50 = joblib.load(os.path.join(MODELS_DIR, "svd_dim50.joblib"))
    pca_dirs, _skip_pca = _top_movies_per_component(pca50.components_, movie_ids,
                                                    movies_table, 5)
    svd_dirs, _skip_svd = _top_movies_per_component(svd50.components_, movie_ids,
                                                    movies_table, 5)

    time_cfg = {
        "data_version": DATA_VERSION,
        "time_boundary": {"T1": T1_LABEL, "T2": T2_LABEL,
                          "T1_timestamp": T1_TS, "T2_timestamp": T2_TS},
        "input": "训练期 user×movie 稀疏画像（{} 用户 × {} 电影）".format(
            len(user_ids), len(movie_ids)),
        "pca_note": "PCA 先中心化（逐列减均值）再分解；SVD 对稀疏原始矩阵直接分解（不中心化）。"
                    "重构误差为 Frobenius 相对误差。",
        "cluster_note": "聚类对照使用聚类任务登记的最优参数（KMeans k={}；DBSCAN eps={}, min_samples={}），"
                        "原始画像与降维坐标完全同一口径。".format(k_best, db_eps, db_ms),
    }
    pca_eval = dict(time_cfg)
    pca_eval.update({
        "artifact_id": VERSIONS["dim_pca"], "algorithm": "pca",
        "version": VERSIONS["dim_pca"], "dim_grid": DIM_GRID,
        "variants": pca_variants,
        "principal_directions_top_movies": pca_dirs,
        "files_2d": ["coords/scatter_pca2d.json", "plots/scatter_pca2d.png"],
    })
    svd_eval = dict(time_cfg)
    svd_eval.update({
        "artifact_id": VERSIONS["dim_svd"], "algorithm": "svd",
        "version": VERSIONS["dim_svd"], "dim_grid": DIM_GRID,
        "variants": svd_variants,
        "principal_directions_top_movies": svd_dirs,
        "files_2d": ["coords/scatter_svd2d.json", "plots/scatter_svd2d.png"],
    })
    with open(os.path.join(EVAL_DIR, "pca_eval.json"), "w", encoding="utf-8") as f:
        json.dump(pca_eval, f, ensure_ascii=False, indent=2)
    with open(os.path.join(EVAL_DIR, "svd_eval.json"), "w", encoding="utf-8") as f:
        json.dump(svd_eval, f, ensure_ascii=False, indent=2)

    # ---------- 注册 ----------
    reg = load_registry() or {"meta": {}, "entries": []}
    for eid, alg, eval_obj in [(VERSIONS["dim_pca"], "pca", pca_eval),
                               (VERSIONS["dim_svd"], "svd", svd_eval)]:
        best = eval_obj["variants"][-1]  # 登记默认口径为最大维数变体
        entry = {
            "artifact_id": eid, "task": "dimension_reduction", "algorithm": alg,
            "version": eid, "data_version": DATA_VERSION,
            "dim_grid": DIM_GRID,
            "selected_params": {"n_components": best["n_components"]},
            "metrics": {"retention": best["retention"],
                        "resource": best["resource"],
                        "cluster_comparison": best["cluster_comparison"]},
            "files": ["models/{}_dim{}.joblib".format(alg, d) for d in DIM_GRID]
                     + ["coords/{}_dim{}.npz".format(alg, d) for d in DIM_GRID]
                     + ["eval/{}_eval.json".format(alg)],
            "created_at": now_str(),
        }
        reg["entries"] = [e for e in reg["entries"]
                          if e.get("artifact_id") != eid]
        reg["entries"].append(entry)
    save_registry(reg)

    return ok({
        "pca_retention_dim50": pca_variants[-1]["retention"],
        "svd_retention_dim50": svd_variants[-1]["retention"],
        "pca_resource_dim50": pca_variants[-1]["resource"],
        "svd_resource_dim50": svd_variants[-1]["resource"],
    })


def _save_scatter_png(pts, path):
    """归档 PNG 散点图（英文标签，规避容器内中文字体问题；前端用 JSON 绘制）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    cs = [p["kmeans_cluster"] for p in pts]
    noise = [p["dbscan_cluster"] == -1 for p in pts]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, c=cs, cmap="tab10", s=6, alpha=0.6,
               label="colored by KMeans cluster")
    ax.scatter([x for x, n in zip(xs, noise) if n],
               [y for y, n in zip(ys, noise) if n],
               s=14, facecolors="none", edgecolors="red", linewidths=0.5,
               label="DBSCAN noise")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("2D user profile scatter (KMeans colors, DBSCAN noise marked)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(train(), ensure_ascii=False))
