"""
HW1 (1) Neighborhood-based CF: ItemKNN vs UserKNN on ml-100k
k in {1, 5, 10, 100, 500}
결과: q1_results.csv, q1_itemknn.png, q1_userknn.png, q1_compare.png
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from recbole.quick_start import run_recbole

KS = [1, 5, 10, 100, 500]
METHODS = {"item": "ItemKNN", "user": "UserKNN"}
METRICS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]

# 1) 실험 실행
rows = []
for method in METHODS:
    for k in KS:
        print(f"\n===== {METHODS[method]}  k={k} =====")
        res = run_recbole(
            model="ItemKNN",
            dataset="ml-100k",
            config_dict={"k": k, "knn_method": method, "show_progress": False},
        )
        test = res["test_result"]
        rows.append({"model": METHODS[method], "k": k,
                     **{m: float(test[m]) for m in METRICS}})

with open("q1_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "k"] + METRICS)
    w.writeheader()
    w.writerows(rows)

# 2) 표 출력
print("\n" + "model    k    " + "  ".join(f"{m:>12}" for m in METRICS))
for r in rows:
    print(f"{r['model']:8} {r['k']:<4} " + "  ".join(f"{r[m]:>12.4f}" for m in METRICS))

# 3) 모델별 그래프 (k에 따른 지표 변화)
for name in METHODS.values():
    sub = [r for r in rows if r["model"] == name]
    plt.figure(figsize=(6, 4))
    for m in METRICS:
        plt.plot([r["k"] for r in sub], [r[m] for r in sub], marker="o", label=m)
    plt.xscale("log")
    plt.xticks(KS, [str(k) for k in KS])
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("score")
    plt.title(f"{name} on ml-100k")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"q1_{name.lower()}.png", dpi=200)
    plt.close()

# 4) ItemKNN vs UserKNN 비교 (NDCG@10, Recall@10)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, m in zip(axes, ["ndcg@10", "recall@10"]):
    for name in METHODS.values():
        sub = [r for r in rows if r["model"] == name]
        ax.plot([r["k"] for r in sub], [r[m] for r in sub], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("k (number of neighbors)")
    ax.set_title(m)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("q1_compare.png", dpi=200)
print("\nSaved: q1_results.csv, q1_itemknn.png, q1_userknn.png, q1_compare.png")
