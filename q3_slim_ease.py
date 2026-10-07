"""
HW1 (3) Item-to-item CF: SLIM, EASE vs ItemKNN on ml-100k (RecBole 기본 하이퍼파라미터)
결과: q3_results.csv, q3_compare.png
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from recbole.quick_start import run_recbole

MODELS = ["ItemKNN", "SLIMElastic", "EASE"]   # ItemKNN: 기본 k=100
METRICS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]

results = {}
for model in MODELS:
    print(f"\n===== {model} =====")
    res = run_recbole(model=model, dataset="ml-100k",
                      config_dict={"show_progress": False})
    results[model] = {m: float(res["test_result"][m]) for m in METRICS}

# CSV 저장
with open("q3_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model"] + METRICS)
    for model in MODELS:
        w.writerow([model] + [results[model][m] for m in METRICS])

# 표 출력
print("\nmodel        " + "  ".join(f"{m:>12}" for m in METRICS))
for model in MODELS:
    print(f"{model:12} " + "  ".join(f"{results[model][m]:>12.4f}" for m in METRICS))

# 지표별 그룹 막대 그래프
x = range(len(METRICS))
width = 0.25
colors = ["#F2A900", "#F26B21", "#E8364F"]
plt.figure(figsize=(9, 4.5))
for i, model in enumerate(MODELS):
    vals = [results[model][m] for m in METRICS]
    pos = [p + (i - 1) * width for p in x]
    bars = plt.bar(pos, vals, width, label=model, color=colors[i])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7)
plt.xticks(list(x), METRICS)
plt.ylabel("score")
plt.title("ItemKNN vs SLIM vs EASE on ml-100k")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("q3_compare.png", dpi=200)
print("\nSaved: q3_results.csv, q3_compare.png")
