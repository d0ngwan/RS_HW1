"""
HW1 (2) Latent Factor Model: BPR on ml-100k
 (i)  reg_lambda = 0.1,  embedding_size in {32, 64, 128, 256}
 (ii) embedding_size = 64, reg_lambda in {0.01, 0.1, 0.5}
결과: q2_results.csv, q2_bpr_embedding.png, q2_bpr_reg.png

RecBole 기본 BPR에는 정규화 항(reg_lambda)이 없어서,
BPR loss에 L2 정규화 항  reg_lambda * (||u||^2 + ||i+||^2 + ||i-||^2) / (2 * batch) 를
추가한 BPRReg 모델을 정의해서 사용한다.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.model.loss import EmbLoss
from recbole.utils import init_seed, init_logger, get_trainer


class BPRReg(BPR):
    """BPR + L2 regularization on the embeddings used in each batch."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.reg_lambda = config["reg_lambda"]
        self.reg_loss = EmbLoss()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        u_e = self.user_embedding(user)
        pos_e = self.item_embedding(pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_score = (u_e * pos_e).sum(dim=1)
        neg_score = (u_e * neg_e).sum(dim=1)
        bpr_loss = self.loss(pos_score, neg_score)
        reg = self.reg_loss(u_e, pos_e, neg_e, require_pow=True)
        return bpr_loss + self.reg_lambda * reg


def run_bpr(embedding_size, reg_lambda):
    config = Config(
        model=BPRReg,
        dataset="ml-100k",
        config_dict={
            "embedding_size": embedding_size,
            "reg_lambda": reg_lambda,
            "show_progress": False,
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = BPRReg(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.fit(train_data, valid_data, saved=True, show_progress=False)
    return trainer.evaluate(test_data, load_best_model=True, show_progress=False)


METRICS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]
EMB_SIZES = [32, 64, 128, 256]
REG_LAMBDAS = [0.01, 0.1, 0.5]

settings = [(e, 0.1) for e in EMB_SIZES] + [(64, r) for r in REG_LAMBDAS if r != 0.1]

results = {}
for emb, reg in settings:
    print(f"\n===== BPR  embedding_size={emb}  reg_lambda={reg} =====")
    test = run_bpr(emb, reg)
    results[(emb, reg)] = {m: float(test[m]) for m in METRICS}

# CSV 저장
with open("q2_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["embedding_size", "reg_lambda"] + METRICS)
    for (emb, reg), r in results.items():
        w.writerow([emb, reg] + [r[m] for m in METRICS])

# 표 출력
print("\nemb   reg    " + "  ".join(f"{m:>12}" for m in METRICS))
for (emb, reg), r in results.items():
    print(f"{emb:<5} {reg:<6} " + "  ".join(f"{r[m]:>12.4f}" for m in METRICS))


def line_plot(xs, ys, xlabel, title, fname):
    # 과제 문서의 예시 그래프와 같은 스타일 (선 + 점, 선형 x축)
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, marker="o", color="#E8364F", label="BPR")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=9)
    plt.xticks(xs, [f"{x:.2f}" if isinstance(x, float) else str(x) for x in xs])
    pad = (max(ys) - min(ys)) * 0.3 or 0.01
    plt.ylim(min(ys) - pad, max(ys) + pad)
    plt.xlabel(xlabel)
    plt.ylabel("NDCG@10")
    plt.title(title)
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


line_plot(EMB_SIZES, [results[(e, 0.1)]["ndcg@10"] for e in EMB_SIZES],
          "Embedding Size", "NDCG@10 vs Embedding Size (reg_lambda = 0.1)",
          "q2_bpr_embedding.png")
line_plot(REG_LAMBDAS, [results[(64, r)]["ndcg@10"] for r in REG_LAMBDAS],
          "Regularization Lambda", "NDCG@10 vs Regularization Lambda (embedding_size = 64)",
          "q2_bpr_reg.png")

print("\nSaved: q2_results.csv, q2_bpr_embedding.png, q2_bpr_reg.png")
