from bert_score import score

# 比較する2つの文章
cands = ["AIは安全性を重視して西を切った。"]
refs = ["AIは安全性を重視して西を切ったよ"]

# BERTScoreを計算（日本語用の適切なモデルを使用）
P, R, F1 = score(cands, refs, lang="ja")

print(f"Precision: {P.mean().item():.4f}")
print(f"Recall: {R.mean().item():.4f}")
print(f"F1: {F1.mean().item():.4f}")
