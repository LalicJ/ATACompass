import pandas as pd

df = pd.read_csv("results/longest_blat_results.txt", sep="\t")
filtered = df[df['percentIdentity'] >= 60]  # 假设存在此列
filtered.to_csv("results/conserved_sequences.txt", sep="\t", index=False)
