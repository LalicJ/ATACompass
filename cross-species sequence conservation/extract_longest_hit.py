import pandas as pd

df = pd.read_csv("results/blat_results.txt", sep="\t", skiprows=5)
df['Q length'] = df['Q end'] - df['Q start']
longest = df.loc[df.groupby('Q name')['Q length'].idxmax()]
longest.to_csv("results/longest_blat_results.txt", sep="\t", index=False)
