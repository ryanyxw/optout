import pandas as pd
import seaborn as sns


df_train = pd.read_csv('out/scoring_out/train_probs.jsonl', header=None, names=['idx', 'sentence', 'contains', 'probability'])
df_val = pd.read_csv('out/scoring_out/val_probs.jsonl', header=None, names=['idx', 'sentence', 'contains', 'probability'])

df_train.probability = df_train.probability.astype(float)
df_val.probability = df_val.probability.astype(float)

print("train")
print(df_train.groupby('contains').mean(numeric_only=True))
print(f"Average prompt of training without comma = {df_train.loc[df_train.contains == False].sentence.map(lambda x: len(x)).mean()}")
print(f"Average prompt of training with comma  = {df_train.loc[df_train.contains].sentence.map(lambda x: len(x)).mean()}")
# print("dabb")

print("val")
print(df_val.groupby('contains').mean(numeric_only=True))
print(f"Average prompt of validation without comma = {df_val.loc[df_val.contains == False].sentence.map(lambda x: len(x)).mean()}")
print(f"Average prompt of validation with comma  = {df_val.loc[df_val.contains].sentence.map(lambda x: len(x)).mean()}")
