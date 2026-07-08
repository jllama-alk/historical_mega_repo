import  os
import pandas as pd

banned_words = []
with open("banned_words.txt", "rt") as f:
    banned_words = f.read().splitlines()


df = pd.read_parquet("hf://datasets/dim/roleplay_instruct_v2_final/data/train-00000-of-00001-52408d4faa8ceeb5.parquet")

for word in banned_words:
    df = df[~df['instruction'].str.contains(word, case=False)]
    df = df[~df['input'].str.contains(word, case=False)]
    df = df[~df['output'].str.contains(word, case=False)]

df = df.iloc[:425]
df.to_csv("filtered_data.csv", index=False)

#df = pd.read_parquet("hf://datasets/allenai/soda/")
#df = df['dialogue'] + df['speakers']


#for word in banned_words:
#    df = df[~df['dialogue'].str.contains(word, case=False)]
#    df = df[~df['speakers'].str.contains(word, case=False)]
