import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train.csv", sep=",")
print(df["identity_hate"][df["identity_hate"] != 0])
print(df.columns)

label_list = []

for i, row in df.iterrows():
    label_list.append([row["toxic"], row["severe_toxic"],
                   row["obscene"], row["threat"], row["insult"], row["identity_hate"]])

#df["label"] = [df["toxic"], df["severe_toxic"],
#                df["obscene"], df["threat"], df["insult"], df["identity_hate"]]

df["label"] = label_list
df["alpha"] = "a"

df = df[["id", "label", "alpha", "comment_text"]]

print(df.head())

# df = pd.DataFrame({"1": df["id"], "2": df[], "3": })

train, dev = train_test_split(df, test_size=0.2)
print(train.shape)
print(dev.shape)

train.to_csv("data/train.tsv", sep="\t", header=False, index=False)
dev.to_csv("data/dev.tsv", sep="\t", header=False, index=False)

df = pd.read_csv("data/test.csv", sep=",")
df.to_csv("data/test.tsv", sep="\t", header=True, index=False)
