import pandas as pd

df = pd.read_csv("dataset_csv/slide_labels.csv")

df["case_id"] = df["slide_id"].apply(lambda x: x.split("-")[0] + "-" + x.split("-")[1] + "-" + x.split("-")[2])

df["slide_id"] = df["slide_id"]

df = df[["case_id", "slide_id", "label"]]

df.to_csv("dataset_csv/tumor_vs_normal_dummy_clean우리꺼.csv", index=False)

print(" csv 컷컷컷")