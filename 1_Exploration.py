import pandas as pd

nvda = pd.read_csv("https://raw.githubusercontent.com/tiagopint0/Trabalho2PCD/main/SourceFile/NVDA.csv")

print("Let's see if our dataset has any NA values:\n")
na_vals = nvda[nvda.isna().any(axis=1)]
print(na_vals)
print("\nLet's see, now, if our dataset has any NULL values:\n")
null_vals = nvda[nvda.isnull().any(axis=1)]
print(null_vals)

if len(na_vals)!=0 or len(null_vals)!=0:
    print("\nLet's delete these.")
    nvda.dropna(axis=0,inplace=True)
    print("\nNA/NULL Values:\n", nvda[nvda.isna().any(axis=1)])
    print("\nDone.")

print("Let's now look at some statistics about our data:\n")
print(nvda.describe())