import pandas as pd
from rich.pretty import pprint


def main():
    df = pd.read_csv("2023_24_grants_and_contributions.csv", header=0, encoding="ISO-8859-1")

    print("Columns: ")
    pprint(list(df.columns))
    print("Head: ")
    print(df.head())
    print("Program names: ")
    pprint(df["Program Name (English)"].unique())
    print("IRAP Contributions: ")
    irap_contribs = df[df["Program Name (English)"] == "Industrial Research Assistance Program ? Contributions to Firms"]
    pprint(irap_contribs.head())
    print("Descriptions: ")
    pprint(irap_contribs["Description (English)"].head())

if __name__ == "__main__":
    main()

