import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv("penguins.csv")
    print(data.head())
