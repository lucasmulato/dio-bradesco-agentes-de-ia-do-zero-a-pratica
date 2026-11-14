import pandas as pd

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

def load_data(url=URL):
    df = pd.read_csv(url)
    print(f"Shape: {df.shape} | Fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df
