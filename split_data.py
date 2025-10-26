# split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_file='dataset-final.csv', train_file='train_data.csv', test_file='test_data.csv', test_size=0.2, random_state=42):
    """
    Membaca dataset input, membaginya menjadi training dan testing set,
    dan menyimpannya ke dalam dua file CSV baru.
    """
    print(f"Membaca dataset dari '{input_file}'...")
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' tidak ditemukan. Jalankan 'preprocess_data.py' terlebih dahulu.")
        return

    try:
        df = pd.read_csv(input_file)
        
        print(f"Membagi dataset ({1-test_size:.0%} train / {test_size:.0%} test)...")
        
        # Melakukan pembagian data
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Menyimpan hasil ke file CSV baru
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"Data berhasil dibagi:")
        print(f"- Data training disimpan di '{train_file}' ({len(train_df)} baris)")
        print(f"- Data testing disimpan di '{test_file}' ({len(test_df)} baris)")

    except Exception as e:
        print(f"Terjadi error saat membagi data: {e}")

if __name__ == "__main__":
    split_dataset()