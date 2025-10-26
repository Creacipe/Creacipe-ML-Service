# merge_datasets.py

import pandas as pd
import os

def merge_csv_files():
    # --- PERUBAHAN DI SINI ---
    # Tambahkan path folder 'dataset/'
    dataset_folder = 'dataset/'
    csv_files = [
        'dataset-ayam.csv', 'dataset-ikan.csv', 'dataset-kambing.csv',
        'dataset-sapi.csv', 'dataset-tahu.csv', 'dataset-telur.csv',
        'dataset-tempe.csv', 'dataset-udang.csv'
    ]
    # -------------------------
    
    all_recipes = []
    print("Mulai membaca file-file CSV dari folder 'dataset/'...")

    for file_name in csv_files:
        file_path = os.path.join(dataset_folder, file_name) # Gabungkan folder dan nama file
        if os.path.exists(file_path):
            category_name = file_name.split('-')[1].split('.')[0]
            temp_df = pd.read_csv(file_path)
            temp_df['category'] = category_name
            all_recipes.append(temp_df)
            print(f"- File '{file_path}' berhasil dibaca.")
        else:
            print(f"- Peringatan: File '{file_path}' tidak ditemukan.")

    # ... (sisa kode sama seperti sebelumnya)
    combined_df = pd.concat(all_recipes, ignore_index=True)
    output_filename = 'dataset-gabungan.csv'
    combined_df.to_csv(output_filename, index=False)
    print(f"\nSemua dataset telah digabungkan menjadi '{output_filename}'.")

if __name__ == "__main__":
    merge_csv_files()