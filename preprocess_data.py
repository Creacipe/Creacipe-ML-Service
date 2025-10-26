# preprocess_data.py

import pandas as pd
import re
import os

def preprocess_and_create_tags():
    """
    Membaca dataset gabungan, membuat tag otomatis, 
    dan menyimpan hasilnya ke file baru.
    """
    input_filename = 'dataset-gabungan.csv'
    output_filename = 'dataset-final.csv'

    if not os.path.exists(input_filename):
        print(f"Error: File '{input_filename}' tidak ditemukan. Jalankan 'merge_datasets.py' terlebih dahulu.")
        return

    print(f"Membaca '{input_filename}'...")
    df = pd.read_csv(input_filename)

    # Definisikan kamus kata kunci untuk membuat tag
    tag_dictionary = {
        'ayam': 'Ayam', 'sapi': 'Sapi', 'kambing': 'Kambing', 'ikan': 'Ikan',
        'telur': 'Telur', 'tahu': 'Tahu', 'tempe': 'Tempe', 'udang': 'Udang',
        'santan': 'Santan', 'nasi': 'Nasi', 'mie': 'Mie', 'sayur': 'Sayuran',
        'goreng': 'Goreng', 'bakar': 'Bakar', 'rebus': 'Rebus', 'tumis': 'Tumis',
        'kuah': 'Kuah', 'panggang': 'Panggang', 'sate': 'Sate', 'soto': 'Soto',
        'pedas': 'Pedas', 'cabe': 'Pedas', 'rawit': 'Pedas',
        'manis': 'Manis', 'asam': 'Asam', 'gurih': 'Gurih'
    }

    def generate_tags(row):
        # Gabungkan title dan ingredients untuk pencarian kata kunci
        content = (str(row['Title']) + ' ' + str(row['Ingredients'])).lower()
        found_tags = set()
        for keyword, tag in tag_dictionary.items():
            if re.search(r'\b' + keyword + r'\b', content):
                found_tags.add(tag)
        return ' '.join(list(found_tags))

    print("Membuat kolom 'tags' otomatis...")
    df['tags'] = df.apply(generate_tags, axis=1)

    # Simpan hasilnya ke file CSV baru
    df.to_csv(output_filename, index=False)
    print(f"Pra-pemrosesan selesai. Data disimpan di '{output_filename}'.")

if __name__ == "__main__":
    preprocess_and_create_tags()