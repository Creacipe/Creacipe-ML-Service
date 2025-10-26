# model.py

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def build_model():
    """
    Fungsi ini membaca data resep yang sudah diproses,
    melatih model, dan menyimpan hasilnya.
    """
    print("Memulai proses pembuatan model dari file 'dataset-final.csv'...")
    
    try:
        # --- LANGKAH 1: MEMBACA DATA ---
        # Membaca dataset yang sudah bersih dan memiliki kolom 'tags'.
        df = pd.read_csv('dataset-final.csv')
        if 'id' not in df.columns:
            df['id'] = df.index + 1
        
        print(f"Berhasil membaca {len(df)} resep.")

        # --- LANGKAH 2: FEATURE ENGINEERING (MEMBENTUK INPUT) ---
        # Membersihkan nilai kosong (NaN) dari kolom yang akan digunakan.
        df['Ingredients'] = df['Ingredients'].fillna('')
        df['tags'] = df['tags'].fillna('')
        df['Title'] = df['Title'].fillna('')

        # Menggabungkan tiga kolom teks menjadi satu "fitur" besar.
        # Kolom inilah yang akan dianalisis oleh model.
        df['features'] = df['Title'] + ' ' + df['Ingredients'] + ' ' + df['tags']
        
        # --- LANGKAH 3: VEKTORISASI (MENGUBAH TEKS MENJADI ANGKA) ---
        # Membuat objek TfidfVectorizer untuk mengubah teks menjadi matriks angka.
        tfidf = TfidfVectorizer(stop_words='english')
        # Melatih dan mengubah kolom 'features' menjadi matriks TF-IDF.
        tfidf_matrix = tfidf.fit_transform(df['features'])
        
        # --- LANGKAH 4: MENGHITUNG KEMIRIPAN ---
        # Menghitung matriks kemiripan antar semua resep menggunakan Cosine Similarity.
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # --- LANGKAH 5: MENYIMPAN HASIL ---
        # Menyimpan objek-objek penting ke dalam folder 'models' agar bisa 
        # digunakan oleh server API (app.py) tanpa perlu training ulang.
        output_folder = 'models'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pickle.dump(df, open(os.path.join(output_folder, 'recipes.pkl'), 'wb')) 
        pickle.dump(cosine_sim, open(os.path.join(output_folder, 'cosine_sim.pkl'), 'wb'))
        pickle.dump(tfidf_matrix, open(os.path.join(output_folder, 'tfidf_matrix.pkl'), 'wb'))
        
        print(f"\nModel berhasil dibuat dan disimpan di dalam folder '{output_folder}'.")

    except FileNotFoundError:
        print("Error: File 'dataset-final.csv' tidak ditemukan. Jalankan 'preprocess_data.py' terlebih dahulu.")
    except Exception as e:
        print(f"Terjadi error: {e}")

if __name__ == "__main__":
    build_model()