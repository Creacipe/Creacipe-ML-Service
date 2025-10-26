# model.py (Final - Membaca dari dataset/train_data.csv)

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def build_model():
    """
    Membaca data training (dataset/train_data.csv), melatih model,
    dan menyimpan hasilnya ke dalam folder 'models/'.
    """
    # --- LANGKAH 1: TENTUKAN PATH & BACA DATA TRAINING ---
    dataset_folder = 'dataset'
    train_file = os.path.join(dataset_folder, 'train_data.csv') # Path ke data training

    print(f"Memulai proses pembuatan model dari '{train_file}'...")

    if not os.path.exists(train_file):
        print(f"Error: File training '{train_file}' tidak ditemukan. Jalankan 'split_data.py' terlebih dahulu.")
        return

    try:
        df = pd.read_csv(train_file)
        if 'id' not in df.columns:
            df['id'] = df.index + 1 # Buat ID sementara jika tidak ada
        print(f"Berhasil membaca {len(df)} resep dari data training.")

        # --- LANGKAH 2: FEATURE ENGINEERING ---
        # Membersihkan nilai kosong.
        df['Ingredients'] = df['Ingredients'].fillna('')
        # Kolom 'tags' seharusnya sudah dibuat oleh preprocess_data.py sebelum split
        df['tags'] = df['tags'].fillna('')
        df['Title'] = df['Title'].fillna('')

        # Menggabungkan fitur teks: Title, Ingredients, Tags.
        df['features'] = df['Title'] + ' ' + df['Ingredients'] + ' ' + df['tags']

        # --- LANGKAH 3: VEKTORISASI (LATIH TF-IDF) ---
        tfidf = TfidfVectorizer(stop_words='english')
        # Melatih TF-IDF *hanya* pada data training.
        tfidf_matrix = tfidf.fit_transform(df['features'])
        print("TF-IDF vectorizer dilatih pada data training.")

        # --- LANGKAH 4: HITUNG KEMIRIPAN ---
        # Menghitung matriks kemiripan antar resep di data training.
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("Matriks kemiripan dihitung.")

        # --- LANGKAH 5: MENYIMPAN HASIL KE FOLDER 'models' ---
        output_folder = 'models'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Simpan DataFrame training (termasuk ID dan fitur), matriks kemiripan,
        # matriks TF-IDF, dan objek TF-IDF vectorizer itu sendiri.
        pickle.dump(df, open(os.path.join(output_folder, 'recipes.pkl'), 'wb'))
        pickle.dump(cosine_sim, open(os.path.join(output_folder, 'cosine_sim.pkl'), 'wb'))
        pickle.dump(tfidf_matrix, open(os.path.join(output_folder, 'tfidf_matrix.pkl'), 'wb'))
        pickle.dump(tfidf, open(os.path.join(output_folder, 'tfidf_vectorizer.pkl'), 'wb'))

        print(f"\nModel berhasil dibuat dari data training dan disimpan di folder '{output_folder}'.")

    except Exception as e:
        print(f"Terjadi error saat membuat model: {e}")

if __name__ == "__main__":
    build_model()