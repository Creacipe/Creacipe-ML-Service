# evaluate.py

import pandas as pd
import pickle
import os

def evaluate_model(k=5):
    """
    Memuat model yang sudah dilatih dari folder 'models/'
    dan menghitung skor Precision@k.
    """
    print("Memulai evaluasi model...")
    
    try:
        # --- LANGKAH 1: MEMUAT MODEL & DATA ---
        # Memuat file-file yang dibutuhkan dari hasil training 'model.py'.
        df = pd.read_csv('dataset-final.csv')
        cosine_sim = pickle.load(open(os.path.join('models', 'cosine_sim.pkl'), 'rb'))
        print("Model dan data berhasil dimuat.")
    except FileNotFoundError:
        print("Error: File model di 'models/' atau dataset tidak ditemukan. Jalankan 'model.py' terlebih dahulu.")
        return

    # --- LANGKAH 2: PERSIAPAN EVALUASI ---
    total_precision = 0
    num_items = len(df)
    indices = pd.Series(df.index, index=df['Title'])

    # --- LANGKAH 3: MELAKUKAN EVALUASI UNTUK SETIAP RESEP ---
    # Looping melalui setiap resep di dataset.
    for idx in range(num_items):
        # Dapatkan kategori dari resep saat ini sebagai 'kunci jawaban'.
        current_category = df['category'].iloc[idx]
        
        # Ambil skor kemiripan untuk resep saat ini dari matriks.
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Urutkan dari yang paling mirip dan ambil 'k' teratas.
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
        
        recommended_indices = [i[0] for i in sim_scores]
        
        # Hitung berapa banyak rekomendasi yang 'benar' (memiliki kategori yang sama).
        correct_predictions = 0
        for i in recommended_indices:
            if df['category'].iloc[i] == current_category:
                correct_predictions += 1
        
        # Akumulasi skor presisi untuk item ini.
        total_precision += correct_predictions / k
        
    # Hitung rata-rata presisi dari semua resep.
    avg_precision = total_precision / num_items
    
    # --- LANGKAH 4: MENAMPILKAN HASIL EVALUASI ---
    print("\n--- Hasil Evaluasi Model ---")
    print(f"Rata-rata Precision@{k}: {avg_precision:.2f}")

    if avg_precision > 0.6:
        print("Kondisi Model: GOOD FITTING")
        print("Penjelasan: Rekomendasi yang diberikan sebagian besar relevan.")
    elif avg_precision > 0.3:
        print("Kondisi Model: POTENSI UNDERFITTING")
        print("Penjelasan: Rekomendasi terkadang relevan, tapi masih banyak yang kurang cocok.")
    else:
        print("Kondisi Model: UNDERFITTING")
        print("Penjelasan: Model kesulitan menemukan pola, rekomendasi cenderung tidak relevan.")

if __name__ == "__main__":
    evaluate_model(k=5)