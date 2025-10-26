# evaluate.py (Final - Dengan Perbandingan Train vs Test)

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_precision_at_k(df_to_evaluate, df_source, cosine_sim_matrix, indices_source, k=5):
    """
    Fungsi helper untuk menghitung Precision@k pada dataset yang diberikan.
    df_to_evaluate: DataFrame yang akan dievaluasi (bisa train atau test).
    df_source: DataFrame sumber (training data, tempat rekomendasi diambil).
    cosine_sim_matrix: Matriks kemiripan yang sesuai (bisa train vs train atau test vs train).
    indices_source: Mapping Title ke index untuk df_source.
    k: Jumlah rekomendasi teratas yang akan dievaluasi.
    """
    total_precision = 0
    evaluated_count = 0

    # Looping melalui setiap item di dataset yang akan dievaluasi
    for idx_eval in range(len(df_to_evaluate)):
        try:
            # Ambil skor kemiripan item ini terhadap semua item di sumber
            sim_scores = list(enumerate(cosine_sim_matrix[idx_eval]))
            # Urutkan
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Ambil K rekomendasi teratas (ini adalah INDEKS dari df_source)
            # Jika evaluasi di training set, abaikan diri sendiri (index 0)
            if df_to_evaluate is df_source:
                 recommended_source_indices = [i[0] for i in sim_scores[1:k+1]]
            else: # Jika evaluasi di test set, ambil k teratas
                 recommended_source_indices = [i[0] for i in sim_scores[:k]]

            # Dapatkan kategori asli dari item yang sedang dievaluasi
            current_category = df_to_evaluate['category'].iloc[idx_eval]

            # Hitung presisi
            correct_predictions = 0
            for source_idx in recommended_source_indices:
                recommended_category = df_source['category'].iloc[source_idx]
                if recommended_category == current_category:
                    correct_predictions += 1

            total_precision += correct_predictions / k
            evaluated_count += 1
        except Exception as e:
            # Beri tahu jika ada error saat evaluasi item spesifik
            print(f"Error saat mengevaluasi item index {idx_eval}: {e}")

    if evaluated_count == 0:
        return 0 # Hindari pembagian dengan nol

    return total_precision / evaluated_count


def evaluate_model_train_test(k=5):
    """
    Memuat model, data train/test, dan menghitung Precision@k pada keduanya.
    """
    print("Memulai evaluasi model pada Training dan Testing Set...")

    # --- LANGKAH 1: TENTUKAN PATH FILE ---
    dataset_folder = 'dataset'
    model_folder = 'models'
    train_file = os.path.join(dataset_folder, 'train_data.csv')
    test_file = os.path.join(dataset_folder, 'test_data.csv')

    try:
        # --- LANGKAH 2: MEMUAT KOMPONEN MODEL & DATA ---
        train_df = pd.read_csv(train_file) # Membaca data training
        test_df = pd.read_csv(test_file)   # Membaca data testing
        
        # Memuat model yang dilatih HANYA pada data training
        cosine_sim_train = pickle.load(open(os.path.join(model_folder, 'cosine_sim.pkl'), 'rb'))
        tfidf_matrix_train = pickle.load(open(os.path.join(model_folder, 'tfidf_matrix.pkl'), 'rb'))
        tfidf_vectorizer = pickle.load(open(os.path.join(model_folder, 'tfidf_vectorizer.pkl'), 'rb'))
        # Pastikan data training dari pickle cocok (opsional, tapi bagus untuk verifikasi)
        # loaded_train_df = pickle.load(open(os.path.join(model_folder, 'recipes.pkl'), 'rb'))
        
        print("Model dan data train/test berhasil dimuat.")

    except FileNotFoundError:
        print(f"Error: Pastikan file model di '{model_folder}/' serta '{train_file}' dan '{test_file}' ada.")
        return
    except Exception as e:
        print(f"Error saat memuat data/model: {e}")
        return

    try:
        # --- LANGKAH 3: EVALUASI PADA TRAINING SET ---
        print("\nMengevaluasi pada Training Set...")
        # Membuat mapping Title -> index untuk data training
        train_indices = pd.Series(train_df.index, index=train_df['Title']).drop_duplicates()
        # Hitung presisi train vs train
        avg_precision_train = calculate_precision_at_k(train_df, train_df, cosine_sim_train, train_indices, k=k)
        print(f"  - Rata-rata Precision@{k} (Training): {avg_precision_train:.2f}")

        # --- LANGKAH 4: PERSIAPAN & EVALUASI PADA TESTING SET ---
        print("\nMengevaluasi pada Testing Set...")
        # Feature engineering pada test set (sama seperti di model.py)
        test_df['Ingredients'] = test_df['Ingredients'].fillna('')
        test_df['tags'] = test_df['tags'].fillna('')
        test_df['Title'] = test_df['Title'].fillna('')
        test_df['features'] = test_df['Title'] + ' ' + test_df['Ingredients'] + ' ' + test_df['tags']
        
        # Transform test set menggunakan vectorizer yang sudah dilatih
        tfidf_matrix_test = tfidf_vectorizer.transform(test_df['features'])
        
        # Hitung kemiripan test vs train
        cosine_sim_test_vs_train = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
        
        # Hitung presisi test vs train
        avg_precision_test = calculate_precision_at_k(test_df, train_df, cosine_sim_test_vs_train, train_indices, k=k)
        print(f"  - Rata-rata Precision@{k} (Testing): {avg_precision_test:.2f}")

        # --- LANGKAH 5: INTERPRETASI HASIL ---
        print("\n--- Kesimpulan Kondisi Model ---")
        # Logika untuk mendeteksi overfitting/underfitting/good fitting
        if avg_precision_test < 0.3: # Jika performa di data baru sangat rendah
            print("Kondisi Model: UNDERFITTING")
            print("Penjelasan: Model gagal mempelajari pola yang cukup baik, performa rendah di data training dan testing.")
        elif avg_precision_train > avg_precision_test + 0.15: # Jika ada penurunan signifikan dari train ke test
            print("Kondisi Model: OVERFITTING")
            print("Penjelasan: Model terlalu 'hafal' data training (skor tinggi) tapi performanya turun drastis pada data baru (testing).")
        elif avg_precision_test > 0.6: # Jika performa di data baru bagus dan tidak jauh dari training
             print("Kondisi Model: GOOD FITTING")
             print("Penjelasan: Model mampu generalisasi dengan baik pada data baru.")
        else: # Jika performa di data baru sedang (tidak underfitting, tidak overfitting)
             print("Kondisi Model: Cukup Baik (Perlu Dipantau)")
             print("Penjelasan: Model belajar, tapi mungkin bisa ditingkatkan lagi.")

    except Exception as e:
        print(f"Terjadi error saat evaluasi: {e}")

if __name__ == "__main__":
    evaluate_model_train_test(k=5) # Gunakan nama yang benar