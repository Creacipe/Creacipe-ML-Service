# app.py 

import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np
import os
import string 
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import unquote

app = Flask(__name__)

# --- 1. FUNGSI HELPER (WAJIB SAMA DENGAN COLAB) ---
def text_cleaning(text):
    if isinstance(text, str):
        # Hapus tanda baca & ubah ke huruf kecil
        text = "".join([c for c in text if c not in string.punctuation])
        return text.lower().strip()
    return ''

# --- 2. MEMUAT MODEL ---
try:
    model_folder = 'models'
    print("Sedang memuat model...")
    
    # Load DataFrame, Cosine Sim, dan TF-IDF Matrix
    recipes_df = pickle.load(open(os.path.join(model_folder, 'recipes.pkl'), 'rb'))
    cosine_sim = pickle.load(open(os.path.join(model_folder, 'cosine_sim.pkl'), 'rb'))
    tfidf_matrix = pickle.load(open(os.path.join(model_folder, 'tfidf_matrix.pkl'), 'rb'))
    
    # Mapping Index menggunakan 'Title_Clean'
    indices = pd.Series(recipes_df.index, index=recipes_df['Title_Clean']).drop_duplicates()
    
    print("✅ Model berhasil dimuat.")
except FileNotFoundError:
    print("❌ ERROR: File model tidak ditemukan. Pastikan folder 'models' berisi 4 file pkl.")
    indices = None
except Exception as e:
    print(f"❌ ERROR: {e}")
    indices = None

# --- KONFIGURASI ---
SIMILARITY_THRESHOLD = 0.25  # Ambang batas kemiripan

# --- Endpoint 1: Rekomendasi Klik Resep (by Title) ---
@app.route("/recommend/title/<string:title_encoded>", methods=['GET'])
def recommend_by_title(title_encoded):
    if indices is None: return jsonify({"error": "Model belum siap."}), 503
    
    try:
        # 1. Bersihkan Input
        title_raw = unquote(title_encoded)
        title_clean = text_cleaning(title_raw)
        
        # 2. Cek Database
        if title_clean not in indices:
            return jsonify({"error": f"Resep '{title_raw}' tidak ditemukan."}), 404
        
        idx = indices[title_clean]
        if isinstance(idx, pd.Series): idx = idx.iloc[0]
        
        # 3. Hitung Kemiripan
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # 4. Filter & Sort
        recommended_indices = []
        for i, score in sim_scores:
            if i == idx: continue # Skip diri sendiri
            
            # Hanya ambil yang nilainya relevan (di atas 0.2)
            if score > SIMILARITY_THRESHOLD:
                recommended_indices.append(i)
        
        # Urutkan dari skor tertinggi
        recommended_indices = sorted(recommended_indices, key=lambda i: sim_scores[i][1], reverse=True)
        
        # CATATAN: Tidak ada slicing [:20] di sini.
        # Semua hasil dikirim ke Golang.

        # 5. Return Data Lengkap
        result = recipes_df[['Title', 'Ingredients', 'Category']].iloc[recommended_indices]
        return jsonify(result.to_dict(orient='records'))

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# --- Endpoint 2: Rekomendasi Profil (Personalized) ---
@app.route("/recommend/profile", methods=['GET'])
def recommend_by_profile():
    if indices is None: return jsonify({"error": "Model belum siap."}), 503

    titles_str = request.args.get('titles')
    if not titles_str: return jsonify({"error": "Parameter 'titles' dibutuhkan"}), 400

    try:
        # 1. Parse Input
        raw_titles = [unquote(t) for t in titles_str.split(',')]
        profile_indices = []
        
        for t in raw_titles:
            t_clean = text_cleaning(t)
            if t_clean in indices:
                idx = indices[t_clean]
                if isinstance(idx, pd.Series): idx = idx.iloc[0]
                profile_indices.append(idx)
        
        if not profile_indices: 
            return jsonify({"error": "Tidak ada resep favorit yang valid"}), 404

        # 2. Hitung Rata-rata Vektor User
        user_profile_vector = np.asarray(np.mean(tfidf_matrix[profile_indices], axis=0)).flatten()
        
        # 3. Hitung Kemiripan dengan Semua Resep
        from sklearn.metrics.pairwise import cosine_similarity
        sim_scores = cosine_similarity(user_profile_vector.reshape(1, -1), tfidf_matrix)[0]
        
        # 4. Filter Hasil
        recommended_indices = []
        for i, score in enumerate(sim_scores):
            if i in profile_indices: continue # Skip yang sudah dilike
            
            if score > SIMILARITY_THRESHOLD:
                recommended_indices.append(i)
        
        # Urutkan
        recommended_indices = sorted(recommended_indices, key=lambda i: sim_scores[i], reverse=True)
        
        # CATATAN: Tidak ada slicing [:20]. Semua dikirim.
        
        # 5. Return Data Lengkap
        result = recipes_df[['Title', 'Ingredients', 'Category']].iloc[recommended_indices]
        return jsonify(result.to_dict(orient='records'))

    except Exception as e:
        print(f"Error Profile: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/ping", methods=['GET'])
def ping(): return jsonify({"message": "pong!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)