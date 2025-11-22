# app.py 

import sys
import os

# Fix numpy compatibility BEFORE any imports
import numpy as np
# Patch numpy._core to numpy.core for compatibility
if not hasattr(np, '_core'):
    np._core = np.core

import pandas as pd
from flask import Flask, jsonify, request
import pickle
import string 
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
    print(f"   Total resep: {len(recipes_df)}")
except FileNotFoundError as e:
    print(f"❌ ERROR: File model tidak ditemukan. Pastikan folder 'models' berisi 4 file pkl.")
    print(f"   Detail: {e}")
    indices = None
except Exception as e:
    print(f"❌ ERROR saat memuat model: {e}")
    print(f"   Tipe error: {type(e).__name__}")
    indices = None

# --- KONFIGURASI ---
SIMILARITY_THRESHOLD = 0.25  # Ambang batas kemiripan

# --- Endpoint 1: Rekomendasi Klik Resep (by Title) ---
@app.route("/recommend/title/<string:title_encoded>", methods=['GET'])
def recommend_by_title(title_encoded):
    if indices is None: 
        return jsonify({"error": "Model belum siap."}), 503
    
    try:
        from urllib.parse import unquote
        from sklearn.metrics.pairwise import cosine_similarity as calc_cosine
        
        # 1. Bersihkan Input
        title_raw = unquote(title_encoded)
        title_clean = text_cleaning(title_raw)
        
        print(f"🔍 Mencari rekomendasi untuk: {title_raw} (cleaned: {title_clean})")
        
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
            
            # Hanya ambil yang nilainya relevan (di atas threshold)
            if score > SIMILARITY_THRESHOLD:
                recommended_indices.append(i)
        
        # Urutkan dari skor tertinggi
        recommended_indices = sorted(recommended_indices, key=lambda i: sim_scores[i][1], reverse=True)
        
        print(f"✅ Ditemukan {len(recommended_indices)} rekomendasi")

        # 5. Return Data Lengkap
        result = recipes_df[['Title', 'Ingredients', 'Category']].iloc[recommended_indices]
        return jsonify(result.to_dict(orient='records'))

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500

# --- Endpoint 2: Rekomendasi Profil (Personalized) ---
@app.route("/recommend/profile", methods=['GET'])
def recommend_by_profile():
    if indices is None: 
        return jsonify({"error": "Model belum siap."}), 503

    titles_str = request.args.get('titles')
    if not titles_str: 
        return jsonify({"error": "Parameter 'titles' dibutuhkan"}), 400

    try:
        from urllib.parse import unquote
        from sklearn.metrics.pairwise import cosine_similarity as calc_cosine
        
        # 1. Parse Input
        raw_titles = [unquote(t) for t in titles_str.split(',')]
        profile_indices = []
        
        print(f"🔍 Mencari rekomendasi untuk {len(raw_titles)} resep favorit")
        
        for t in raw_titles:
            t_clean = text_cleaning(t)
            if t_clean in indices:
                idx = indices[t_clean]
                if isinstance(idx, pd.Series): idx = idx.iloc[0]
                profile_indices.append(idx)
        
        if not profile_indices: 
            return jsonify({"error": "Tidak ada resep favorit yang valid"}), 404

        print(f"✅ Valid: {len(profile_indices)} resep")

        # 2. Hitung Rata-rata Vektor User
        user_profile_vector = np.asarray(np.mean(tfidf_matrix[profile_indices], axis=0)).flatten()
        
        # 3. Hitung Kemiripan dengan Semua Resep
        sim_scores = calc_cosine(user_profile_vector.reshape(1, -1), tfidf_matrix)[0]
        
        # 4. Filter Hasil
        recommended_indices = []
        for i, score in enumerate(sim_scores):
            if i in profile_indices: continue # Skip yang sudah dilike
            
            if score > SIMILARITY_THRESHOLD:
                recommended_indices.append(i)
        
        # Urutkan
        recommended_indices = sorted(recommended_indices, key=lambda i: sim_scores[i], reverse=True)
        
        print(f"✅ Ditemukan {len(recommended_indices)} rekomendasi")
        
        # 5. Return Data Lengkap
        result = recipes_df[['Title', 'Ingredients', 'Category']].iloc[recommended_indices]
        return jsonify(result.to_dict(orient='records'))

    except Exception as e:
        print(f"❌ Error Profile: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "detail": str(e)}), 500

@app.route("/ping", methods=['GET'])
def ping(): return jsonify({"message": "pong!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)