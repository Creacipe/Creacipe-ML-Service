# app.py (Versi Threshold)

import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import unquote

app = Flask(__name__)

# --- Memuat Model ---
try:
    model_folder = 'models'
    recipes_df = pickle.load(open(os.path.join(model_folder, 'recipes.pkl'), 'rb'))
    cosine_sim = pickle.load(open(os.path.join(model_folder, 'cosine_sim.pkl'), 'rb'))
    tfidf_matrix = pickle.load(open(os.path.join(model_folder, 'tfidf_matrix.pkl'), 'rb'))
    indices = pd.Series(recipes_df.index, index=recipes_df['Title']).drop_duplicates()
    print("Model (Title-based) berhasil dimuat.")
except FileNotFoundError:
    print("File model tidak ditemukan.")
    indices = None
except Exception as e:
    print(f"Error saat memuat model: {e}")
    indices = None

# --- KONFIGURASI THRESHOLD ---
SIMILARITY_THRESHOLD = 0.22 # Ambil resep dengan skor kemiripan > 0.2
# -----------------------------

# --- Endpoint Rekomendasi General (Resep Serupa by Title) ---
@app.route("/recommend/title/<string:title_encoded>", methods=['GET'])
def recommend_by_title(title_encoded):
    if indices is None: return jsonify({"error": "Model belum siap."}), 503
    try:
        title = unquote(title_encoded)
        # Ambil index, jika ada duplikat ambil yang pertama
        if title not in indices:
            return jsonify({"error": f"Resep dengan judul '{title}' tidak ditemukan."}), 404
        
        idx = indices[title]
        # Pastikan idx adalah integer, bukan Series
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # --- PERUBAHAN DI SINI ---
        # Filter skor yang di atas threshold (abaikan diri sendiri)
        recommended_indices = [i for i, score in sim_scores if score > SIMILARITY_THRESHOLD and i != idx]
        # Urutkan berdasarkan skor (opsional, tapi bagus)
        recommended_indices = sorted(recommended_indices, key=lambda i: sim_scores[i][1], reverse=True)
        # -------------------------

        return jsonify(recipes_df['Title'].iloc[recommended_indices].tolist())
    except KeyError:
        return jsonify({"error": f"Resep dengan judul '{title}' tidak ditemukan."}), 404
    except Exception as e:
        print(f"Error di /recommend/title/{title_encoded}: {e}")
        return jsonify({"error": "Terjadi kesalahan internal"}), 500

# --- Endpoint Rekomendasi Personal (by Titles) ---
@app.route("/recommend/profile", methods=['GET'])
def recommend_by_profile():
    if indices is None: return jsonify({"error": "Model belum siap."}), 503

    titles_str = request.args.get('titles')
    if not titles_str: return jsonify({"error": "Parameter 'titles' dibutuhkan"}), 400

    try:
        favorite_titles = [unquote(title) for title in titles_str.split(',')]
    except Exception as e:
         return jsonify({"error": "Format parameter 'titles' salah"}), 400

    try:
        profile_indices = []
        for title in favorite_titles:
            if title in indices:
                idx = indices[title]
                # Pastikan idx adalah integer
                if isinstance(idx, pd.Series):
                    idx = idx.iloc[0]
                profile_indices.append(idx)
        
        if not profile_indices: 
            return jsonify({"error": "Tidak ada resep favorit yang valid"}), 404

        user_profile_vector = np.asarray(np.mean(tfidf_matrix[profile_indices], axis=0)).flatten()
        sim_scores = cosine_similarity(user_profile_vector.reshape(1, -1), tfidf_matrix)
        sim_scores = list(enumerate(sim_scores[0]))

        # --- PERUBAHAN DI SINI ---
        # Filter skor di atas threshold DAN yang bukan favorit user
        recommended_menu_indices = []
        for i, score in sim_scores:
             if score > SIMILARITY_THRESHOLD and i < len(recipes_df): # Pastikan index valid
                 # Cek apakah judul rekomendasi tidak ada di daftar favorit
                 if recipes_df['Title'].iloc[i] not in favorite_titles:
                     recommended_menu_indices.append(i)
        # Urutkan berdasarkan skor (opsional)
        recommended_menu_indices = sorted(recommended_menu_indices, key=lambda i: sim_scores[i][1], reverse=True)
        # -------------------------

        return jsonify(recipes_df['Title'].iloc[recommended_menu_indices].tolist())
    except Exception as e:
        print(f"Error di /recommend/profile: {e}")
        return jsonify({"error": "Terjadi kesalahan internal saat membuat profil"}), 500

# --- Endpoint Lainnya ---
@app.route("/ping", methods=['GET'])
def ping(): return jsonify({"message": "pong!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)