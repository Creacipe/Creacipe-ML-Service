# app.py

import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# --- LANGKAH 1: MEMUAT MODEL SAAT SERVER DIMULAI ---
# Model dimuat sekali saja di awal untuk performa yang cepat.
try:
    model_folder = 'models'
    recipes_df = pickle.load(open(os.path.join(model_folder, 'recipes.pkl'), 'rb'))
    cosine_sim = pickle.load(open(os.path.join(model_folder, 'cosine_sim.pkl'), 'rb'))
    tfidf_matrix = pickle.load(open(os.path.join(model_folder, 'tfidf_matrix.pkl'), 'rb'))
    # Mapping antara ID resep dan indeksnya di DataFrame.
    indices = pd.Series(recipes_df.index, index=recipes_df['id']).drop_duplicates()
    print("Model rekomendasi berhasil dimuat.")
except FileNotFoundError:
    print(f"File model di folder '{model_folder}/' tidak ditemukan. Jalankan 'python model.py' terlebih dahulu.")
    indices = None

# --- LANGKAH 2: MEMBUAT ENDPOINT API ---

# Endpoint untuk rekomendasi general (resep serupa).
@app.route("/recommend/<int:menu_id>", methods=['GET'])
def recommend(menu_id):
    if indices is None: return jsonify({"error": "Model belum siap."}), 503
    try:
        # Cari indeks resep berdasarkan menu_id.
        idx = indices[menu_id]
        # Ambil skor kemiripan untuk resep tersebut.
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Urutkan dan ambil 10 teratas.
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        menu_indices = [i[0] for i in sim_scores]
        # Kembalikan daftar ID resep yang direkomendasikan.
        return jsonify(recipes_df['id'].iloc[menu_indices].tolist())
    except KeyError:
        return jsonify({"error": f"Resep dengan ID {menu_id} tidak ditemukan."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk rekomendasi personal (profil pengguna).
@app.route("/recommend/profile", methods=['GET'])
def recommend_by_profile():
    if indices is None: return jsonify({"error": "Model belum siap."}), 503

    # Ambil daftar ID resep favorit dari parameter URL.
    ids_str = request.args.get('ids')
    if not ids_str: return jsonify({"error": "Parameter 'ids' dibutuhkan"}), 400
    
    try:
        favorite_ids = [int(id) for id in ids_str.split(',')]
    except ValueError:
        return jsonify({"error": "Parameter 'ids' harus berupa angka"}), 400

    try:
        # Cari indeks dari resep-resep favorit.
        profile_indices = [indices[menu_id] for menu_id in favorite_ids if menu_id in indices]
        if not profile_indices: return jsonify({"error": "Tidak ada resep favorit yang valid"}), 404
            
        # Buat "profil selera" dengan merata-ratakan vektor TF-IDF dari resep favorit.
        user_profile_vector = np.asarray(np.mean(tfidf_matrix[profile_indices], axis=0)).flatten()
        
        # Hitung kemiripan profil selera dengan semua resep lain.
        sim_scores = cosine_similarity(user_profile_vector.reshape(1, -1), tfidf_matrix)
        sim_scores = list(enumerate(sim_scores[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Ambil 10 rekomendasi teratas, filter resep yang sudah disukai.
        recommended_menu_indices = []
        count = 0
        for i, score in sim_scores:
            if recipes_df['id'].iloc[i] not in favorite_ids:
                recommended_menu_indices.append(i)
                count += 1
            if count == 10:
                break
        
        return jsonify(recipes_df['id'].iloc[recommended_menu_indices].tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk mengecek apakah server berjalan.
@app.route("/ping", methods=['GET'])
def ping():
    return jsonify({"message": "pong!"})

# --- LANGKAH 3: MENJALANKAN SERVER ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)