# app.py 

import sys
import os

# Fix numpy compatibility
import numpy as np
if not hasattr(np, '_core'):
    np._core = np.core

import pandas as pd
from flask import Flask, jsonify, request
import pickle
import string 
import warnings
import traceback
from datetime import datetime
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- STORAGE UNTUK LOG EVALUASI REAL-TIME ---
# Menyimpan max 100 evaluasi terakhir
evaluation_logs = deque(maxlen=100)

# --- 1. FUNGSI HELPER ---
def text_cleaning(text):
    if isinstance(text, str):
        # Hapus tanda baca & ubah ke huruf kecil
        text = "".join([c for c in text if c not in string.punctuation])
        return text.lower().strip()
    return ''

# --- 2. MEMUAT MODEL (VERSI KNN) ---
model_loaded = False
recipes_df = None
tfidf_matrix = None
tfidf_vectorizer = None
knn_model = None
indices = None

try:
    model_folder = 'models'
    print("=" * 50)
    print("🔄 Sedang memuat model...")
    print(f"📁 Model folder: {os.path.abspath(model_folder)}")
    
    # Cek file yang ada di folder models
    if os.path.exists(model_folder):
        files = os.listdir(model_folder)
        print(f"📂 File dalam folder models: {files}")
    else:
        print(f"❌ Folder models tidak ditemukan!")
    
    # Load recipes.pkl
    recipes_path = os.path.join(model_folder, 'recipes.pkl')
    if os.path.exists(recipes_path):
        recipes_df = pickle.load(open(recipes_path, 'rb'))
        print(f"✅ recipes.pkl dimuat: {len(recipes_df)} resep")
        print(f"   Kolom: {list(recipes_df.columns)}")
    else:
        print(f"❌ recipes.pkl tidak ditemukan!")
    
    # Load tfidf_matrix.pkl
    tfidf_path = os.path.join(model_folder, 'tfidf_matrix.pkl')
    if os.path.exists(tfidf_path):
        tfidf_matrix = pickle.load(open(tfidf_path, 'rb'))
        print(f"✅ tfidf_matrix.pkl dimuat: shape = {tfidf_matrix.shape}")
    else:
        print(f"❌ tfidf_matrix.pkl tidak ditemukan!")
    
    # Load tfidf_vectorizer.pkl (optional untuk KNN)
    vectorizer_path = os.path.join(model_folder, 'tfidf_vectorizer.pkl')
    if os.path.exists(vectorizer_path):
        tfidf_vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        print(f"✅ tfidf_vectorizer.pkl dimuat")
    else:
        print(f"⚠️ tfidf_vectorizer.pkl tidak ditemukan (optional)")
    
    # Load knn_model.pkl
    knn_path = os.path.join(model_folder, 'knn_model.pkl')
    if os.path.exists(knn_path):
        knn_model = pickle.load(open(knn_path, 'rb'))
        print(f"✅ knn_model.pkl dimuat: {type(knn_model)}")
        print(f"   KNN params: n_neighbors={knn_model.n_neighbors}, metric={knn_model.metric}")
    else:
        print(f"❌ knn_model.pkl tidak ditemukan!")
        print(f"   → Akan mencoba buat KNN model dari tfidf_matrix...")
        # Fallback: Buat KNN model dari tfidf_matrix
        if tfidf_matrix is not None:
            from sklearn.neighbors import NearestNeighbors
            knn_model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
            knn_model.fit(tfidf_matrix)
            print(f"✅ KNN model dibuat secara dinamis!")
    
    # Mapping Index menggunakan 'Title_Clean'
    if recipes_df is not None:
        if 'Title_Clean' not in recipes_df.columns:
            print(f"⚠️ Kolom 'Title_Clean' tidak ada, membuat dari 'Title'...")
            recipes_df['Title_Clean'] = recipes_df['Title'].apply(text_cleaning)
        
        indices = pd.Series(recipes_df.index, index=recipes_df['Title_Clean']).drop_duplicates()
        print(f"✅ Indices dibuat: {len(indices)} unique titles")
        print(f"   Sample titles (cleaned): {list(indices.index[:5])}")
    
    # Cek semua komponen siap
    if recipes_df is not None and tfidf_matrix is not None and knn_model is not None and indices is not None:
        model_loaded = True
        print("=" * 50)
        print("✅ SEMUA MODEL BERHASIL DIMUAT!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("❌ Ada komponen model yang gagal dimuat!")
        print(f"   recipes_df: {'OK' if recipes_df is not None else 'MISSING'}")
        print(f"   tfidf_matrix: {'OK' if tfidf_matrix is not None else 'MISSING'}")
        print(f"   knn_model: {'OK' if knn_model is not None else 'MISSING'}")
        print(f"   indices: {'OK' if indices is not None else 'MISSING'}")
        print("=" * 50)

except Exception as e:
    print(f"❌ ERROR saat memuat model: {e}")
    traceback.print_exc()

# --- KONFIGURASI ---
DISTANCE_THRESHOLD = 0.8  # KNN jarak < 0.8 = mirip

# --- FUNGSI HITUNG METRICS (Untuk Admin Evaluation) ---
def calculate_metrics(target_category, recommended_indices, k=10):
    relevant_count = 0
    recommended_categories = recipes_df.iloc[recommended_indices]['Category'].tolist()
    
    for cat in recommended_categories:
        if cat == target_category:
            relevant_count += 1
            
    precision = relevant_count / k if k > 0 else 0
    total_relevant_in_db = len(recipes_df[recipes_df['Category'] == target_category])
    recall = relevant_count / total_relevant_in_db if total_relevant_in_db > 0 else 0
    
    return precision, recall, recommended_categories

# --- Endpoint 1: Rekomendasi Klik Resep (KNN) ---
@app.route("/recommend/title/<string:title_encoded>", methods=['GET'])
def recommend_by_title(title_encoded):
    print("\n" + "=" * 50)
    print(f"📥 REQUEST: /recommend/title/{title_encoded}")
    
    if not model_loaded:
        print("❌ Model belum siap!")
        return jsonify({"error": "Model belum siap."}), 503
    
    try:
        from urllib.parse import unquote
        title_raw = unquote(title_encoded)
        title_clean = text_cleaning(title_raw)
        
        print(f"🔍 Title raw: '{title_raw}'")
        print(f"🔍 Title clean: '{title_clean}'")
        
        # 1. Cek Database
        if title_clean not in indices:
            print(f"❌ Title tidak ditemukan dalam indices!")
            print(f"   Available sample: {list(indices.index[:10])}")
            return jsonify({"error": f"Resep '{title_raw}' tidak ditemukan."}), 404
        
        idx = indices[title_clean]
        if isinstance(idx, pd.Series): idx = idx.iloc[0]
        print(f"✅ Index ditemukan: {idx}")
        
        # 2. Hitung KNN
        target_vector = tfidf_matrix[idx]
        print(f"📊 Target vector shape: {target_vector.shape}")
        
        distances, neighbor_indices = knn_model.kneighbors(target_vector, n_neighbors=20)
        
        neighbor_indices = neighbor_indices.flatten()
        distances = distances.flatten()
        
        print(f"📊 KNN Results: {len(neighbor_indices)} neighbors")
        print(f"   Indices: {neighbor_indices[:10]}")
        print(f"   Distances: {distances[:10]}")
        
        # 3. Filter Hasil
        recommendations = []
        for i, neighbor_idx in enumerate(neighbor_indices):
            if neighbor_idx != idx and distances[i] < DISTANCE_THRESHOLD:
                recommendations.append(neighbor_idx)
        
        print(f"📊 After filter (threshold={DISTANCE_THRESHOLD}): {len(recommendations)} items")
        
        # 4. Ambil Judul
        if len(recommendations) > 0:
            titles = recipes_df['Title'].iloc[recommendations].tolist()
        else:
            titles = []
        
        # Filter nama yang persis sama
        title_raw_lower = title_raw.lower()
        filtered_titles = [t for t in titles if t.lower() != title_raw_lower]
        
        # 5. HITUNG EVALUASI METRICS DAN LOG
        if len(recommendations) > 0:
            target_category = recipes_df.iloc[idx]['Category']
            rec_indices_for_eval = recommendations[:10]  # Top 10 untuk evaluasi
            precision, recall, rec_categories = calculate_metrics(target_category, rec_indices_for_eval, k=len(rec_indices_for_eval))
            
            # Buat log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "similar_recipe",  # Tipe 1: Rekomendasi resep serupa
                "target_recipe": {
                    "title": title_raw,
                    "category": target_category
                },
                "metrics": {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "precision_display": f"{precision*100:.2f}",
                    "recall_display": f"{recall*100:.2f}",
                    "k": len(rec_indices_for_eval)
                },
                "recommendations_count": len(filtered_titles),
                "recommendations_analysis": [
                    {
                        "title": recipes_df['Title'].iloc[rec_indices_for_eval[i]],
                        "category": rec_categories[i],
                        "is_relevant": rec_categories[i] == target_category
                    }
                    for i in range(len(rec_indices_for_eval))
                ]
            }
            evaluation_logs.append(log_entry)
            print(f"📊 EVALUATION: Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
        
        print(f"✅ Final result: {len(filtered_titles)} recommendations")
        print(f"   Titles: {filtered_titles[:5]}...")  # Log only first 5
        print("=" * 50 + "\n")
        
        return jsonify(filtered_titles)  # Return ALL, frontend handles limit

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Endpoint 2: Rekomendasi Profil (KNN) ---
@app.route("/recommend/profile", methods=['GET'])
def recommend_by_profile():
    print("\n" + "=" * 50)
    print(f"📥 REQUEST: /recommend/profile")
    
    if not model_loaded:
        print("❌ Model belum siap!")
        return jsonify({"error": "Model belum siap."}), 503

    titles_str = request.args.get('titles')
    if not titles_str:
        print("❌ Parameter 'titles' tidak ada!")
        return jsonify({"error": "Parameter 'titles' dibutuhkan"}), 400

    try:
        from urllib.parse import unquote
        raw_titles = [unquote(t) for t in titles_str.split(',')]
        print(f"🔍 Input titles: {raw_titles}")
        
        profile_indices = []
        for t in raw_titles:
            t_clean = text_cleaning(t)
            if t_clean in indices:
                idx = indices[t_clean]
                if isinstance(idx, pd.Series): idx = idx.iloc[0]
                profile_indices.append(idx)
                print(f"   ✅ '{t}' -> index {idx}")
            else:
                print(f"   ❌ '{t}' tidak ditemukan")
        
        if not profile_indices:
            print("❌ Tidak ada resep favorit yang valid!")
            return jsonify({"error": "Tidak ada resep favorit yang valid"}), 404

        print(f"📊 Valid profile indices: {profile_indices}")

        # 1. Hitung Vektor Profil User (Rata-rata)
        user_profile_vector = np.asarray(np.mean(tfidf_matrix[profile_indices], axis=0)).reshape(1, -1)
        print(f"📊 User profile vector shape: {user_profile_vector.shape}")
        
        # 2. Cari Tetangga Terdekat
        distances, neighbor_indices = knn_model.kneighbors(user_profile_vector, n_neighbors=20)
        
        neighbor_indices = neighbor_indices.flatten()
        distances = distances.flatten()
        
        print(f"📊 KNN Results: {len(neighbor_indices)} neighbors")
        print(f"   Indices: {neighbor_indices[:10]}")
        print(f"   Distances: {distances[:10]}")
        
        recommendations = []
        for i, neighbor_idx in enumerate(neighbor_indices):
            if neighbor_idx not in profile_indices and distances[i] < DISTANCE_THRESHOLD:
                recommendations.append(neighbor_idx)
        
        print(f"📊 After filter: {len(recommendations)} items")
        
        if len(recommendations) > 0:
            titles = recipes_df['Title'].iloc[recommendations].tolist()
        else:
            titles = []
        
        # 5. HITUNG EVALUASI METRICS DAN LOG (untuk profile recommendation)
        if len(recommendations) > 0:
            # Untuk profile, gunakan kategori mayoritas dari profil user
            profile_categories = recipes_df.iloc[profile_indices]['Category'].tolist()
            target_category = max(set(profile_categories), key=profile_categories.count)  # Mayoritas
            
            rec_indices_for_eval = recommendations[:10]
            precision, recall, rec_categories = calculate_metrics(target_category, rec_indices_for_eval, k=len(rec_indices_for_eval))
            
            # Buat log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "personal_recommendation",  # Tipe 2: Rekomendasi personal
                "target_recipe": {
                    "title": f"Profil User ({len(raw_titles)} favorit)",
                    "category": target_category,
                    "favorites": raw_titles
                },
                "metrics": {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "precision_display": f"{precision*100:.2f}",
                    "recall_display": f"{recall*100:.2f}",
                    "k": len(rec_indices_for_eval)
                },
                "recommendations_count": len(titles),
                "recommendations_analysis": [
                    {
                        "title": recipes_df['Title'].iloc[rec_indices_for_eval[i]],
                        "category": rec_categories[i],
                        "is_relevant": rec_categories[i] == target_category
                    }
                    for i in range(len(rec_indices_for_eval))
                ]
            }
            evaluation_logs.append(log_entry)
            print(f"📊 EVALUATION: Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
        
        print(f"✅ Final result: {len(titles)} recommendations")
        print(f"   Titles: {titles[:5]}...")  # Log only first 5
        print("=" * 50 + "\n")
        
        return jsonify(titles)  # Return ALL, frontend handles limit

    except Exception as e:
        print(f"❌ Error Profile: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Endpoint BARU: Ambil Log Evaluasi Real-time ---
@app.route("/admin/logs", methods=['GET'])
def get_evaluation_logs():
    """
    Mengambil log evaluasi real-time.
    Query params:
    - limit (int): Jumlah log terakhir (default: 50)
    """
    try:
        limit = int(request.args.get('limit', 50))
        
        # Convert deque ke list dan reverse (terbaru dulu)
        logs = list(evaluation_logs)[::-1][:limit]
        
        # Hitung statistik keseluruhan
        if logs:
            total_precision = sum(log['metrics']['precision'] for log in logs)
            total_recall = sum(log['metrics']['recall'] for log in logs)
            avg_precision = (total_precision / len(logs)) * 100
            avg_recall = (total_recall / len(logs)) * 100
        else:
            avg_precision = 0
            avg_recall = 0
        
        return jsonify({
            "total_logs": len(evaluation_logs),
            "returned_logs": len(logs),
            "average_metrics": {
                "precision": f"{avg_precision:.2f}",
                "recall": f"{avg_recall:.2f}"
            },
            "logs": logs
        })
    
    except Exception as e:
        print(f"❌ Error getting logs: {e}")
        return jsonify({"error": str(e)}), 500

# --- Health Check ---
@app.route("/ping", methods=['GET'])
def ping():
    return jsonify({
        "message": "pong!",
        "model_loaded": model_loaded,
        "recipes_count": len(recipes_df) if recipes_df is not None else 0
    })

@app.route("/debug", methods=['GET'])
def debug_info():
    """Endpoint untuk debug - lihat status model"""
    return jsonify({
        "model_loaded": model_loaded,
        "recipes_df": "OK" if recipes_df is not None else "MISSING",
        "recipes_count": len(recipes_df) if recipes_df is not None else 0,
        "tfidf_matrix": f"shape={tfidf_matrix.shape}" if tfidf_matrix is not None else "MISSING",
        "knn_model": str(type(knn_model)) if knn_model is not None else "MISSING",
        "indices_count": len(indices) if indices is not None else 0,
        "sample_titles": list(indices.index[:5]) if indices is not None else [],
        "distance_threshold": DISTANCE_THRESHOLD
    })

if __name__ == "__main__":
    print("\n🚀 Starting ML Service on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)