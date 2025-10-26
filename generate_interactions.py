# generate_interactions.py

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import random

def generate_data():
    """
    Fungsi untuk membuat data interaksi (rating & bookmark) secara otomatis.
    """
    print("Memulai pembuatan data interaksi...")
    load_dotenv()
    
    # --- KONEKSI DATABASE ---
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    DATABASE_URI = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(DATABASE_URI)

    with engine.connect() as connection:
        # Hapus data interaksi lama agar tidak tumpang tindih
        connection.execute(text("TRUNCATE TABLE user_bookmarks;"))
        connection.execute(text("TRUNCATE TABLE menu_ratings;"))
        print("Tabel interaksi lama (bookmarks, ratings) telah dibersihkan.")

        # Ambil semua user (selain admin) dan semua menu
        users_df = pd.read_sql(text("SELECT user_id FROM users WHERE role = 'member'"), connection)
        menus_df = pd.read_sql(text("SELECT m.menu_id, GROUP_CONCAT(t.name) as tags FROM menus m JOIN menu_tags mt ON m.menu_id = mt.menu_id JOIN tags t ON mt.tag_id = t.tag_id GROUP BY m.menu_id"), connection)
        
        if len(users_df) == 0 or len(menus_df) == 0:
            print("Pastikan ada data di tabel 'users' dan 'menus'.")
            return
            
        # Definisikan "Persona" pengguna berdasarkan tag
        personas = {
            'pecinta_daging': ['Ayam', 'Sapi'],
            'suka_pedas': ['Pedas'],
            'suka_gorengan': ['Goreng'],
            'vegetarian': ['Sayuran', 'Tahu', 'Tempe'],
            'suka_kuah': ['Kuah', 'Rebus']
        }
        
        # Assign persona acak ke setiap user
        user_list = users_df['user_id'].tolist()
        user_personas = {user_id: random.choice(list(personas.keys())) for user_id in user_list}
        
        print(f"Menjalankan simulasi untuk {len(user_list)} pengguna...")
        
        all_bookmarks = []
        all_ratings = []

        # Lakukan simulasi untuk setiap pengguna
        for user_id, persona_name in user_personas.items():
            tags_selera = personas[persona_name]
            print(f"  - User {user_id} (Persona: {persona_name}) menyukai resep dengan tag: {', '.join(tags_selera)}")
            
            # Cari semua resep yang cocok dengan selera persona
            for index, menu in menus_df.iterrows():
                # Jika ada tag selera yang cocok di dalam resep
                if any(tag_selera in menu['tags'] for tag_selera in tags_selera):
                    # 50% kemungkinan akan di-bookmark
                    if random.random() < 0.5:
                        all_bookmarks.append({'user_id': user_id, 'menu_id': menu['menu_id']})
                    
                    # 70% kemungkinan akan diberi rating tinggi (4 atau 5)
                    if random.random() < 0.7:
                        all_ratings.append({'user_id': user_id, 'menu_id': menu['menu_id'], 'rating': random.randint(4, 5)})

        # Simpan semua data interaksi yang dihasilkan ke database
        if all_bookmarks:
            bookmarks_df = pd.DataFrame(all_bookmarks)
            bookmarks_df.to_sql('user_bookmarks', con=engine, if_exists='append', index=False)
            print(f"{len(all_bookmarks)} data bookmark berhasil dibuat.")
        
        if all_ratings:
            ratings_df = pd.DataFrame(all_ratings)
            ratings_df.to_sql('menu_ratings', con=engine, if_exists='append', index=False)
            print(f"{len(all_ratings)} data rating berhasil dibuat.")
            
    print("Pembuatan data interaksi selesai.")

if __name__ == "__main__":
    generate_data()