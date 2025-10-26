# clear_data.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def clear_data():
    """
    Fungsi untuk menghapus data konten dan pengguna dengan peran 'member'.
    """
    print("Memulai proses pembersihan data...")
    load_dotenv()
    
    # --- KONEKSI DATABASE ---
    db_user, db_password, db_host, db_port, db_name = (
        os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_HOST"),
        os.getenv("DB_PORT"), os.getenv("DB_NAME")
    )
    DATABASE_URI = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(DATABASE_URI)

    # Daftar tabel yang akan dikosongkan sepenuhnya (TRUNCATE)
    tables_to_truncate = [
        'user_bookmarks',
        'menu_ratings',
        'menu_tags',
        'menus'
        # Tambahkan tabel lain di sini jika perlu, misal: 'comments'
    ]

    try:
        with engine.connect() as connection:
            print("Menonaktifkan foreign key checks...")
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            
            # 1. Hapus semua data konten
            for table in tables_to_truncate:
                print(f"Mengosongkan tabel `{table}`...")
                connection.execute(text(f"TRUNCATE TABLE `{table}`;"))

            # 2. Hapus pengguna dengan peran 'member' saja
            print("Menghapus pengguna dengan peran 'member'...")
            result = connection.execute(text("DELETE FROM users WHERE role = 'member';"))
            print(f"{result.rowcount} user 'member' telah dihapus.")
            
            print("Mengaktifkan kembali foreign key checks...")
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            connection.commit()
            
        print("\nPembersihan data selesai.")
    except Exception as e:
        print(f"Terjadi error: {e}")

if __name__ == "__main__":
    clear_data()