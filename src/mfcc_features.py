# mfcc_features.py
# Muzik turu siniflandirmasi icin GERCEK ses ozellikleri cikarir.
# MFCC + Spectral Contrast + Chroma + ZCR + Spectral Centroid
# Bu ozellikler ImageNet ResNet'ten cok daha guvenilirdir.

import sqlite3
import os

import pandas as pd
import numpy as np
import librosa

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# -- Ozellik Parametreleri --
N_MFCC = 13          # Mel-Frequency Cepstral Coefficients
N_CONTRAST = 6       # Spectral contrast bands (7 Nyquist'i asar)
N_CHROMA = 12        # Chroma bins
TOP_DB = 20          # Sessizlik kirpma

# Toplam: 13 MFCC + 7 contrast + 12 chroma + 1 zcr + 1 centroid = 34 ozellik
MFCC_COLS = [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
CONTRAST_COLS = [f"contrast_{i}" for i in range(1, N_CONTRAST + 1)]
CHROMA_COLS = [f"chroma_{i}" for i in range(1, N_CHROMA + 1)]
OTHER_COLS = ["zcr", "centroid"]
ALL_AUDIO_COLS = MFCC_COLS + CONTRAST_COLS + CHROMA_COLS + OTHER_COLS


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_columns(conn: sqlite3.Connection) -> None:
    """34 yeni ozellik sutununu features tablosuna ekler."""
    for col in ALL_AUDIO_COLS:
        try:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} REAL;")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    print(f"[OK] {len(ALL_AUDIO_COLS)} ses ozelligi sutunu hazir.")


def extract_features(file_path: str) -> dict:
    """Bir ses dosyasindan 34 gercek muzik ozelligi cikarir.

    Returns:
        dict: mfcc_1..13, contrast_1..7, chroma_1..12, zcr, centroid
    """
    y, sr = librosa.load(file_path, duration=60)  # max 60 sn (hiz icin)
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)

    features = {}

    # 1. MFCC (13 katsayi) — tur ayriminda altin standart
    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}"] = float(np.mean(mfcc[i]))

    # 2. Spectral Contrast (7 band) — enstruman dokusu
    contrast = librosa.feature.spectral_contrast(y=yt, sr=sr, n_bands=N_CONTRAST, fmin=200.0)
    for i in range(N_CONTRAST):
        features[f"contrast_{i+1}"] = float(np.mean(contrast[i]))

    # 3. Chroma (12 nota) — armoni / akor yapisi
    chroma = librosa.feature.chroma_stft(y=yt, sr=sr)
    for i in range(N_CHROMA):
        features[f"chroma_{i+1}"] = float(np.mean(chroma[i]))

    # 4. ZCR — vurusculuk
    zcr = librosa.feature.zero_crossing_rate(yt)
    features["zcr"] = float(np.mean(zcr))

    # 5. Spectral Centroid — parlaklik
    cent = librosa.feature.spectral_centroid(y=yt, sr=sr)
    features["centroid"] = float(np.mean(cent))

    return features


def main() -> None:
    conn = get_connection()

    try:
        ensure_columns(conn)

        query = """
            SELECT s.id AS song_id, s.title AS title, s.file_path AS file_path
            FROM songs AS s
            INNER JOIN features AS f ON f.song_id = s.id;
        """
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        total = len(df)
        print(f"\nToplam {total} sarki icin 34 ses ozelligi cikarilacak.\n")

        processed = 0
        for _, row in df.iterrows():
            song_id = row["song_id"]
            title = row["title"]
            file_path = row["file_path"]

            print(f"  [{processed+1}/{total}] {title}")

            if not os.path.isfile(file_path):
                print(f"    [UYARI] Dosya bulunamadi, atlaniyor.")
                continue

            try:
                feats = extract_features(file_path)

                # UPDATE sorgusu olustur
                set_clause = ", ".join([f"{col} = ?" for col in ALL_AUDIO_COLS])
                values = [feats[col] for col in ALL_AUDIO_COLS] + [song_id]

                conn.execute(
                    f"UPDATE features SET {set_clause} WHERE song_id = ?;",
                    values,
                )
                conn.commit()
                processed += 1

            except Exception as e:
                print(f"    [HATA] {e}")
                continue

        print(f"\n{'=' * 60}")
        print(f"  MFCC + Spectral ozellik cikarimi tamamlandi!")
        print(f"  Islenen: {processed}/{total}")
        print(f"  Ozellik sayisi: {len(ALL_AUDIO_COLS)} (per sarki)")
        print(f"{'=' * 60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
