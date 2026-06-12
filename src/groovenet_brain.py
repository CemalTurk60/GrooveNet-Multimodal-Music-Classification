# groovenet_brain.py
# GrooveNet Beyni — TEK KOMUTLA tum pipeline'i calistirir.
#
# Kullanim:
#   1. Yeni sarkilari data/raw_audio/ klasorune at
#   2. py src/groovenet_brain.py
#   3. Bitti! Veritabanina eklendi, ozellikleri cikarildi, turu tahmin edildi.
#
# Pipeline sirasi:
#   [1] Ingestion     : raw_audio/ taranir, yeni dosyalar DB'ye eklenir
#   [2] Spektrogram   : Mel-Spektrogram gorsellerini uretir
#   [3] MFCC          : 33 ses ozelligi cikarir (genre siniflandirma icin)
#   [4] Genre Tahmin  : Kaydedilmis modelle tur tahmini yapar
#   [5] Rapor         : Tum sarkilari ture gore listeler

import sqlite3
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")
SPECTROGRAM_DIR = os.path.join(BASE_DIR, "data", "spectrograms")
MODEL_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "genre_classifier.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "checkpoints", "genre_scaler.joblib")
SUPPORTED_EXTENSIONS = {".mp3", ".wav"}

# -- Spektrogram Parametreleri --
FIGSIZE = (2.24, 2.24)
DPI_VAL = 100
N_MELS = 128
FMAX = 8000
TOP_DB = 20

# -- MFCC Parametreleri --
N_MFCC = 13
N_CONTRAST = 6
N_CHROMA = 12

import re

def _slugify(text):
    text = text.translate(str.maketrans(
        "çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU"
    ))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_tables(conn):
    """Gerekli tablolari olusturur (yoksa)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            title            TEXT,
            file_path        TEXT UNIQUE,
            duration_seconds REAL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id INTEGER UNIQUE,
            FOREIGN KEY (song_id) REFERENCES songs (id)
        );
    """)
    conn.commit()

    # Dinamik sutunlar (yoksa ekle)
    optional_cols = [
        ("spectrogram_path", "TEXT"),
        ("genre", "TEXT"),
    ]
    # MFCC sutunlari
    for i in range(1, N_MFCC + 1):
        optional_cols.append((f"mfcc_{i}", "REAL"))
    for i in range(1, N_CONTRAST + 1):
        optional_cols.append((f"contrast_{i}", "REAL"))
    for i in range(1, N_CHROMA + 1):
        optional_cols.append((f"chroma_{i}", "REAL"))
    optional_cols.append(("zcr", "REAL"))
    optional_cols.append(("centroid", "REAL"))

    for col, ctype in optional_cols:
        try:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} {ctype};")
        except sqlite3.OperationalError:
            pass
    conn.commit()


# ============================================================
#  ADIM 1: INGESTION — raw_audio/ taranir, DB'ye eklenir
# ============================================================
def step_ingestion(conn):
    from pathlib import Path

    print("\n  [ADIM 1/4] INGESTION — raw_audio/ taraniyor...")

    if not os.path.isdir(AUDIO_DIR):
        print(f"    [UYARI] Klasor bulunamadi: {AUDIO_DIR}")
        return 0

    audio_files = [
        e for e in Path(AUDIO_DIR).iterdir()
        if e.is_file() and e.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        print("    [UYARI] Ses dosyasi bulunamadi.")
        return 0

    cursor = conn.cursor()
    new_count = 0

    for entry in audio_files:
        title = entry.stem
        file_path = str(entry.resolve())
        cursor.execute(
            "INSERT OR IGNORE INTO songs (title, file_path) VALUES (?, ?)",
            (title, file_path),
        )
        if cursor.rowcount > 0:
            new_count += 1
            song_id = cursor.lastrowid
            cursor.execute(
                "INSERT OR IGNORE INTO features (song_id) VALUES (?)",
                (song_id,),
            )

    conn.commit()
    total = len(audio_files)
    print(f"    Toplam dosya: {total}  |  Yeni eklenen: {new_count}")
    return new_count


# ============================================================
#  ADIM 2: SPEKTROGRAM URETIMI
# ============================================================
def step_spectrograms(conn):
    print("\n  [ADIM 2/4] SPEKTROGRAM — eksik gorseller uretiliyor...")

    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

    query = """
        SELECT s.id, s.title, s.file_path, f.spectrogram_path
        FROM songs s
        INNER JOIN features f ON f.song_id = s.id
        WHERE f.spectrogram_path IS NULL OR f.spectrogram_path = '';
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("    Tum spektrogramlar zaten mevcut.")
        return 0

    count = 0
    for _, row in df.iterrows():
        song_id, title, file_path = row["id"], row["title"], row["file_path"]

        if not os.path.isfile(file_path):
            continue

        fname = f"id{song_id}_{_slugify(title)}.png"
        out_path = os.path.join(SPECTROGRAM_DIR, fname)

        try:
            y, sr = librosa.load(file_path)
            yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
            S = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=N_MELS, fmax=FMAX)
            S_dB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=DPI_VAL)
            librosa.display.specshow(S_dB, sr=sr, fmax=FMAX, ax=ax)
            ax.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            conn.execute(
                "UPDATE features SET spectrogram_path = ? WHERE song_id = ?;",
                (out_path, song_id),
            )
            conn.commit()
            count += 1
            print(f"    [{count}] {title}")
        except Exception as e:
            print(f"    [HATA] {title}: {e}")

    print(f"    {count} spektrogram uretildi.")
    return count


# ============================================================
#  ADIM 3: MFCC OZELLIK CIKARIMI
# ============================================================
def extract_mfcc_features(file_path):
    """33 ses ozelligi cikarir."""
    y, sr = librosa.load(file_path, duration=60)
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
    features = []

    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features.append(float(np.mean(mfcc[i])))

    contrast = librosa.feature.spectral_contrast(y=yt, sr=sr, n_bands=N_CONTRAST, fmin=200.0)
    for i in range(N_CONTRAST):
        features.append(float(np.mean(contrast[i])))

    chroma = librosa.feature.chroma_stft(y=yt, sr=sr)
    for i in range(N_CHROMA):
        features.append(float(np.mean(chroma[i])))

    zcr = librosa.feature.zero_crossing_rate(yt)
    features.append(float(np.mean(zcr)))

    cent = librosa.feature.spectral_centroid(y=yt, sr=sr)
    features.append(float(np.mean(cent)))

    return features


def step_mfcc(conn):
    print("\n  [ADIM 3/4] MFCC — ses ozellikleri cikariliyor...")

    # MFCC'si eksik olan sarkilari bul
    query = """
        SELECT s.id, s.title, s.file_path
        FROM songs s
        INNER JOIN features f ON f.song_id = s.id
        WHERE f.mfcc_1 IS NULL;
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("    Tum MFCC ozellikleri zaten cikarilmis.")
        return 0

    MFCC_COLS = [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
    CONTRAST_COLS = [f"contrast_{i}" for i in range(1, N_CONTRAST + 1)]
    CHROMA_COLS = [f"chroma_{i}" for i in range(1, N_CHROMA + 1)]
    ALL_COLS = MFCC_COLS + CONTRAST_COLS + CHROMA_COLS + ["zcr", "centroid"]

    count = 0
    for _, row in df.iterrows():
        song_id = row["id"]
        title = row["title"]
        file_path = row["file_path"]

        if not os.path.isfile(file_path):
            continue

        try:
            feats = extract_mfcc_features(file_path)
            set_clause = ", ".join([f"{c} = ?" for c in ALL_COLS])
            values = feats + [song_id]
            conn.execute(
                f"UPDATE features SET {set_clause} WHERE song_id = ?;",
                values,
            )
            conn.commit()
            count += 1
            print(f"    [{count}] {title}")
        except Exception as e:
            print(f"    [HATA] {title}: {e}")

    print(f"    {count} sarkinin MFCC ozellikleri cikarildi.")
    return count


# ============================================================
#  ADIM 4: GENRE TAHMINI
# ============================================================
def step_predict(conn):
    print("\n  [ADIM 4/4] TAHMIN — kaydedilmis modelle tur tahmini...")

    if not os.path.isfile(MODEL_PATH):
        print("    [UYARI] Model bulunamadi! Once egitim yapin:")
        print("    py src/train_model.py")
        return 0

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Genre'si olmayan sarkilari bul
    MFCC_COLS = [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
    CONTRAST_COLS = [f"contrast_{i}" for i in range(1, N_CONTRAST + 1)]
    CHROMA_COLS = [f"chroma_{i}" for i in range(1, N_CHROMA + 1)]
    ALL_COLS = MFCC_COLS + CONTRAST_COLS + CHROMA_COLS + ["zcr", "centroid"]

    cols_sql = ", ".join([f"f.{c}" for c in ALL_COLS])
    query = f"""
        SELECT s.id AS song_id, s.title AS title, s.file_path AS file_path,
               f.genre AS genre, {cols_sql}
        FROM songs s
        INNER JOIN features f ON f.song_id = s.id
        WHERE f.mfcc_1 IS NOT NULL;
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("    MFCC ozellikleri cikarilmis sarki bulunamadi.")
        return 0

    # Yeni sarkilar (genre yok veya BILINMIYOR)
    new_songs = df[
        (df["genre"].isna()) |
        (df["genre"] == "") |
        (df["genre"] == "BILINMIYOR")
    ]

    if new_songs.empty:
        print("    Yeni (etiketsiz) sarki yok — tum turler zaten atanmis.")
        return 0

    count = 0
    for _, row in new_songs.iterrows():
        song_id = row["song_id"]
        title = row["title"]

        X = np.array([row[ALL_COLS].values.astype(float)])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = max(proba) * 100

        conn.execute(
            "UPDATE features SET genre = ? WHERE song_id = ?;",
            (pred, song_id),
        )
        conn.commit()
        count += 1
        print(f"    [YENI] {title}")
        print(f"           Tahmin: {pred}  |  Guven: %{confidence:.0f}")

    print(f"    {count} yeni sarki siniflandirildi.")
    return count


# ============================================================
#  FINAL RAPOR
# ============================================================
def print_final_report(conn):
    query = """
        SELECT s.title, f.genre
        FROM songs s
        INNER JOIN features f ON f.song_id = s.id
        WHERE f.genre IS NOT NULL AND f.genre != '' AND f.genre != 'BILINMIYOR'
        ORDER BY f.genre, s.title;
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        return

    print(f"\n{'=' * 60}")
    print("  GROOVENET — TURE GORE SARKI KUTUPHANESI")
    print(f"{'=' * 60}")

    genres = df["genre"].unique()
    for genre in sorted(genres):
        songs = df[df["genre"] == genre]["title"].tolist()
        print(f"\n  {genre} ({len(songs)} sarki)")
        print(f"  {'-' * 40}")
        for s in songs:
            print(f"    - {s}")

    print(f"\n  Toplam: {len(df)} sarki  |  {len(genres)} tur")
    print(f"{'=' * 60}\n")


# ============================================================
#  MAIN — TEK KOMUTLA TUM PIPELINE
# ============================================================
def main():
    start = time.time()

    print()
    print("=" * 60)
    print("  GROOVENET BEYNI — OTOMATIK PIPELINE")
    print("  Sarki at, tek komut calistir, tur ogrensin!")
    print("=" * 60)

    conn = get_connection()

    try:
        ensure_tables(conn)

        new_ingested = step_ingestion(conn)
        new_spectrograms = step_spectrograms(conn)
        new_mfcc = step_mfcc(conn)
        new_predicted = step_predict(conn)

        elapsed = time.time() - start

        print(f"\n{'=' * 60}")
        print(f"  PIPELINE TAMAMLANDI ({elapsed:.1f} saniye)")
        print(f"  Yeni eklenen   : {new_ingested}")
        print(f"  Yeni spektro.  : {new_spectrograms}")
        print(f"  Yeni MFCC      : {new_mfcc}")
        print(f"  Yeni tahmin    : {new_predicted}")
        print(f"{'=' * 60}")

        print_final_report(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
