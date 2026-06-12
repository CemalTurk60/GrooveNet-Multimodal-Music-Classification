# predict.py
# Kaydedilmis modeli yukleyip YENI sarkilarin turunu tahmin eder.
# Kullanim: data/raw_audio/ klasorune yeni sarkiyi at, bu scripti calistir.
# Model egitilmis olmali (once train_model.py calistirilmali).

import sqlite3
import os

import joblib
import pandas as pd
import numpy as np
import librosa

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
MODEL_PATH = os.path.join(MODEL_DIR, "genre_classifier.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "genre_scaler.joblib")

# -- Ozellik Parametreleri (mfcc_features.py ile ayni) --
N_MFCC = 13
N_CONTRAST = 6
N_CHROMA = 12
TOP_DB = 20


def extract_features(file_path: str) -> np.ndarray:
    """Bir ses dosyasindan 34 ozellik cikarir (mfcc_features.py ile ayni)."""
    y, sr = librosa.load(file_path, duration=60)
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features.append(float(np.mean(mfcc[i])))

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=yt, sr=sr, n_bands=N_CONTRAST, fmin=200.0)
    for i in range(N_CONTRAST):
        features.append(float(np.mean(contrast[i])))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=yt, sr=sr)
    for i in range(N_CHROMA):
        features.append(float(np.mean(chroma[i])))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(yt)
    features.append(float(np.mean(zcr)))

    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=yt, sr=sr)
    features.append(float(np.mean(cent)))

    return np.array(features).reshape(1, -1)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def main() -> None:
    # -- Model yukle --
    if not os.path.isfile(MODEL_PATH):
        print("Model dosyasi bulunamadi!")
        print("Once egitim yapin: py src/train_model.py")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Model ve scaler yuklendi.")

    # -- Veritabanindan genre'si olmayan (yeni) sarkilari bul --
    conn = get_connection()

    try:
        query = """
            SELECT s.id AS song_id, s.title AS title, s.file_path AS file_path,
                   f.genre AS genre
            FROM songs AS s
            INNER JOIN features AS f ON f.song_id = s.id;
        """
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        # Genre'si olmayan veya BILINMIYOR olan sarkilar = yeni
        new_songs = df[
            (df["genre"].isna()) |
            (df["genre"] == "BILINMIYOR") |
            (df["genre"] == "")
        ]

        # Tum sarkilari da goster
        labeled_songs = df[
            (df["genre"].notna()) &
            (df["genre"] != "BILINMIYOR") &
            (df["genre"] != "")
        ]

        print()
        print("=" * 60)
        print("  GROOVENET — TUR TAHMINI (Predict)")
        print("=" * 60)
        print(f"  Etiketli sarki    : {len(labeled_songs)}")
        print(f"  Yeni (etiketsiz)  : {len(new_songs)}")
        print(f"  Toplam            : {len(df)}")
        print("=" * 60)

        if len(new_songs) == 0:
            print("\n  Yeni (etiketsiz) sarki bulunamadi.")
            print("  Tum sarkilarin genre etiketi zaten atanmis.\n")

            # Mevcut sarkilarin tahminlerini goster
            print("  Mevcut sarkilarin model tahminleri:\n")
            for _, row in labeled_songs.iterrows():
                file_path = row["file_path"]
                title = row["title"]
                actual = row["genre"]

                if not os.path.isfile(file_path):
                    continue

                try:
                    X = extract_features(file_path)
                    X_scaled = scaler.transform(X)
                    pred = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = max(proba) * 100

                    match = "OK" if pred == actual else "XX"
                    print(f"  [{match}] {title}")
                    print(f"       Gercek: {actual:10s}  |  "
                          f"Tahmin: {pred:10s}  |  "
                          f"Guven: %{confidence:.0f}")
                except Exception as e:
                    print(f"  [HATA] {title}: {e}")

        else:
            print(f"\n  Yeni sarkilar icin tur tahmini:\n")
            for _, row in new_songs.iterrows():
                song_id = row["song_id"]
                title = row["title"]
                file_path = row["file_path"]

                if not os.path.isfile(file_path):
                    print(f"  [UYARI] Dosya bulunamadi: {title}")
                    continue

                try:
                    X = extract_features(file_path)
                    X_scaled = scaler.transform(X)
                    pred = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = max(proba) * 100

                    print(f"  [YENI] {title}")
                    print(f"         Tahmin: {pred}  |  Guven: %{confidence:.0f}")

                    # Genre'yi DB'ye kaydet
                    conn.execute(
                        "UPDATE features SET genre = ? WHERE song_id = ?;",
                        (pred, song_id),
                    )
                    conn.commit()

                except Exception as e:
                    print(f"  [HATA] {title}: {e}")

        print(f"\n{'=' * 60}")
        print("  Tahmin tamamlandi!")
        print(f"{'=' * 60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
