# acoustic_features.py
# Librosa kullanarak muziklerin temel akustik ve fiziksel (DSP)
# ozelliklerini cikarir: BPM, RMS, ZCR, Onset Strength, Spectral Centroid.
# Sonuclar features tablosuna kaydedilir.

import sqlite3
import os

import pandas as pd
import numpy as np
import librosa

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# -- DSP Parametreleri --
TOP_DB = 20  # sessizlik kirpma esigi (dB)


def get_connection() -> sqlite3.Connection:
    """SQLite veritabanina baglanti acar."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_acoustic_columns(conn: sqlite3.Connection) -> None:
    """features tablosuna akustik ozellik sutunlarini ekler (yoksa)."""
    columns = [
        ("bpm", "REAL"),
        ("rms_mean", "REAL"),
        ("rms_var", "REAL"),
        ("zcr_mean", "REAL"),
        ("onset_strength", "REAL"),
        ("spectral_centroid", "REAL"),
    ]
    for col, col_type in columns:
        try:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} {col_type};")
            conn.commit()
            print(f"  [OK] '{col}' sutunu eklendi.")
        except sqlite3.OperationalError:
            pass
    print("[BILGI] Akustik ozellik sutunlari hazir.\n")


def fetch_songs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Veritabanindan song_id, title ve dosya yollarini ceker."""
    query = """
        SELECT
            s.id        AS song_id,
            s.title     AS title,
            s.file_path AS file_path
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id;
    """
    df = pd.read_sql_query(query, conn)
    return df


def extract_acoustic_features(file_path: str) -> dict:
    """Bir ses dosyasindan 6 temel akustik ozellik cikarir.

    Args:
        file_path: Ses dosyasinin tam yolu.

    Returns:
        dict: bpm, rms_mean, rms_var, zcr_mean, onset_strength, spectral_centroid
    """
    # Sesi yukle
    y, sr = librosa.load(file_path)

    # Sessizlikleri kirp
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # 1. BPM
    tempo, _ = librosa.beat.beat_track(y=yt, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # 2. RMS (Enerji Ortalamasi ve Varyansi)
    rms = librosa.feature.rms(y=yt)
    rms_mean = float(np.mean(rms))
    rms_var = float(np.var(rms))

    # 3. ZCR (Sifir Gecis Orani)
    zcr = librosa.feature.zero_crossing_rate(yt)
    zcr_mean_val = float(np.mean(zcr))

    # 4. Onset Strength (Vurus Sertligi)
    onset_env = librosa.onset.onset_strength(y=yt, sr=sr)
    onset_str = float(np.mean(onset_env))

    # 5. Spectral Centroid (Parlaklik/Yirticlik)
    cent = librosa.feature.spectral_centroid(y=yt, sr=sr)
    spec_centroid = float(np.mean(cent))

    return {
        "bpm": bpm,
        "rms_mean": rms_mean,
        "rms_var": rms_var,
        "zcr_mean": zcr_mean_val,
        "onset_strength": onset_str,
        "spectral_centroid": spec_centroid,
    }


def main() -> None:
    """Ana akis: sarkilari tara -> DSP ozellikleri cikar -> DB'ye kaydet."""

    # -- Veritabani baglantisi --
    conn = get_connection()

    try:
        # -- Akustik sutunlari garanti altina al --
        ensure_acoustic_columns(conn)

        # -- Sarki listesini cek --
        df = fetch_songs(conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        total = len(df)
        print(f"Toplam {total} sarki islenecek.\n")

        processed = 0
        for idx, row in df.iterrows():
            song_id = row["song_id"]
            title = row["title"]
            file_path = row["file_path"]

            print(f"  [{idx + 1}/{total}] {title}")

            if not os.path.isfile(file_path):
                print(f"    [UYARI] Dosya bulunamadi, atlaniyor: {file_path}")
                continue

            try:
                feats = extract_acoustic_features(file_path)

                # Sonuclari konsola yazdir
                print(f"    BPM: {feats['bpm']:.1f}  |  "
                      f"RMS: {feats['rms_mean']:.4f}  |  "
                      f"ZCR: {feats['zcr_mean']:.4f}  |  "
                      f"Onset: {feats['onset_strength']:.2f}  |  "
                      f"Centroid: {feats['spectral_centroid']:.1f}")

                # Veritabanini guncelle
                conn.execute(
                    """
                    UPDATE features
                    SET bpm = ?, rms_mean = ?, rms_var = ?,
                        zcr_mean = ?, onset_strength = ?, spectral_centroid = ?
                    WHERE song_id = ?;
                    """,
                    (
                        feats["bpm"],
                        feats["rms_mean"],
                        feats["rms_var"],
                        feats["zcr_mean"],
                        feats["onset_strength"],
                        feats["spectral_centroid"],
                        song_id,
                    ),
                )
                conn.commit()
                processed += 1

            except Exception as e:
                print(f"    [HATA] {e}")
                continue

        print(f"\n{'=' * 60}")
        print(f"  1. Kol (Akustik/DSP) analizleri tamamlandi!")
        print(f"  Saf fiziksel veriler veritabanina islendi.")
        print(f"  Islenen: {processed}/{total}")
        print(f"{'=' * 60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
