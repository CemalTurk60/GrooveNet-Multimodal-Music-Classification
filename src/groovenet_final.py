# groovenet_final.py
# GROOVENET FINAL — BERT cikarildi, Lexicon + CNN ile temiz kumeleme.
# DSP gurultusu yok, BERT yanilgisi yok.
# Sadece spektrogram dokusu (CNN) + sozluk tabanli duygu (Lexicon).

import sqlite3
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# -- Kumeleme Parametreleri --
N_CLUSTERS = 4
RANDOM_STATE = 42

# -- Ozellik Secimi --
CNN_COLS = ["cnn_feat1", "cnn_feat2", "cnn_feat3"]
NLP_COL = "lexicon_score"  # BERT degil, sozluk tabanli skor
FEATURE_COLS = CNN_COLS + [NLP_COL]

# -- Agirliklar --
WEIGHT_CNN = 2.0    # Gorsel doku — tur ayriminda en guvenilir
WEIGHT_NLP = 1.0    # Sozlerin ruhu — destek rolu


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def fetch_features(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT
            s.title        AS title,
            f.cnn_feat1    AS cnn_feat1,
            f.cnn_feat2    AS cnn_feat2,
            f.cnn_feat3    AS cnn_feat3,
            f.lexicon_score AS lexicon_score
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id;
    """
    return pd.read_sql_query(query, conn)


def apply_weights(X: np.ndarray) -> np.ndarray:
    """Sutun sirasi: cnn1, cnn2, cnn3, lexicon_score"""
    Xw = X.copy()
    Xw[:, 0] *= WEIGHT_CNN
    Xw[:, 1] *= WEIGHT_CNN
    Xw[:, 2] *= WEIGHT_CNN
    Xw[:, 3] *= WEIGHT_NLP
    return Xw


def emotion_label(score: float) -> str:
    if score < 0.35:
        return "Melankolik"
    elif score < 0.50:
        return "Huzunlu"
    elif score < 0.60:
        return "Notr"
    elif score < 0.75:
        return "Pozitif"
    else:
        return "Neseli"


def main() -> None:
    conn = get_connection()

    try:
        df = fetch_features(conn)

        before = len(df)
        df = df.dropna(subset=FEATURE_COLS)
        dropped = before - len(df)

        if df.empty:
            print("Yeterli veri yok. Once su sirayla calistirin:")
            print("  1. py src/generate_spectrograms.py")
            print("  2. py src/cnn_features.py")
            print("  3. py src/nlp_features.py  (Genius'tan sozleri ceker)")
            print("  4. py src/lexicon_nlp.py   (Sozluk tabanli skor)")
            return

        # -- Olceklendirme --
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[FEATURE_COLS])

        # -- Agirliklandirma --
        X_weighted = apply_weights(X_scaled)

        # -- K-Means --
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        df = df.copy()
        df["cluster"] = kmeans.fit_predict(X_weighted)

        # ============================================================
        #  FINAL RAPOR
        # ============================================================
        print()
        print("=" * 68)
        print("        GROOVENET FINAL — KURTARMA OPERASYONU TAMAM")
        print("        CNN (Gorsel Doku) + Lexicon (Soz Ruhu)")
        print("=" * 68)
        print(f"  BERT    : DEVRE DISI (urun yorumu modeli, muzige uygun degil)")
        print(f"  DSP     : DEVRE DISI (BPM gurultusu, yanlis olcumler)")
        print(f"  CNN     : AKTIF x{WEIGHT_CNN} (ResNet18 spektrogram ozellikleri)")
        print(f"  Lexicon : AKTIF x{WEIGHT_NLP} "
              f"({len(df)} sarkinin sozleri, kok eslestirme)")
        print(f"  Kume    : {N_CLUSTERS}  |  Sarki: {len(df)}  |  Elenen: {dropped}")
        print("=" * 68)

        for cid in range(N_CLUSTERS):
            cdf = df[df["cluster"] == cid].sort_values(NLP_COL)
            count = len(cdf)
            avg_lex = cdf[NLP_COL].mean()
            label = emotion_label(avg_lex)

            print(f"\n  {'~' * 64}")
            print(f"  KUME {cid}  |  {count} sarki  |  "
                  f"Ort. Duygu: {avg_lex:.3f} ({label})")
            print(f"  {'~' * 64}")

            for _, row in cdf.iterrows():
                lex = row[NLP_COL]
                tag = emotion_label(lex)
                title = row["title"]
                print(f"    {lex:.3f} ({tag:12s})  |  {title}")

        print(f"\n{'=' * 68}")
        print("  GrooveNet Final modeli basariyla calistirildi!")
        print("  Sozluk tabanli NLP + CNN gorsel kumeleme tamamlandi.")
        print(f"{'=' * 68}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
