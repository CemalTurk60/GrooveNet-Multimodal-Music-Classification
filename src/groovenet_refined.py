# groovenet_refined.py
# KURTARMA OPERASYONU — Gurultulu DSP verilerini devre disi birakip
# sadece CNN (gorsel doku) ve NLP (sozlerin ruhu) ile rafine kumeleme yapar.
# Feature Selection: 10 ozellikten 4'e dusuruldu, gurultu elendi.

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

# -- Rafine Ozellik Secimi (Feature Selection) --
# DSP (bpm, rms, zcr, onset, centroid) TAMAMEN CIKARILDI
CNN_COLS = ["cnn_feat1", "cnn_feat2", "cnn_feat3"]
NLP_COLS = ["nlp_sentiment_score"]
FEATURE_COLS = CNN_COLS + NLP_COLS

# -- Stratejik Agirliklar --
WEIGHT_CNN = 1.5   # Gorsel doku — turu en iyi resimler belirliyor
WEIGHT_NLP = 2.0   # Sozlerin ruhu — Asik Veysel'i maNga'dan ayiran anahtar


def get_connection() -> sqlite3.Connection:
    """SQLite veritabanina baglanti acar."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def fetch_refined_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """songs ve features tablolarini birlestirip
    SADECE CNN + NLP sutunlarini ceker.
    """
    query = """
        SELECT
            s.title              AS title,
            f.cnn_feat1          AS cnn_feat1,
            f.cnn_feat2          AS cnn_feat2,
            f.cnn_feat3          AS cnn_feat3,
            f.nlp_sentiment_score AS nlp_sentiment_score
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id;
    """
    df = pd.read_sql_query(query, conn)
    return df


def apply_weights(X_scaled: np.ndarray) -> np.ndarray:
    """Olceklenmis matrise stratejik agirliklari uygular.

    Sutun sirasi: [cnn_feat1, cnn_feat2, cnn_feat3, nlp_sentiment_score]
    """
    X_weighted = X_scaled.copy()
    # ilk 3 sutun: CNN
    X_weighted[:, 0] *= WEIGHT_CNN
    X_weighted[:, 1] *= WEIGHT_CNN
    X_weighted[:, 2] *= WEIGHT_CNN
    # son sutun: NLP
    X_weighted[:, 3] *= WEIGHT_NLP
    return X_weighted


def sentiment_tag(score: float) -> str:
    """NLP skorunu kisa etikete cevirir."""
    if score < 0.20:
        return "Cok Melankolik"
    elif score < 0.40:
        return "Melankolik"
    elif score < 0.55:
        return "Notr"
    elif score < 0.75:
        return "Pozitif"
    else:
        return "Cok Pozitif"


def main() -> None:
    """Ana akis: rafine veri cek -> olcekle -> agirliklandir -> kumele -> raporla."""

    conn = get_connection()

    try:
        df = fetch_refined_features(conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        # -- Eksik verileri temizle --
        before = len(df)
        df = df.dropna(subset=FEATURE_COLS)
        dropped = before - len(df)

        if dropped > 0:
            print(f"[BILGI] {dropped} sarki eksik veri (NLP/CNN) nedeniyle elendi.")

        if df.empty:
            print("Yeterli veri yok. Once pipeline'lari calistirin.")
            return

        print(f"[OK] {len(df)} sarki rafine modele alindi (elenen: {dropped}).\n")

        # -- StandardScaler --
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[FEATURE_COLS])

        # -- Stratejik agirliklandirma --
        X_weighted = apply_weights(X_scaled)

        # -- K-Means kumeleme --
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        df = df.copy()
        df["cluster"] = kmeans.fit_predict(X_weighted)

        # ============================================================
        #  RAPOR
        # ============================================================
        print("=" * 68)
        print("       GROOVENET: RAFINE MODEL (CNN + NLP)")
        print("       Kurtarma Operasyonu — DSP Gurultusu Elendi")
        print("=" * 68)
        print(f"  Kullanilan ozellikler : {len(FEATURE_COLS)} "
              f"(CNN x{WEIGHT_CNN} + NLP x{WEIGHT_NLP})")
        print(f"  Elenen DSP ozellikleri: bpm, rms_mean, rms_var, "
              f"zcr_mean, onset, centroid")
        print(f"  Kume sayisi           : {N_CLUSTERS}")
        print(f"  Toplam sarki          : {len(df)}")
        print("=" * 68)

        for cid in range(N_CLUSTERS):
            cluster_df = df[df["cluster"] == cid].sort_values(
                "nlp_sentiment_score", ascending=True
            )
            count = len(cluster_df)
            avg_nlp = cluster_df["nlp_sentiment_score"].mean()
            tag = sentiment_tag(avg_nlp)

            print(f"\n  {'~' * 64}")
            print(f"  KUME {cid}  |  {count} sarki  |  "
                  f"Ort. Duygu: {avg_nlp:.4f} ({tag})")
            print(f"  {'~' * 64}")

            for _, row in cluster_df.iterrows():
                title = row["title"]
                nlp = row["nlp_sentiment_score"]
                nlp_tag = sentiment_tag(nlp)
                print(f"    {nlp:.4f} ({nlp_tag:15s})  |  {title}")

        print(f"\n{'=' * 68}")
        print("  Rafine model basariyla calistirildi!")
        print("  DSP gurultusu elendi, sadece CNN + NLP ile kumelendi.")
        print(f"{'=' * 68}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
