# test_cnn_clustering.py
# ResNet18'den elde edilen gorsel ozellikler (cnn_feat1/2/3) ile
# K-Means kumeleme testi yapar ve sonuclari konsola bastirir.

import sqlite3
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# -- Kumeleme Parametreleri --
N_CLUSTERS = 4
RANDOM_STATE = 42
CNN_FEATURE_COLS = ["cnn_feat1", "cnn_feat2", "cnn_feat3"]


def get_connection() -> sqlite3.Connection:
    """SQLite veritabanina baglanti acar."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def fetch_cnn_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """songs ve features tablolarini JOIN ile birlestirerek
    title ve CNN ozellik sutunlarini ceker.
    """
    query = """
        SELECT
            s.title     AS title,
            f.cnn_feat1 AS cnn_feat1,
            f.cnn_feat2 AS cnn_feat2,
            f.cnn_feat3 AS cnn_feat3
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id;
    """
    df = pd.read_sql_query(query, conn)
    return df


def main() -> None:
    """Ana akis: veri cek -> olcekle -> kumele -> raporla."""

    # -- Veritabani baglantisi --
    conn = get_connection()

    try:
        # -- Veriyi cek --
        df = fetch_cnn_features(conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        # -- CNN ozellikleri henuz cikarilmamis sarkilari temizle --
        before_count = len(df)
        df = df.dropna(subset=CNN_FEATURE_COLS)
        dropped = before_count - len(df)

        if dropped > 0:
            print(f"[BILGI] {dropped} sarki CNN ozelligi eksik oldugu icin cikarildi.")

        if df.empty:
            print("CNN ozellikleri cikarilmis sarki bulunamadi.")
            print("Once 'cnn_features.py' scriptini calistirin.")
            return

        # -- StandardScaler ile olceklendirme --
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[CNN_FEATURE_COLS])

        # -- K-Means kumeleme --
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        # -- Sonuclari konsola bastir --
        print()
        print("=" * 60)
        print("  GROOVENET: SADECE GORSEL (CNN) KUMELEME TESTI")
        print("=" * 60)
        print(f"  Toplam sarki  : {len(df)}")
        print(f"  Kume sayisi   : {N_CLUSTERS}")
        print(f"  Ozellik sayisi: {len(CNN_FEATURE_COLS)} (cnn_feat1/2/3)")
        print("=" * 60)

        for cluster_id in range(N_CLUSTERS):
            cluster_songs = df[df["cluster"] == cluster_id]["title"].tolist()
            print(f"\n  [Cluster {cluster_id}] ({len(cluster_songs)} sarki)")
            print(f"  {'-' * 40}")
            for song in cluster_songs:
                print(f"    - {song}")

        print(f"\n{'=' * 60}")
        print("  Kumeleme testi tamamlandi.")
        print(f"{'=' * 60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
