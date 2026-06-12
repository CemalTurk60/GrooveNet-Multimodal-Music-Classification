# groovenet_genre.py
# GROOVENET — TURE GORE SINIFLANDIRMA
# Elle atanan tur etiketleri + CNN ozellikleri + KNN siniflandirici.
# Duygu (lexicon_score) sadece ek bilgi olarak gosterilir, kumeleme'de kullanilmaz.

import sqlite3
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# -- Ozellik Secimi --
# SADECE CNN: tur ayrimini gorsel doku yapar
# NLP (lexicon_score) kumeleme'de KULLANILMAZ, raporda gosterilir
CNN_COLS = ["cnn_feat1", "cnn_feat2", "cnn_feat3"]

# -- KNN Parametreleri --
K_NEIGHBORS = 3  # 30 sarki icin k=3 uygun


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def fetch_all_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Tum sarki verilerini getirir: baslik, CNN, lexicon, genre."""
    query = """
        SELECT
            s.title          AS title,
            f.cnn_feat1      AS cnn_feat1,
            f.cnn_feat2      AS cnn_feat2,
            f.cnn_feat3      AS cnn_feat3,
            f.lexicon_score  AS lexicon_score,
            f.genre          AS genre
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id;
    """
    return pd.read_sql_query(query, conn)


def emotion_label(score: float) -> str:
    """Lexicon skorunu etiketler (sadece rapor icin)."""
    if pd.isna(score):
        return "?"
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


def evaluate_knn(X: np.ndarray, y: np.ndarray, genre_names: list) -> float:
    """Leave-One-Out Cross Validation ile KNN dogrulugunu olcer.

    30 sarki icin train/test split yerine LOO daha guvenilirdir.
    Her seferinde 1 sarki test, 29 sarki egitim olarak kullanilir.
    """
    loo = LeaveOneOut()
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)

    correct = 0
    total = 0
    misclassified = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)[0]
        actual = y_test[0]

        if pred == actual:
            correct += 1
        else:
            misclassified.append((test_idx[0], actual, pred))
        total += 1

    accuracy = correct / total
    return accuracy, misclassified


def main() -> None:
    conn = get_connection()

    try:
        df = fetch_all_data(conn)

        # CNN ozellikleri ve genre etiketi olmayan sarkilari cikar
        df_clean = df.dropna(subset=CNN_COLS + ["genre"])
        df_clean = df_clean[df_clean["genre"] != "BILINMIYOR"]

        if df_clean.empty:
            print("Yeterli etiketlenmis veri yok.")
            print("Once calistirin: py src/genre_labeler.py")
            return

        print()
        print("=" * 68)
        print("     GROOVENET — TURE GORE SINIFLANDIRMA")
        print("     CNN Spektrogram Ozellikleri + KNN Siniflandirici")
        print("=" * 68)

        # -- Ozellik matrisi --
        X = df_clean[CNN_COLS].values
        y = df_clean["genre"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -- Tur dagilimi --
        genre_counts = df_clean["genre"].value_counts()
        print(f"\n  Egitim verisi: {len(df_clean)} sarki\n")
        for genre, count in genre_counts.items():
            bar = "#" * (count * 3)
            print(f"    {genre:10s}: {count:2d}  {bar}")

        # -- LOO Cross Validation --
        print(f"\n  KNN (k={K_NEIGHBORS}) + Leave-One-Out Cross Validation:")
        accuracy, misclassified = evaluate_knn(X_scaled, y, list(genre_counts.index))

        print(f"  Dogruluk: {accuracy:.1%} ({int(accuracy * len(df_clean))}/{len(df_clean)})")

        if misclassified:
            print(f"\n  Yanlis siniflandirilanlar:")
            for idx, actual, pred in misclassified:
                title = df_clean.iloc[idx]["title"]
                print(f"    {title}")
                print(f"      Gercek: {actual}  |  Tahmin: {pred}")

        # -- Final model egit (tum veri) --
        knn_final = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        knn_final.fit(X_scaled, y)

        df_clean = df_clean.copy()
        df_clean["predicted_genre"] = knn_final.predict(X_scaled)

        # -- TURE GORE RAPOR --
        print(f"\n{'=' * 68}")
        print("  TURE GORE SARKI LISTESI")
        print("  (Duygu = ek bilgi, siniflandirmada kullanilmadi)")
        print(f"{'=' * 68}")

        genres_sorted = sorted(df_clean["genre"].unique())

        for genre in genres_sorted:
            genre_df = df_clean[df_clean["genre"] == genre].sort_values(
                "lexicon_score", ascending=True, na_position="last"
            )
            count = len(genre_df)

            # Ortalama duygu
            avg_lex = genre_df["lexicon_score"].mean()
            avg_tag = emotion_label(avg_lex)

            print(f"\n  {'~' * 64}")
            print(f"  {genre}  |  {count} sarki  |  Ort. Duygu: {avg_lex:.3f} ({avg_tag})")
            print(f"  {'~' * 64}")

            for _, row in genre_df.iterrows():
                title = row["title"]
                lex = row["lexicon_score"]
                tag = emotion_label(lex)
                print(f"    {tag:12s}  |  {title}")

        print(f"\n{'=' * 68}")
        print("  GrooveNet Genre siniflandirmasi tamamlandi!")
        print(f"  Model: KNN (k={K_NEIGHBORS}) + CNN Spektrogram Ozellikleri")
        print(f"  Dogruluk: {accuracy:.1%}")
        print(f"{'=' * 68}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
