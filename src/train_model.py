# train_model.py
# Etiketlenmis sarkilar uzerinde tur siniflandirma modeli egitir ve
# DISKE KAYDEDER. Yeni sarkiler icin tekrar egitim gerektirmez.

import sqlite3
import os
import joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
MODEL_PATH = os.path.join(MODEL_DIR, "genre_classifier.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "genre_scaler.joblib")

# -- MFCC Ozellik Sutunlari (mfcc_features.py ile uyumlu) --
N_MFCC = 13
N_CONTRAST = 6
N_CHROMA = 12
MFCC_COLS = [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
CONTRAST_COLS = [f"contrast_{i}" for i in range(1, N_CONTRAST + 1)]
CHROMA_COLS = [f"chroma_{i}" for i in range(1, N_CHROMA + 1)]
OTHER_COLS = ["zcr", "centroid"]
FEATURE_COLS = MFCC_COLS + CONTRAST_COLS + CHROMA_COLS + OTHER_COLS

# -- Model Parametreleri --
N_ESTIMATORS = 100
RANDOM_STATE = 42


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def fetch_training_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Etiketlenmis ve MFCC ozellikleri cikarilmis sarkilari getirir."""
    cols = ", ".join([f"f.{c}" for c in FEATURE_COLS])
    query = f"""
        SELECT
            s.title   AS title,
            f.genre   AS genre,
            {cols}
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id
        WHERE f.genre IS NOT NULL
          AND f.genre != 'BILINMIYOR'
          AND f.mfcc_1 IS NOT NULL;
    """
    return pd.read_sql_query(query, conn)


def main() -> None:
    conn = get_connection()

    try:
        df = fetch_training_data(conn)

        if df.empty:
            print("Egitim verisi yok. Once calistirin:")
            print("  1. py src/genre_labeler.py")
            print("  2. py src/mfcc_features.py")
            return

        print()
        print("=" * 60)
        print("  GROOVENET — MODEL EGITIMI")
        print("  Random Forest + 34 MFCC Ses Ozelligi")
        print("=" * 60)

        X = df[FEATURE_COLS].values
        y = df["genre"].values
        titles = df["title"].values

        print(f"\n  Egitim verisi: {len(df)} sarki")
        print(f"  Ozellik sayisi: {len(FEATURE_COLS)}")

        # Tur dagilimi
        genre_counts = pd.Series(y).value_counts()
        for genre, count in genre_counts.items():
            print(f"    {genre:10s}: {count}")

        # -- Olceklendirme --
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -- Leave-One-Out Cross Validation --
        print(f"\n  Leave-One-Out Cross Validation:")
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        loo = LeaveOneOut()
        y_pred = cross_val_predict(rf, X_scaled, y, cv=loo)

        acc = accuracy_score(y, y_pred)
        print(f"  Dogruluk: {acc:.1%} ({int(acc * len(y))}/{len(y)})")

        # Yanlis siniflandirilanlar
        wrong_mask = y != y_pred
        if wrong_mask.any():
            print(f"\n  Yanlis siniflandirilanlar:")
            for i in np.where(wrong_mask)[0]:
                print(f"    {titles[i]}")
                print(f"      Gercek: {y[i]}  |  Tahmin: {y_pred[i]}")

        # -- Siniflandirma raporu --
        print(f"\n  Detayli Siniflandirma Raporu:")
        report = classification_report(y, y_pred, zero_division=0)
        for line in report.split("\n"):
            print(f"    {line}")

        # -- FINAL MODEL EGIT (tum veri) --
        rf_final = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
        )
        rf_final.fit(X_scaled, y)

        # -- DISKE KAYDET --
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(rf_final, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

        print(f"\n  [OK] Model kaydedildi: {MODEL_PATH}")
        print(f"  [OK] Scaler kaydedildi: {SCALER_PATH}")

        # -- Ozellik onemliligi --
        importances = rf_final.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print(f"\n  En onemli 10 ozellik:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {FEATURE_COLS[idx]:12s}  "
                  f"({importances[idx]:.4f})")

        print(f"\n{'=' * 60}")
        print("  Model egitimi tamamlandi ve diske kaydedildi!")
        print("  Yeni sarkilar icin: py src/predict.py")
        print(f"{'=' * 60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
