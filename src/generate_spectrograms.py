# generate_spectrograms.py
# CNN modeli için ses dosyalarından Mel-Spektrogram görselleri üretir.
# Üretilen görseller data/spectrograms/ klasörüne kaydedilir ve
# veritabanındaki features tablosuna yol bilgisi yazılır.

import sqlite3
import os
import re

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # GUI penceresi açmadan kaydetmek için
import matplotlib.pyplot as plt

# ── Proje Yolları ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")
SPECTROGRAM_DIR = os.path.join(BASE_DIR, "data", "spectrograms")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")
SUPPORTED_EXTENSIONS = {".mp3", ".wav"}

# ── Görsel Parametreleri ───────────────────────────────────────────────────
# CNN'e beslenecek boyut: ~224×224 piksel
FIGSIZE = (2.24, 2.24)
DPI = 100
N_MELS = 128
FMAX = 8000
TOP_DB = 20  # sessizlik kırpma eşiği (dB)


def _slugify(text: str) -> str:
    """Dosya adı için güvenli bir slug oluşturur.

    Türkçe karakterleri ASCII karşılıklarına dönüştürür,
    alfanumerik olmayan karakterleri alt çizgiye çevirir.
    """
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(tr_map)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


def get_connection() -> sqlite3.Connection:
    """SQLite veritabanına baglanti acar. Dosya yoksa otomatik olusturur."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    """songs ve features tablolarini olusturur (yoksa).

    GrooveNet bagimsiz calisabilsin diye SongAnalyzer'daki
    sema buraya da eklenmistir.
    """
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
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id               INTEGER UNIQUE,
            bpm                   REAL,
            key_signature         TEXT,
            rms_energy            REAL,
            spectral_centroid     REAL,
            mfcc_1                REAL,
            mfcc_2                REAL,
            mfcc_3                REAL,
            mfcc_4                REAL,
            mfcc_5                REAL,
            chroma_stft           REAL,
            zero_crossing_rate    REAL,
            chord_progression     TEXT,
            onset_strength        REAL,
            rms_var               REAL,
            zcr_mean              REAL,
            melancholy_score      REAL,
            danceability_score    REAL,
            acousticness_score    REAL,
            aggressiveness_score  REAL,
            FOREIGN KEY (song_id) REFERENCES songs (id)
        );
    """)
    conn.commit()
    print("[OK] Veritabani tablolari hazir.")


def ensure_spectrogram_column(conn: sqlite3.Connection) -> None:
    """features tablosuna spectrogram_path sutununu ekler (yoksa).

    SQLite ALTER TABLE ... ADD COLUMN IF NOT EXISTS desteklemedigi icin
    try-except ile kontrol edilir.
    """
    try:
        conn.execute("ALTER TABLE features ADD COLUMN spectrogram_path TEXT;")
        conn.commit()
        print("[OK] features tablosuna 'spectrogram_path' sutunu eklendi.")
    except sqlite3.OperationalError:
        # Sutun zaten mevcut - sorun yok
        print("[BILGI] 'spectrogram_path' sutunu zaten mevcut, atlaniyor.")


def scan_and_ingest(conn: sqlite3.Connection) -> None:
    """data/raw_audio/ klasorunu tarar ve yeni sarkilari veritabanina ekler.

    - songs tablosuna dosya adi ve yolu INSERT OR IGNORE ile eklenir.
    - Her yeni sarki icin features tablosuna bos bir kayit olusturulur.
    """
    from pathlib import Path

    if not os.path.isdir(AUDIO_DIR):
        print(f"[UYARI] Ses dosyasi klasoru bulunamadi: {AUDIO_DIR}")
        return

    audio_files = [
        entry for entry in Path(AUDIO_DIR).iterdir()
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        print("[UYARI] data/raw_audio/ klasorunde ses dosyasi bulunamadi.")
        return

    print(f"[BILGI] Klasorde {len(audio_files)} ses dosyasi bulundu.")

    cursor = conn.cursor()
    new_count = 0

    for entry in audio_files:
        title = entry.stem  # uzantisiz dosya adi
        file_path = str(entry.resolve())  # tam yol

        # songs tablosuna ekle (zaten varsa atla)
        cursor.execute(
            "INSERT OR IGNORE INTO songs (title, file_path) VALUES (?, ?)",
            (title, file_path),
        )

        if cursor.rowcount > 0:
            new_count += 1
            song_id = cursor.lastrowid
            # features tablosuna bos kayit olustur
            cursor.execute(
                "INSERT OR IGNORE INTO features (song_id) VALUES (?)",
                (song_id,),
            )

    conn.commit()
    print(f"[OK] Veritabanina {new_count} yeni sarki eklendi (toplam: {len(audio_files)}).")


def fetch_song_list(conn: sqlite3.Connection) -> pd.DataFrame:
    """songs ve features tablolarını JOIN ile birleştirip şarkı listesini döndürür."""
    query = """
        SELECT
            s.id        AS song_id,
            s.title     AS title,
            s.file_path AS file_path,
            f.id        AS feature_id
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id;
    """
    df = pd.read_sql_query(query, conn)
    return df


def generate_spectrogram(file_path: str, save_path: str) -> None:
    """Bir ses dosyasından Mel-Spektrogram görseli üretir ve kaydeder.

    Adımlar:
        1. librosa ile ses dosyasını yükle
        2. Sessizlikleri kırp (top_db=20)
        3. Mel-Spektrogram matrisini hesapla
        4. Genliği desibele çevir
        5. Eksensiz, saf görseli kaydet (CNN için)
    """
    # 1. Ses dosyasını yükle
    y, sr = librosa.load(file_path)

    # 2. Sessizlikleri kırp
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)

    # 3. Mel-Spektrogram hesapla
    S = librosa.feature.melspectrogram(
        y=y_trimmed, sr=sr, n_mels=N_MELS, fmax=FMAX
    )

    # 4. Desibel dönüşümü
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 5. CNN için saf görselleştirme — eksen, başlık, renk skalası yok
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=DPI)
    librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        fmax=FMAX,
        cmap="magma",
        ax=ax,
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Ana akış: veritabanını hazırla → şarkıları tara → spektrogramları üret."""

    # ── Klasör hazırlığı ──
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

    # ── Veritabanı bağlantısı ──
    conn = get_connection()

    try:
        # -- Tablolari garanti altina al --
        ensure_tables(conn)

        # -- spectrogram_path sutununu garanti altina al --
        ensure_spectrogram_column(conn)

        # -- raw_audio klasorunu tara ve yeni sarkilari DB'ye ekle --
        scan_and_ingest(conn)

        # ── Şarkı listesini çek ──
        df = fetch_song_list(conn)

        if df.empty:
            print("Veritabaninda islenecek sarki bulunamadi.")
            return

        print(f"Toplam {len(df)} sarki islenecek.\n")

        processed = 0
        for _, row in df.iterrows():
            song_id = row["song_id"]
            title = row["title"]
            file_path = row["file_path"]
            feature_id = row["feature_id"]

            # Dosya adı oluştur: id123_bir_ay_dogar.png
            slug = _slugify(title)
            filename = f"id{song_id}_{slug}.png"
            save_path = os.path.join(SPECTROGRAM_DIR, filename)

            # Göreceli yol (veritabanına yazılacak)
            relative_path = os.path.join("data", "spectrograms", filename)

            print(f"  [{processed + 1}/{len(df)}] {title}  ->  {filename}")

            try:
                generate_spectrogram(file_path, save_path)

                # Veritabanını güncelle
                conn.execute(
                    "UPDATE features SET spectrogram_path = ? WHERE id = ?;",
                    (relative_path, feature_id),
                )
                conn.commit()
                processed += 1

            except Exception as e:
                print(f"  [HATA] ({title}): {e}")
                continue

        print(f"\n{'-' * 50}")
        print(f"Basariyla islenen: {processed}/{len(df)}")
        print("Gorsel veri seti hazirlandi ve veritabani guncellendi.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
