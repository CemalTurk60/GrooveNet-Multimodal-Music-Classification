# nlp_features.py
# Sarki sozlerini Genius API'den cekip Turkce BERT modeli ile
# duygu analizi (Sentiment Analysis) yapar.
# Sonuclar features tablosundaki lyrics ve nlp_sentiment_score sutunlarina yazilir.

import sqlite3
import os

import pandas as pd
import lyricsgenius
from transformers import pipeline

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# -- Genius API --
GENIUS_TOKEN = "8md71ayU7PadgpaYZXPFZ5Dnkalm4E1pt3eTfEzrhfslfAGADjwACif72BQQ4ahy"

# -- NLP Parametreleri --
BERT_MODEL = "savasy/bert-base-turkish-sentiment-cased"
MAX_SNIPPET_LENGTH = 512  # BERT max token siniri icin karakter kirpma


def get_connection() -> sqlite3.Connection:
    """SQLite veritabanina baglanti acar."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_nlp_columns(conn: sqlite3.Connection) -> None:
    """features tablosuna lyrics ve nlp_sentiment_score sutunlarini ekler (yoksa)."""
    columns = {"lyrics": "TEXT", "nlp_sentiment_score": "REAL"}
    for col, col_type in columns.items():
        try:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} {col_type};")
            conn.commit()
            print(f"  [OK] '{col}' sutunu eklendi.")
        except sqlite3.OperationalError:
            pass
    print("[BILGI] NLP sutunlari hazir.\n")


def setup_genius() -> lyricsgenius.Genius:
    """Genius API istemcisini yapilandirir ve dondurur."""
    genius = lyricsgenius.Genius(GENIUS_TOKEN)
    genius.verbose = False
    genius.remove_section_headers = True
    print("[OK] Genius API baglantisi kuruldu.")
    return genius


def setup_sentiment_analyzer():
    """Turkce BERT duygu analizi pipeline'ini yukler."""
    print("[BILGI] BERT modeli yukleniyor (ilk calistirmada indirme gerekebilir)...")
    analyzer = pipeline("sentiment-analysis", model=BERT_MODEL)
    print("[OK] BERT duygu analizi modeli hazir.\n")
    return analyzer


def fetch_songs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Veritabanindan song_id ve title bilgilerini ceker."""
    query = """
        SELECT
            s.id    AS song_id,
            s.title AS title
        FROM songs AS s
        INNER JOIN features AS f ON f.song_id = s.id;
    """
    df = pd.read_sql_query(query, conn)
    return df


def clean_title(title: str) -> str:
    """Genius aramasini iyilestirmek icin basliktan gereksiz kisimları temizler.

    YouTube indirmelerindeki [...] kodlarini, (Official Video) gibi
    ekleri ve fazla bosluklari kaldirir.
    """
    import re
    # [...] icerigini kaldir (YouTube video ID'leri)
    title = re.sub(r"\[.*?\]", "", title)
    # (Official ...), (HQ), (Lyric Video) gibi ekleri kaldir
    title = re.sub(r"\((?:Official|HQ|HD|Lyric|Audio|Video|Akustik).*?\)", "", title, flags=re.IGNORECASE)
    # Fazla bosluklari temizle
    title = re.sub(r"\s+", " ", title).strip()
    # Bas/son tire ve bosluk
    title = title.strip("- ")
    return title


def analyze_sentiment(lyrics: str, analyzer) -> float:
    """Sarki sozlerinin duygu skorunu hesaplar.

    Args:
        lyrics:   Sarki sozleri metni.
        analyzer: Hugging Face sentiment-analysis pipeline'i.

    Returns:
        float: 0.0 (cok negatif/melankolik) ile 1.0 (cok pozitif) arasi skor.
    """
    # BERT token sinirini asmamak icin ilk N karakteri al
    snippet = lyrics[:MAX_SNIPPET_LENGTH]

    result = analyzer(snippet)[0]
    label = result["label"]
    score = result["score"]

    # Normalizasyon: 0 = negatif, 1 = pozitif
    if label.lower() == "positive":
        return score
    else:
        return 1.0 - score


def main() -> None:
    """Ana akis: API kur -> sozleri cek -> duygu analizi -> DB'ye kaydet."""

    # -- Veritabani baglantisi --
    conn = get_connection()

    try:
        # -- NLP sutunlarini garanti altina al --
        ensure_nlp_columns(conn)

        # -- Genius API --
        genius = setup_genius()

        # -- BERT modeli --
        analyzer = setup_sentiment_analyzer()

        # -- Sarki listesini cek --
        df = fetch_songs(conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        total = len(df)
        print(f"Toplam {total} sarki islenecek.\n")

        results = []  # (title, score, lyrics_preview) — rapor icin

        for idx, row in df.iterrows():
            song_id = row["song_id"]
            raw_title = row["title"]
            search_title = clean_title(raw_title)

            print(f"  [{idx + 1}/{total}] {search_title}")

            # -- Sozleri Genius'tan cek --
            try:
                song = genius.search_song(search_title)
            except Exception as e:
                print(f"    [HATA] Genius arama hatasi: {e}")
                continue

            if song is None or song.lyrics is None:
                print(f"    [UYARI] Soz bulunamadi, atlaniyor.")
                continue

            lyrics = song.lyrics
            print(f"    [OK] Sozler bulundu ({len(lyrics)} karakter)")

            # -- Duygu analizi --
            try:
                score = analyze_sentiment(lyrics, analyzer)
            except Exception as e:
                print(f"    [HATA] Duygu analizi hatasi: {e}")
                continue

            print(f"    [OK] Sentiment skoru: {score:.4f}")

            # -- Veritabanini guncelle --
            conn.execute(
                """
                UPDATE features
                SET lyrics = ?, nlp_sentiment_score = ?
                WHERE song_id = ?;
                """,
                (lyrics, score, song_id),
            )
            conn.commit()

            # Rapor icin ilk 5 kelimeyi kaydet
            words = lyrics.split()[:5]
            preview = " ".join(words) + "..."
            results.append((raw_title, score, preview))

        # -- Sonuc raporu --
        print(f"\n{'=' * 60}")
        print("  NLP Duygu Analizi tamamlandi!")
        print("  Sozler ve skorlar veritabanina islendi.")
        print(f"{'=' * 60}")

        if results:
            # Skora gore sirala
            sorted_results = sorted(results, key=lambda x: x[1])

            print(f"\n  --- EN MELANKOLIK 3 SARKI (en dusuk skor) ---")
            for title, score, preview in sorted_results[:3]:
                print(f"    {score:.4f}  |  {title}")
                print(f"             |  \"{preview}\"")

            print(f"\n  --- EN POZITIF 3 SARKI (en yuksek skor) ---")
            for title, score, preview in sorted_results[-3:]:
                print(f"    {score:.4f}  |  {title}")
                print(f"             |  \"{preview}\"")

            print(f"\n{'=' * 60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
