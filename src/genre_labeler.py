# genre_labeler.py
# Sarkilara elle tur etiketi (genre label) atayan script.
# Bu etiketler egitim verisi olarak kullanilacak.
# Yeni sarki eklediginde bu dosyadaki GENRE_MAP sozlugune eklemen yeterli.

import sqlite3
import os

import pandas as pd

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# ============================================================
#  ELLE TUR ETIKETI HARITASI
#  Anahtar: sarki basliginin ICINDE gecen benzersiz metin
#  Deger: tur etiketi
#
#  TURLER:
#    THM       = Turk Halk Muzigi
#    ARABESK   = Arabesk / Fantezi
#    ROCK      = Turk Rock / Alternatif Rock
#    POP       = Turk Pop
#    RAP       = Rap / Hip-Hop
#    INDIE     = Indie / Alternatif Pop
# ============================================================

GENRE_MAP = {
    # -- TURK HALK MUZIGI (THM) --
    "ASIK VEYSEL":          "THM",
    "Musa Eroglu":          "THM",
    "Cengiz Ozkan":         "THM",
    "Mumin Sarikaya":       "THM",
    "Neredesin Sen":        "THM",
    "Kivircik Ali":         "THM",


    # -- ARABESK --
    "Ferdi Tayfur":         "ARABESK",
    "Orhan Gencebay":       "ARABESK",
    "Muslum Gurses - Affet":"ARABESK",
    "Isyankar":             "ARABESK",
    "CENGIZ KURTOGLU":      "ARABESK",

    # -- TURK ROCK / ALTERNATIF --
    "Duman":                "ROCK",
    "maNga":                "ROCK",
    "mor ve otesi":         "ROCK",
    "Sebnem Ferah":         "ROCK",
    "Teoman":               "ROCK",
    "Adamlar":              "ROCK",
    "Can Gox":              "ROCK",

    # -- TURK POP --
    "Tarkan":               "POP",
    "Hande Yener":          "POP",
    "Serdar Ortac":         "POP",
    "Mabel Matiz":          "POP",
    "Sezen Aksu":           "POP",

    # -- RAP / HIP-HOP --
    "Ceza":                 "RAP",
    "Ezhel":                "RAP",
    "Hidra":                "RAP",
    "Heyecani Yok":         "RAP",
    "Atesten Gomlek":       "RAP",
    "Contra":               "RAP",

    # -- INDIE / ALTERNATIF POP --
    "Dolu Kadehi":          "INDIE",
    "Son Feci Bisiklet":    "INDIE",
    "Yuzyuzeyken":          "INDIE",
    "Madrigal":             "INDIE",

}


def normalize_for_match(text: str) -> str:
    """Baslik eslestirme icin Turkce karakterleri ASCII'ye cevirir."""
    tr_map = str.maketrans(
        "çğıöşüâîûÇĞİÖŞÜÂÎÛ",
        "cgiosuaiuCGIOSUAIU"
    )
    return text.translate(tr_map)


def find_genre(title: str) -> str:
    """Sarki basligina gore tur etiketini bulur.

    GENRE_MAP'teki anahtarlar basligin ICINDE aranir.
    Bulunamazsa 'BILINMIYOR' doner.
    """
    title_norm = normalize_for_match(title)
    for key, genre in GENRE_MAP.items():
        key_norm = normalize_for_match(key)
        if key_norm.lower() in title_norm.lower():
            return genre
    return "BILINMIYOR"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_genre_column(conn: sqlite3.Connection) -> None:
    """features tablosuna genre sutununu ekler (yoksa)."""
    try:
        conn.execute("ALTER TABLE features ADD COLUMN genre TEXT;")
        conn.commit()
        print("[OK] 'genre' sutunu eklendi.")
    except sqlite3.OperationalError:
        pass


def main() -> None:
    conn = get_connection()

    try:
        ensure_genre_column(conn)

        query = """
            SELECT s.id AS song_id, s.title AS title
            FROM songs AS s;
        """
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        print(f"\n{'=' * 60}")
        print("  ELLE TUR ETIKETLEME (Genre Labeling)")
        print(f"{'=' * 60}\n")

        # Tur sayaclari
        genre_counts = {}

        for _, row in df.iterrows():
            song_id = row["song_id"]
            title = row["title"]
            genre = find_genre(title)

            conn.execute(
                "UPDATE features SET genre = ? WHERE song_id = ?;",
                (genre, song_id),
            )

            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            print(f"  [{genre:10s}]  {title}")

        conn.commit()

        # Ozet
        print(f"\n{'-' * 60}")
        print("  TUR DAGILIMI:")
        for genre, count in sorted(genre_counts.items()):
            bar = "#" * (count * 3)
            print(f"    {genre:12s}: {count:2d} sarki  {bar}")
        print(f"{'-' * 60}")

        unknown = genre_counts.get("BILINMIYOR", 0)
        if unknown > 0:
            print(f"\n  [UYARI] {unknown} sarki etiketlenemedi!")
            print("  genre_labeler.py icindeki GENRE_MAP sozlugunu guncelle.")

        print(f"\n  Toplam {len(df)} sarki etiketlendi.")
        print(f"{'=' * 60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
