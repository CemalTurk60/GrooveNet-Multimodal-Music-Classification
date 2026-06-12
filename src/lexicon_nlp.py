# lexicon_nlp.py
# v2 — Kelime siniri eslestirme + temizlenmis sozluk.
# Substring hatalarini (kul->kullan, son->sonra) ortadan kaldirir.
# Turkce'nin eklemeli yapisina uygun: kelime BASI eslestirme kullanir.

import sqlite3
import os
import re

import pandas as pd
import numpy as np

# -- Proje Yollari --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# ============================================================
#  TURKCE MUZIK DUYGU SOZLUGU v2
#  - Kisa/belirsiz kokler cikarildi (son, bit, gel, yaz, kul, olur)
#  - Agirliklar dengelendi
#  - Kelime basi (prefix) eslestirme kullanilacak
# ============================================================

NEGATIVE_LEXICON = {
    # -- Temel huzun --
    "huzun": 1.5, "huzunl": 1.5, "uzgun": 1.5, "uzul": 1.5,
    "keder": 1.5, "kasvet": 1.5, "elem": 1.0,
    "dertl": 1.5, "derdim": 1.5, "derdine": 1.5,
    # -- Ayrilik / kayip --
    "ayril": 2.0, "ayrilik": 2.0, "vedal": 1.5,
    "terket": 2.0, "terkedi": 2.0,
    "kaybet": 2.0, "kaybett": 2.0, "kayip": 1.5,
    # -- Yalnizlik --
    "yalniz": 2.0, "yalnizl": 2.0, "sensiz": 2.0,
    "kimsesiz": 2.0, "yapayalniz": 2.0,
    # -- Aci / yara --
    "aciy": 1.5, "acilar": 1.5, "acisi": 1.5, "acim": 1.5,
    "sizi": 1.5, "sizla": 1.5,
    "yarala": 1.5, "yarali": 1.5, "yaram": 1.5, "yaralar": 1.5,
    # -- Aglama --
    "agla": 2.0, "aglat": 2.0, "agliy": 2.0,
    "gozyasi": 2.0, "gozyasla": 2.0,
    "feryat": 1.5, "figan": 1.5,
    # -- Olum --
    "olum": 2.0, "olumu": 2.0, "olmek": 1.5, "olece": 1.5,
    "oldur": 1.5,
    # -- Tukenme / bitis --
    "tuken": 1.5, "tukend": 1.5, "tukeni": 1.5,
    "bitti": 1.0, "bitis": 1.0, "bitir": 1.0,
    # -- Ozlem / hasret --
    "hasret": 1.5, "ozledim": 1.5, "ozlem": 1.5, "ozluy": 1.5,
    # -- Karanlik imgeler --
    "karanlik": 1.0, "soguk": 0.5, "firtina": 1.0,
    "geceler": 0.8, "geceleri": 0.8,
    # -- Arabesk --
    "kahir": 2.0, "cile": 2.0, "gurbet": 2.0, "sitem": 1.5,
    "isyan": 1.5, "yanik": 1.0, "kuller": 1.5,
    "batsin": 1.5, "mahkum": 1.5, "zulm": 1.5,
    # -- Mutsuzluk --
    "mutsuz": 1.5, "yoruldum": 1.5, "yorgunum": 1.5,
    "biktim": 1.5, "usandim": 1.0, "bezdim": 1.0,
    "bezgin": 1.0,
    # -- Pisimanlik --
    "pisman": 1.5, "affet": 1.0, "gunah": 1.0,
    # -- Ihanet --
    "hain": 1.5, "ihanet": 2.0, "nefret": 1.5, "lanet": 1.0,
    # -- Gitme baglami --
    "gitme": 1.5, "gidiyor": 1.0, "gidis": 1.5,
}

POSITIVE_LEXICON = {
    # -- Mutluluk --
    "mutlu": 2.0, "mutlul": 2.0, "mutluy": 2.0,
    "sevinc": 2.0, "nesel": 2.0, "nese": 2.0,
    "kahkaha": 1.5, "gulums": 1.5, "guluy": 1.5,
    "guldu": 1.0, "guler": 1.0,
    # -- Guzellik --
    "guzel": 1.0, "guzellik": 1.5, "tatli": 1.0,
    # -- Isik / sicaklik --
    "gunes": 1.0, "gunesl": 1.0,
    "isik": 0.8, "isikl": 0.8, "aydinl": 1.0,
    "bahar": 1.0,
    # -- Dogal guzellik --
    "cicek": 1.0, "cicekl": 1.0,
    "deniz": 0.8, "yildiz": 0.5,
    # -- Sevgi --
    "sevgi": 1.5, "sevgili": 1.5, "sevdam": 1.5, "sevdali": 1.5,
    "seviyor": 1.5, "sevdim": 1.5, "seviy": 1.5,
    "kucakl": 1.5, "saril": 1.5, "opucu": 1.0, "optu": 1.0,
    # -- Yasam / umut --
    "umut": 2.0, "umud": 2.0, "umutl": 2.0,
    "hayalim": 1.0, "hayaller": 1.0,
    # -- Dans / eglence --
    "dans": 2.0, "danset": 2.0, "eglen": 2.0, "eglenc": 2.0,
    "oyna": 1.5, "cosku": 2.0, "coskul": 2.0,
    "parti": 1.5, "keyif": 1.5, "keyifl": 1.5,
    "zevk": 1.0,
    # -- Birliktelik --
    "kavus": 2.0, "kavust": 2.0,
    "bulus": 1.5, "birlikte": 1.0, "beraber": 1.0,
    # -- Ozgurluk --
    "ozgur": 1.5, "ozgurl": 1.5,
    # -- Enerji (pozitif) --
    "heyecan": 1.5, "heyecanl": 1.5,
    "simarik": 2.0, "cilgin": 1.5, "cilginl": 1.5,
    "enerji": 1.0, "hadi": 0.8, "haydi": 0.8,
    "patlat": 1.0, "coskun": 1.5,
}

# Yanlis eslesmeleri onlemek icin dislanacak kelime baslangiclari
# Ornek: "kullan" kelimesi "kul" (kul/kül) olarak ESLESMEMELI
EXCLUDE_PREFIXES = {
    "kullan", "kulak", "sonra", "sonsu", "bitak", "gecen",
    "gecik", "acik", "acil", "acikla", "gelec", "gelin",
    "gelir", "gelis", "yazil", "yazdi", "yazma", "yaziy",
    "gider",
}


def normalize_text(text: str) -> str:
    """Metni normalize eder: kucuk harf, Turkce->ASCII, ozel karakter temizligi."""
    text = text.lower()
    tr_map = str.maketrans(
        "çğıöşüâîûÇĞİÖŞÜÂÎÛ",
        "cgiosuaiuCGIOSUAIU"
    )
    text = text.translate(tr_map)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_prefix_count(words: list, stem: str, exclude_set: set) -> int:
    """Kelime listesinde, verilen kokle BASLAYAN kelimeleri sayar.
    Exclude listesindeki kelimelerle baslayanlari atlar.

    Bu yontem substring hatasini (kul->kullan) onler.
    """
    count = 0
    for word in words:
        if word.startswith(stem):
            # Dislanacak kelime mi kontrol et
            excluded = False
            for exc in exclude_set:
                if word.startswith(exc):
                    excluded = True
                    break
            if not excluded:
                count += 1
    return count


def calculate_lexicon_score(lyrics: str) -> dict:
    """Kelime siniri eslestirme ile duygu skoru hesaplar.

    v2 Yenilikleri:
        - Substring yerine kelime-basi (prefix) eslestirme
        - Belirsiz kisa kokler cikarildi
        - Exclude listesi ile yanlis pozitifler onlendi
    """
    normalized = normalize_text(lyrics)
    words = normalized.split()

    pos_total = 0.0
    neg_total = 0.0
    pos_words = []
    neg_words = []

    for stem, weight in NEGATIVE_LEXICON.items():
        count = word_prefix_count(words, stem, EXCLUDE_PREFIXES)
        if count > 0:
            neg_total += weight * count
            neg_words.append(f"{stem}({count})")

    for stem, weight in POSITIVE_LEXICON.items():
        count = word_prefix_count(words, stem, EXCLUDE_PREFIXES)
        if count > 0:
            pos_total += weight * count
            pos_words.append(f"{stem}({count})")

    epsilon = 1.0
    net = (pos_total - neg_total) / (pos_total + neg_total + epsilon)
    score = (net + 1.0) / 2.0
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "pos_total": pos_total,
        "neg_total": neg_total,
        "pos_words": pos_words,
        "neg_words": neg_words,
    }


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_lexicon_column(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE features ADD COLUMN lexicon_score REAL;")
        conn.commit()
    except sqlite3.OperationalError:
        pass


def main() -> None:
    conn = get_connection()

    try:
        ensure_lexicon_column(conn)

        query = """
            SELECT s.id AS song_id, s.title AS title, f.lyrics AS lyrics
            FROM songs AS s
            INNER JOIN features AS f ON f.song_id = s.id;
        """
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("Veritabaninda sarki bulunamadi.")
            return

        total = len(df)
        print(f"\n{'=' * 68}")
        print("  LEXICON v2 — KELIME SINIRI ESLESTIRME")
        print("  Substring hatalari giderildi, temiz kokler")
        print(f"{'=' * 68}")
        print(f"  Negatif sozluk : {len(NEGATIVE_LEXICON)} kok")
        print(f"  Pozitif sozluk : {len(POSITIVE_LEXICON)} kok")
        print(f"  Exclude listesi: {len(EXCLUDE_PREFIXES)} kelime")
        print(f"  Toplam sarki   : {total}\n")

        results = []

        for _, row in df.iterrows():
            song_id = row["song_id"]
            title = row["title"]
            lyrics = row["lyrics"]

            if pd.isna(lyrics) or not lyrics.strip():
                print(f"  [--] {title} -> Soz yok, atlaniyor.")
                continue

            analysis = calculate_lexicon_score(lyrics)
            score = analysis["score"]

            conn.execute(
                "UPDATE features SET lexicon_score = ? WHERE song_id = ?;",
                (score, song_id),
            )
            conn.commit()

            if score < 0.35:
                tag = "MELANKOLIK"
            elif score < 0.50:
                tag = "HUZUNLU"
            elif score < 0.60:
                tag = "NOTR"
            elif score < 0.75:
                tag = "POZITIF"
            else:
                tag = "NESELI"

            print(f"  {score:.3f} [{tag:11s}]  {title}")
            if analysis["neg_words"]:
                print(f"        Neg: {', '.join(analysis['neg_words'][:5])}")
            if analysis["pos_words"]:
                print(f"        Pos: {', '.join(analysis['pos_words'][:5])}")

            results.append((title, score, tag))

        if results:
            sorted_r = sorted(results, key=lambda x: x[1])
            print(f"\n{'=' * 68}")
            print("  EN MELANKOLIK 5:")
            for t, s, tag in sorted_r[:5]:
                print(f"    {s:.3f}  |  {t}")
            print(f"\n  EN POZITIF 5:")
            for t, s, tag in sorted_r[-5:]:
                print(f"    {s:.3f}  |  {t}")
            print(f"{'=' * 68}")
            print("  Lexicon v2 duygu analizi tamamlandi!")
            print(f"{'=' * 68}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
