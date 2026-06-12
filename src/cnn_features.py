# cnn_features.py
# Önceden eğitilmiş ResNet18 kullanarak spektrogram görsellerinden
# özellik çıkarımı (Feature Extraction) yapar ve PCA ile 3 boyuta indirger.
# Sonuçlar features tablosundaki cnn_feat1, cnn_feat2, cnn_feat3 sütunlarına yazılır.

import sqlite3
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA

# ── Proje Yolları ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "music_analysis.db")

# ── CNN Parametreleri ──────────────────────────────────────────────────────
RESNET_FEATURE_DIM = 512   # ResNet18 son katman çıktı boyutu
PCA_COMPONENTS = 3          # PCA ile indirgenen boyut sayısı
IMAGE_SIZE = (224, 224)     # ResNet giriş boyutu

# ── ImageNet Normalizasyon Değerleri ───────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_connection() -> sqlite3.Connection:
    """SQLite veritabanına bağlantı açar."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_cnn_columns(conn: sqlite3.Connection) -> None:
    """features tablosuna cnn_feat1, cnn_feat2, cnn_feat3 sütunlarını ekler (yoksa).

    SQLite ALTER TABLE ... ADD COLUMN IF NOT EXISTS desteklemediği için
    her sütun için ayrı try-except kullanılır.
    """
    columns = ["cnn_feat1", "cnn_feat2", "cnn_feat3"]
    for col in columns:
        try:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} REAL;")
            conn.commit()
            print(f"  [OK] '{col}' sutunu eklendi.")
        except sqlite3.OperationalError:
            # Sutun zaten mevcut
            pass
    print("[BILGI] CNN ozellik sutunlari hazir.\n")


def load_resnet18() -> nn.Module:
    """Önceden eğitilmiş ResNet18 modelini yükler ve özellik çıkarıcı olarak hazırlar.

    - Sınıflandırma katmanı (fc) nn.Identity() ile değiştirilir.
    - Model değerlendirme moduna (eval) alınır.
    - Gradyan hesaplaması kapatılır (daha hızlı çıkarım).

    Returns:
        nn.Module: Özellik çıkarıcı olarak yapılandırılmış ResNet18 modeli.
    """
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()  # 512-d özellik vektörü döndür
    resnet.eval()
    print("[OK] ResNet18 modeli yuklendi (ozellik cikarici modda).\n")
    return resnet


def build_transforms() -> transforms.Compose:
    """ResNet'in beklediği formata uygun görsel dönüşüm pipeline'ı oluşturur.

    Pipeline:
        1. Resize(224, 224)
        2. ToTensor()
        3. Normalize (ImageNet istatistikleri)

    Not: RGBA → RGB dönüşümü load_and_transform_image() içinde yapılır.
    """
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def load_and_transform_image(
    image_path: str,
    transform: transforms.Compose,
) -> torch.Tensor:
    """Bir spektrogram görselini yükler, RGB'ye çevirir ve dönüşümleri uygular.

    RGBA formatındaki PNG dosyaları otomatik olarak RGB'ye dönüştürülür.

    Args:
        image_path: Görselin tam dosya yolu.
        transform:  Uygulanacak torchvision dönüşüm pipeline'ı.

    Returns:
        torch.Tensor: (1, 3, 224, 224) boyutunda model giriş tensörü.
    """
    img = Image.open(image_path)

    # RGBA veya diğer modlardan RGB'ye dönüştür
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Dönüşümleri uygula ve batch boyutu ekle
    img_tensor = transform(img)          # (3, 224, 224)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 224, 224)

    return img_tensor


def fetch_spectrogram_paths(conn: sqlite3.Connection) -> pd.DataFrame:
    """Veritabanından song_id ve spectrogram_path bilgilerini çeker.

    Yalnızca spectrogram_path değeri dolu olan kayıtlar döndürülür.
    """
    query = """
        SELECT
            f.song_id           AS song_id,
            f.spectrogram_path  AS spectrogram_path
        FROM features AS f
        WHERE f.spectrogram_path IS NOT NULL
          AND f.spectrogram_path != '';
    """
    df = pd.read_sql_query(query, conn)
    return df


def extract_features(
    df: pd.DataFrame,
    resnet: nn.Module,
    transform: transforms.Compose,
) -> tuple[list[int], np.ndarray]:
    """Her spektrogram görseli için ResNet18 özellik vektörünü çıkarır.

    Args:
        df:        song_id ve spectrogram_path içeren DataFrame.
        resnet:    Hazırlanmış ResNet18 modeli.
        transform: Görsel dönüşüm pipeline'ı.

    Returns:
        (song_ids, feature_matrix):
            song_ids       – Başarıyla işlenen şarkıların ID listesi.
            feature_matrix – (N, 512) boyutunda özellik matrisi.
    """
    song_ids: list[int] = []
    feature_vectors: list[np.ndarray] = []

    total = len(df)
    print(f"Toplam {total} spektrogram islenecek.\n")

    for idx, row in df.iterrows():
        song_id = row["song_id"]
        spec_path = row["spectrogram_path"]

        # Göreceli yolu tam yola çevir
        full_path = os.path.join(BASE_DIR, spec_path)

        if not os.path.isfile(full_path):
            print(f"  [UYARI] Dosya bulunamadi, atlaniyor: {spec_path}")
            continue

        print(f"  [{len(feature_vectors) + 1}/{total}] song_id={song_id}  ->  {spec_path}")

        try:
            img_tensor = load_and_transform_image(full_path, transform)

            # Gradyan hesaplaması kapalı olarak çıkarım yap
            with torch.no_grad():
                features = resnet(img_tensor)  # (1, 512)

            feature_np = features.squeeze().numpy()  # (512,)
            feature_vectors.append(feature_np)
            song_ids.append(song_id)

        except Exception as e:
            print(f"  [HATA] (song_id={song_id}): {e}")
            continue

    feature_matrix = np.array(feature_vectors)  # (N, 512)
    return song_ids, feature_matrix


def reduce_with_pca(feature_matrix: np.ndarray) -> np.ndarray:
    """512 boyutlu özellik matrisini PCA ile 3 boyuta indirger.

    Args:
        feature_matrix: (N, 512) boyutunda özellik matrisi.

    Returns:
        (N, 3) boyutunda sıkıştırılmış özellik matrisi.
    """
    pca = PCA(n_components=PCA_COMPONENTS)
    reduced = pca.fit_transform(feature_matrix)

    explained = pca.explained_variance_ratio_
    print(f"\nPCA Aciklanan Varyans Oranlari:")
    for i, ratio in enumerate(explained, start=1):
        print(f"  Bilesen {i}: {ratio:.4f}  ({ratio * 100:.2f}%)")
    print(f"  Toplam  : {sum(explained):.4f}  ({sum(explained) * 100:.2f}%)\n")

    return reduced


def save_to_database(
    conn: sqlite3.Connection,
    song_ids: list[int],
    reduced_features: np.ndarray,
) -> None:
    """PCA ile indirgenmiş özellikleri veritabanına kaydeder.

    Args:
        conn:             SQLite bağlantısı.
        song_ids:         İşlenen şarkıların ID listesi.
        reduced_features: (N, 3) boyutunda PCA çıktısı.
    """
    for i, song_id in enumerate(song_ids):
        feat1 = float(reduced_features[i, 0])
        feat2 = float(reduced_features[i, 1])
        feat3 = float(reduced_features[i, 2])

        conn.execute(
            """
            UPDATE features
            SET cnn_feat1 = ?, cnn_feat2 = ?, cnn_feat3 = ?
            WHERE song_id = ?;
            """,
            (feat1, feat2, feat3, song_id),
        )

    conn.commit()
    print(f"[OK] {len(song_ids)} sarkinin CNN ozellikleri veritabanina kaydedildi.")


def main() -> None:
    """Ana akış: model hazırla → özellik çıkar → PCA → veritabanına kaydet."""

    # ── Veritabanı bağlantısı ──
    conn = get_connection()

    try:
        # ── CNN sütunlarını garanti altına al ──
        ensure_cnn_columns(conn)

        # ── ResNet18 modelini yükle ──
        resnet = load_resnet18()

        # ── Görsel dönüşüm pipeline'ı ──
        transform = build_transforms()

        # ── Spektrogram yollarını çek ──
        df = fetch_spectrogram_paths(conn)

        if df.empty:
            print("Veritabaninda spectrogram_path bilgisi olan sarki bulunamadi.")
            print("Once 'generate_spectrograms.py' scriptini calistirin.")
            return

        # ── Özellik çıkarımı (512-d) ──
        song_ids, feature_matrix = extract_features(df, resnet, transform)

        if len(song_ids) == 0:
            print("Hicbir sarki icin ozellik cikarilamadi.")
            return

        print(f"\nOzellik matrisi boyutu: {feature_matrix.shape}")

        # ── PCA ile boyut indirgeme (512 → 3) ──
        reduced_features = reduce_with_pca(feature_matrix)

        # ── Veritabanına kaydet ──
        save_to_database(conn, song_ids, reduced_features)

        print("\nDerin Ogrenme (CNN) ozellik cikarimi tamamlandi ve veritabanina kaydedildi!")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
