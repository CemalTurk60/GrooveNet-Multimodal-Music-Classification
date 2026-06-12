# 🎵 GrooveNet

**Müzik Analizi & Derin Öğrenme ile Şarkı Sınıflandırma Projesi**

Ses dosyalarından akustik özellik çıkarımı, Mel-Spektrogram üretimi ve
CNN tabanlı özellik çıkarımı ile müzikal içerikleri analiz eden uçtan uca (end-to-end) pipeline.

---

## 📁 Proje Yapısı

```
GrooveNet/
│
├── configs/                    # Proje konfigürasyon dosyaları (YAML/JSON)
│
├── data/                       # ── VERİ KATMANI ──
│   ├── raw_audio/              #    🎧 Şarkıları BURAYA AT (.mp3, .wav)
│   ├── spectrograms/           #    🖼️ Üretilen Mel-Spektrogram görselleri (otomatik)
│   ├── augmented/              #    🔄 Veri artırma çıktıları (otomatik)
│   └── processed/              #    📊 Ara işlem çıktıları (CSV, NPY vb.)
│
├── db/                         # ── VERİTABANI ──
│   └── music_analysis.db       #    SQLite veritabanı (otomatik oluşur)
│
├── logs/                       # Eğitim logları, TensorBoard çıktıları
│
├── models/                     # ── MODEL KATMANI ──
│   ├── checkpoints/            #    Eğitim sırasındaki model snapshot'ları
│   └── exported/               #    Üretim modelleri (ONNX, TorchScript)
│
├── notebooks/                  # Jupyter Notebook'ları (EDA, prototip)
│
├── reports/                    # ── ÇIKTI KATMANI ──
│   └── figures/                #    Analiz grafikleri ve görselleştirmeler
│
├── src/                        # ── KAYNAK KOD ──
│   ├── generate_spectrograms.py    # Mel-Spektrogram üretim pipeline'ı
│   └── cnn_features.py             # ResNet18 özellik çıkarımı + PCA
│
├── tests/                      # Birim ve entegrasyon testleri
│
├── .gitignore                  # Git izleme kuralları
├── requirements.txt            # Python bağımlılıkları
└── README.md                   # Bu dosya
```

---

## 🚀 Hızlı Başlangıç

### 1. Sanal Ortam Oluştur

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
```

### 2. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 3. Şarkıları Ekle

`.mp3` veya `.wav` formatındaki ses dosyalarını şu klasöre at:

```
data/raw_audio/
```

### 4. Pipeline'ı Çalıştır

```bash
# Adım 1 — Mel-Spektrogram üret
python src/generate_spectrograms.py

# Adım 2 — CNN özellik çıkarımı
python src/cnn_features.py
```

---

## 🧠 Pipeline Akışı

```
🎵 raw_audio/        Ses dosyaları (.mp3, .wav)
       │
       ▼
🔬 DSP Analizi       Sessizlik kırpma, Mel-Spektrogram hesaplama
       │
       ▼
🖼️ spectrograms/     Saf CNN görselleri (224×224, eksensiz)
       │
       ▼
🤖 ResNet18           512-d özellik vektörü çıkarımı
       │
       ▼
📉 PCA                512 → 3 boyut indirgeme
       │
       ▼
🗄️ SQLite DB          features tablosuna kayıt
```
