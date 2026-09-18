# IndoTextMiner

Aplikasi web analisis teks Bahasa Indonesia berbasis [Streamlit](https://streamlit.io/) — alternatif ringan untuk **KH Coder** yang berfokus pada analisis linguistik komputasi dan eksplorasi data teks.

## Fitur Utama

| Tab | Fitur | Deskripsi |
|-----|-------|-----------|
| 📊 Frekuensi Kata | Word Frequency List | Daftar & grafik kata yang paling sering muncul |
| 🔍 Konteks Kata | KWIC (Keyword in Context) | Melihat konteks penggunaan kata tertentu dalam dokumen |
| 🌐 Jaringan Ko-okurensi | Co-occurrence Network | Visualisasi interaktif hubungan antar kata yang sering muncul bersamaan |
| 📈 Eksplorasi Ko-okurensi | MDS & Hierarchical Clustering | Pengelompokan kata berdasarkan kemiripan pola kemunculan |
| 🗺️ Analisis Korespondensi | Correspondence Analysis | Memetakan hubungan kata vs kategori dalam ruang 2D |
| 🏷️ Kata Khas per Bagian | Characteristic Words / Crosstab | Menemukan kata ciri khas tiap kelompok/kategori |

## Format Berkas yang Didukung

- `CSV`
- `Excel` (`.xls`, `.xlsx`)
- `Teks` (`.txt`)
- `Markdown` (`.md`)

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Kemudian buka URL yang muncul di terminal (biasanya `http://localhost:8501`).

## Cara Penggunaan

1. Unggah berkas dokumen melalui sidebar.
2. Pilih kolom teks utama (dan kolom kategori opsional).
3. Opsional: aktifkan *Stemming Sastrawi* untuk normalisasi kata (lebih lambat).
4. Jelajahi hasil analisis melalui keenam tab yang tersedia.

## Teknologi

- **Streamlit** — antarmuka web
- **Pandas / NumPy** — pengolahan data
- **Sastrawi** — stopword & stemming Bahasa Indonesia
- **scikit-learn** — vektorisasi teks, MDS, clustering
- **NetworkX + Pyvis** — jaringan ko-okurensi
- **Plotly** — visualisasi interaktif
- **prince** — analisis korespondensi
