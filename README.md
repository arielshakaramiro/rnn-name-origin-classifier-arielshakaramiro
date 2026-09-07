# RNN Name-Origin Classifier

Klasifikasi asal bahasa/negara dari sebuah nama menggunakan **character-level vanilla Recurrent Neural Network (RNN)** yang dibangun dengan PyTorch. Setiap nama diproses huruf demi huruf; RNN menyimpan konteks antar-huruf lewat hidden state untuk memprediksi salah satu dari 18 kategori bahasa/asal.

## Arsitektur

Model mengimplementasikan persamaan hidden state RNN standar:

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t)
```

| Komponen | Ukuran |
|---|---|
| Input size (jumlah karakter unik) | 57 |
| Hidden size | 128 |
| Output size (jumlah kategori) | 18 |
| Optimizer | Manual SGD (learning rate 0.005) |
| Loss function | Negative Log-Likelihood (NLLLoss) |
| Training method | Backpropagation Through Time (BPTT) |

## Hasil Training (Terverifikasi)

| Checkpoint | Loss |
|---|---|
| Iterasi ke-1.000 | 2.87 |
| Iterasi ke-50.000 | 1.28 |
| Iterasi ke-100.000 | 0.83 |

Loss turun konsisten sepanjang 100.000 iterasi training, menandakan model berhasil mempelajari pola karakter per kategori bahasa.

## Contoh Prediksi

**Nama dari dalam distribusi dataset (top-3 prediksi):**

| Nama | Top-3 Prediksi |
|---|---|
| Abdiel | Spanish, French, English |
| Jackson | Scottish, English, Russian |
| Satoshi | Japanese, Polish, Italian |

**Nama modern/populer (di luar distribusi dataset asli):**

| Nama | Prediksi | Confidence |
|---|---|---|
| albert | Scottish | 40.2% |
| ronaldo | Italian | 56.1% |
| messi | Italian | 74.7% |
| mbappe | Czech | 21.8% |
| ibra | Scottish | 62.4% |
| loli | Italian | 94.1% |

### Keterbatasan

Dataset training berisi nama-nama historis yang dikelompokkan per bahasa/asal, bukan basis data kewarganegaraan tokoh publik. Karena itu, nama-nama pesepakbola modern (`ronaldo`, `mbappe`, `messi`) sering salah diklasifikasikan — ini adalah contoh nyata dari **distribution shift** antara data latih dan data uji, bukan bug pada implementasi model.

## Cara Menjalankan

1. Buka `rnn_name_classifier.ipynb` di Google Colab atau Jupyter environment dengan PyTorch terpasang.
2. Jalankan cell secara berurutan — notebook akan otomatis meng-clone dataset yang dibutuhkan.
3. Model akan otomatis dilatih (±100.000 iterasi) dan disimpan sebagai `rnn.pt` di akhir notebook.
4. Untuk inference pada nama baru, gunakan fungsi `predict_with_confidence(nama)` pada bagian Inference.

## Dataset

Dataset nama-per-kategori-bahasa diambil dari repo publik: [Hands-On-NLP-Super-Class-Batch1](https://github.com/Muhammad-Ikhwan-Fathulloh/Hands-On-NLP-Super-Class-Batch1).

## Struktur Repo

```
.
├── rnn_name_classifier.ipynb   # Notebook lengkap: training, evaluasi, inference
└── README.md
```
