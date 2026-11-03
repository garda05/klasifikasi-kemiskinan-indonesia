# Klasifikasi Kemiskinan Kabupaten/Kota

Aplikasi Streamlit untuk memprediksi label pada dataset Kaggle *Klasifikasi Tingkat Kemiskinan di Indonesia* menggunakan model DNN dan layanan Qwen/Groq.

## Menjalankan

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Sampel input CSV dapat diunduh dari tab **Unggah CSV**. CSV harus memuat sembilan kolom fitur yang dipakai model; kolom nama wilayah opsional.

## Secrets untuk AI service

Salin `.streamlit/secrets.toml.example` menjadi `.streamlit/secrets.toml`, lalu atur `GROQ_API_KEY`. Jangan commit `secrets.toml`. Di Streamlit Community Cloud, masukkan nilai yang sama ke **App settings > Secrets**.

## Informasi model

DNN dievaluasi pada 103 data uji stratified: accuracy 0,8155 dan F1 macro 0,7208. Qwen/Groq dievaluasi pada test set yang sama: accuracy 0,7379 dan F1 macro 0,6385. Hasil merupakan demonstrasi dataset agregat kabupaten/kota, bukan dasar penetapan bantuan sosial.
