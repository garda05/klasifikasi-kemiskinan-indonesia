import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from classifier import classify, load_config

st.set_page_config(page_title="Klasifikasi Kemiskinan Kabupaten/Kota", layout="wide")
ROOT = Path(__file__).parent
FEATURES = [
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
    "Indeks Pembangunan Manusia",
    "Umur Harapan Hidup (Tahun)",
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
    "Tingkat Pengangguran Terbuka",
    "Tingkat Partisipasi Angkatan Kerja",
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
]
LABEL_NAMES = {0: "Tidak Miskin", 1: "Miskin"}
DISPLAY = {
    FEATURES[0]: "Rata-rata lama sekolah penduduk usia 15+ (tahun)",
    FEATURES[1]: "Pengeluaran per kapita disesuaikan (ribu rupiah/orang/tahun)",
    FEATURES[2]: "Indeks Pembangunan Manusia (IPM)",
    FEATURES[3]: "Umur harapan hidup (tahun)",
    FEATURES[4]: "Rumah tangga dengan sanitasi layak (%)",
    FEATURES[5]: "Rumah tangga dengan akses air minum layak (%)",
    FEATURES[6]: "Tingkat pengangguran terbuka (%)",
    FEATURES[7]: "Tingkat partisipasi angkatan kerja (%)",
    FEATURES[8]: "PDRB harga konstan menurut pengeluaran (rupiah)",
}


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model_kemiskinan_dnn.pkl")


@st.cache_data
def load_info():
    return json.loads((ROOT / "model_info.json").read_text(encoding="utf-8"))


def get_api_key():
    try:
        return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


def predict_service(row):
    from groq import Groq

    cfg = load_config(ROOT / "config.json")
    client = Groq(api_key=get_api_key())
    return classify(client, cfg, row)


info = load_info()
model = load_model()
ai_eval = load_config(ROOT / "config.json")["evaluasi"]
with st.sidebar:
    st.header("Informasi model")
    st.write(info["nama_model"])
    st.metric("Accuracy (data uji)", f"{info['metrik']['accuracy']:.1%}")
    st.metric("F1 macro (data uji)", f"{info['metrik']['f1_macro']:.3f}")
    st.caption(f"{info['metrik']['n_test']} data uji, split stratified 80:20")
    st.write("AI model service: Qwen melalui Groq")
    st.metric("Qwen accuracy (test)", f"{ai_eval['accuracy']:.1%}")
    st.metric("Qwen F1 macro (test)", f"{ai_eval['f1_macro']:.4f}")
    st.caption(f"{ai_eval['n_test']} data uji; jawaban tidak valid: {ai_eval['invalid']}")

st.title("Klasifikasi Tingkat Kemiskinan Kabupaten/Kota")
st.write("Masukkan indikator sosial-ekonomi untuk memperkirakan kelas berdasarkan label pada dataset Kaggle.")
st.warning("Hasil ini untuk praktikum, bukan penetapan penerima bantuan atau keputusan kebijakan.")
tab_input, tab_csv = st.tabs(["Input satu wilayah", "Unggah CSV"])

with tab_input:
    with st.form("poverty_form"):
        col_a, col_b = st.columns(2)
        region = st.text_input("Nama kabupaten/kota (opsional)")
        values = {}
        for idx, feature in enumerate(FEATURES):
            lo, hi = info["rentang_fitur"][feature]
            initial = info["default_fitur"][feature]
            with (col_a if idx % 2 == 0 else col_b):
                values[feature] = st.number_input(
                    DISPLAY[feature], min_value=float(lo), max_value=float(hi),
                    value=float(initial), format="%.4f", key=f"input_{idx}",
                )
        submitted = st.form_submit_button("Prediksi kelas")
    if submitted:
        row = {**values, "Kab/Kota": region}
        frame = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
        pred = int(model.predict(frame)[0])
        proba = pd.Series(model.predict_proba(frame)[0], index=[LABEL_NAMES[int(c)] for c in model.classes_])
        st.subheader("Prediksi model library (DNN)")
        st.success(f"{region + ': ' if region else ''}**{LABEL_NAMES[pred]}**")
        st.bar_chart(proba)
        st.subheader("Prediksi AI model service")
        if not get_api_key():
            st.info("Untuk mengaktifkan prediksi Groq, atur GROQ_API_KEY pada .streamlit/secrets.toml atau Streamlit Cloud Secrets.")
        else:
            try:
                with st.spinner("Menghubungi layanan AI..."):
                    service_pred, raw = predict_service(row)
                if service_pred is None:
                    st.warning(f"Jawaban belum cocok dengan label yang diizinkan: {raw}")
                else:
                    st.write(f"Prediksi AI service: **{LABEL_NAMES[service_pred]}**")
                    with st.expander("Jawaban model"):
                        st.code(raw)
            except Exception as exc:
                st.error(f"Pemanggilan AI service gagal: {exc}")

with tab_csv:
    st.write("Unggah CSV dengan 9 kolom fitur numerik berikut. Kolom wilayah dapat disertakan untuk identitas tampilan.")
    st.download_button("Unduh sampel input CSV", (ROOT / "contoh_data_baru.csv").read_bytes(),
                       "contoh_data_baru.csv", "text/csv")
    uploaded = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded:
        data = pd.read_csv(uploaded)
        missing = [feature for feature in FEATURES if feature not in data.columns]
        if missing:
            st.error("Kolom wajib tidak ditemukan: " + ", ".join(missing))
        elif data.empty:
            st.error("CSV tidak berisi baris.")
        else:
            st.dataframe(data, use_container_width=True)
            run_ai = False
            if get_api_key():
                run_ai = st.checkbox("Sertakan prediksi AI service (maksimum 20 baris, ada jeda antarpermintaan)")
            if st.button("Jalankan prediksi"):
                output = data.copy()
                output["prediksi_dnn"] = model.predict(data[FEATURES]).astype(int)
                output["label_dnn"] = output["prediksi_dnn"].map(LABEL_NAMES)
                output["keyakinan_dnn"] = model.predict_proba(data[FEATURES]).max(axis=1).round(4)
                if run_ai:
                    count = min(len(data), 20)
                    results = []
                    progress = st.progress(0)
                    for i in range(count):
                        try:
                            label, _ = predict_service(data.iloc[i].to_dict())
                            results.append(LABEL_NAMES[label] if label is not None else "Jawaban tidak valid")
                        except Exception as exc:
                            results.append(f"Error: {exc}")
                        progress.progress((i + 1) / count)
                        if i + 1 < count:
                            time.sleep(2.1)
                    output["prediksi_ai_service"] = "Tidak diproses (batas 20 baris)"
                    output.loc[output.index[:count], "prediksi_ai_service"] = results
                st.dataframe(output, use_container_width=True)
                st.download_button("Unduh hasil prediksi", output.to_csv(index=False).encode("utf-8"),
                                   "hasil_prediksi_kemiskinan.csv", "text/csv")
