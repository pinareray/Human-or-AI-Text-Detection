import streamlit as st
import joblib
import re
import pandas as pd

# Sayfa Ayarları
st.set_page_config(page_title="Human or AI Detector", page_icon="🕵️", layout="centered")

# --- 1. Modelleri Yükle ---
@st.cache_resource
def load_models():
    try:
        vec = joblib.load('vectorizer.pkl')
        nb = joblib.load('model_nb.pkl')
        lr = joblib.load('model_lr.pkl')
        rf = joblib.load('model_rf.pkl')
        return vec, nb, lr, rf
    except Exception as e:
        return None, None, None, None

vectorizer, nb_model, lr_model, rf_model = load_models()

# --- 2. Temizleme Fonksiyonu ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# --- 3. Arayüz Tasarımı ---
st.title("🕵️ Human or AI?")
st.markdown("### Makale Özeti Analiz Aracı")
st.info("User Story 5: Sonuçlar 3 farklı ML modeli ile oransal olarak gösterilmektedir.")

# Hata Kontrolü
if vectorizer is None:
    st.error("🚨 Model dosyaları (.pkl) bulunamadı! Lütfen terminale 'git pull' yazarak dosyaları çekin.")
    st.stop()

# Metin Giriş Alanı
user_input = st.text_area("Analiz edilecek metni girin:", height=150, placeholder="Example: Artificial intelligence is transforming the world...")

if st.button("ANALİZ ET 🚀", type="primary"):
    if not user_input:
        st.warning("Lütfen önce bir metin girin.")
    else:
        # Metni İşle
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        st.markdown("---")
        st.subheader("📊 Analiz Sonuçları")

        col1, col2, col3 = st.columns(3)

        # --- Model 1: Naive Bayes ---
        with col1:
            st.write("**Naive Bayes**")
            prob = nb_model.predict_proba(vectorized_text)[0]
            # Sınıf isimlerini kontrol et (AI nerede?)
            classes = list(nb_model.classes_)
            if 'AI' in classes:
                ai_score = prob[classes.index('AI')]
            else:
                ai_score = prob[0] 
            
            prediction = nb_model.predict(vectorized_text)[0]
            st.metric(label="Tahmin", value=prediction)
            st.progress(int(ai_score * 100))
            st.caption(f"AI İhtimali: %{ai_score*100:.1f}")

        # --- Model 2: Logistic Regression ---
        with col2:
            st.write("**Logistic Reg.**")
            prob = lr_model.predict_proba(vectorized_text)[0]
            classes = list(lr_model.classes_)
            if 'AI' in classes:
                ai_score = prob[classes.index('AI')]
            else:
                ai_score = prob[0]
            
            prediction = lr_model.predict(vectorized_text)[0]
            st.metric(label="Tahmin", value=prediction)
            st.progress(int(ai_score * 100))
            st.caption(f"AI İhtimali: %{ai_score*100:.1f}")

        # --- Model 3: Random Forest ---
        with col3:
            st.write("**Random Forest**")
            prob = rf_model.predict_proba(vectorized_text)[0]
            classes = list(rf_model.classes_)
            if 'AI' in classes:
                ai_score = prob[classes.index('AI')]
            else:
                ai_score = prob[0]
            
            prediction = rf_model.predict(vectorized_text)[0]
            st.metric(label="Tahmin", value=prediction)
            st.progress(int(ai_score * 100))
            st.caption(f"AI İhtimali: %{ai_score*100:.1f}")