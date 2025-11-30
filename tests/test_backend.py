import unittest
import pandas as pd
import joblib
import re
import os
import numpy as np

#Fatmanın yazdığı bazı fonksiyonları test etmek için unittest framework'ünü kullanıyoruz.
# --- Test Edilecek Fonksiyonun Kopyası ---
def clean_text_logic(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)      # Çoklu boşlukları sil
    text = re.sub(r'[^\w\s]', '', text)   # Noktalamaları sil
    return text.strip()

class TestBackendAdvanced(unittest.TestCase):

    # 1. Zorlu Metin Temizleme Testi (Edge Cases)
    def test_clean_text_edge_cases(self):
        print("\n--- Test 1: Zorlu Metin Temizliği ---")
        
        # Senaryo A: Sayılar ve Semboller
        girdi1 = "AI version 2.0 is #awesome!!!"
        beklenen1 = "ai version 20 is awesome"
        self.assertEqual(clean_text_logic(girdi1), beklenen1)
        
        # Senaryo B: Çoklu Boşluk ve Satır Sonu
        girdi2 = "Hello    World\n\nThis is   Test"
        beklenen2 = "hello world this is test"
        self.assertEqual(clean_text_logic(girdi2), beklenen2)
        
        # Senaryo C: Boş veya Sayısal Girdi
        self.assertEqual(clean_text_logic(None), "")
        self.assertEqual(clean_text_logic(12345), "")
        
        print("✅ Temizleme fonksiyonu tüm zorlu koşulları geçti.")

    # 2. Modellerin Yüklenebilirliği ve Tahmin Testi
    def test_model_prediction_capability(self):
        print("\n--- Test 2: Model Tahmin Simülasyonu ---")
        
        try:
            vec = joblib.load('vectorizer.pkl')
            nb = joblib.load('model_nb.pkl')
            
            # Rastgele bir girdi ile tahmin denemesi
            test_metni = ["This is a simple test abstract about deep learning."]
            vektor = vec.transform(test_metni)
            tahmin = nb.predict(vektor)
            
            # Tahmin sonucu 'AI' veya 'Human' olmalı
            self.assertIn(tahmin[0], ['AI', 'Human', 'İnsan Yazımı', 'Yapay Zeka'])
            print(f"✅ Model başarıyla yüklendi ve tahmin üretti: {tahmin[0]}")
            
        except FileNotFoundError:
            self.fail("❌ Model dosyaları bulunamadı! Önce eğitimi tamamlayın.")

    # 3. Vektör Boyut Testi
    def test_vectorizer_vocab(self):
        print("\n--- Test 3: Kelime Havuzu Kontrolü ---")
        if os.path.exists('vectorizer.pkl'):
            vec = joblib.load('vectorizer.pkl')
            # Kelime dağarcığı boş olmamalı
            vocab_size = len(vec.vocabulary_)
            self.assertGreater(vocab_size, 100, "Kelime havuzu çok küçük!")
            print(f"✅ Vektörleştirici {vocab_size} kelime öğrenmiş.")

if __name__ == '__main__':
    unittest.main()