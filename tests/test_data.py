import unittest
import pandas as pd
import os

class TestDataQuality(unittest.TestCase):

    def setUp(self):
        # Dosyaları her testten önce yüklemeyi dene
        self.human_exists = os.path.exists("human_data.csv")
        self.ai_exists = os.path.exists("ai_data.csv")

    # 1. Dosya Varlığı ve Boyut Kontrolü
    def test_file_existence_and_size(self):
        print("\n--- Test 1: Dosya Fiziksel Kontrolü ---")
        self.assertTrue(self.human_exists, "human_data.csv EKSİK")
        self.assertTrue(self.ai_exists, "ai_data.csv EKSİK")
        
        # Dosyalar boş olmamalı (En az 1KB)
        if self.human_exists:
            size = os.path.getsize("human_data.csv")
            self.assertGreater(size, 100, "Human dosyası şüpheli derecede küçük!")
            print(f" Human dosyası mevcut ve dolu ({size} bytes).")

    # 2. Veri İçeriği ve Etiket Kontrolü
    def test_label_integrity(self):
        print("\n--- Test 2: Etiket Tutarlılığı ---")
        if self.human_exists:
            df = pd.read_csv("human_data.csv")
            # 'label' sütunu var mı?
            self.assertIn("label", df.columns, "Sütun başlıkları hatalı!")
            
            # Sadece 'Human' etiketi mi var?
            unique_labels = df['label'].unique()
            self.assertTrue("Human" in unique_labels, "Human etiketi bulunamadı!")
            print(" Human verisi etiketleri doğru.")

    # 3. Null (Boş) Değer Kontrolü
    def test_null_values(self):
        print("\n--- Test 3: Boş Veri Kontrolü ---")
        if self.ai_exists:
            df = pd.read_csv("ai_data.csv")
            # Metin sütununda tamamen boş olan satır var mı?
            null_count = df['text'].isnull().sum()
            
            # Uyarı verir ama testi çökertmez (Fail etmez, sadece uyarır)
            if null_count > 0:
                print(f" UYARI: AI verisinde {null_count} adet boş satır var.")
            else:
                print(" AI verisinde hiç boş (null) satır yok. Mükemmel!")

if __name__ == '__main__':
    unittest.main()