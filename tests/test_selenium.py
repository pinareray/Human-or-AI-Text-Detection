import unittest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

class TestWebInterface(unittest.TestCase):

    def setUp(self):
        # 1. Tarayıcıyı Başlat (Chrome)
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless') # İsterseniz tarayıcıyı görmeden arkada çalıştırır
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
    def test_app_opens(self):
        print("\n--- Test: Selenium ile Arayüz Kontrolü ---")
        
        # 2. Siteye Git (Streamlit genelde 8501 portunda çalışır)
        # ÖNEMLİ: Bu test çalışırken terminalde 'streamlit run app.py' açık olmalıdır!
        self.driver.get("http://localhost:8501")
        
        # Sitenin yüklenmesi için 3 saniye bekle
        time.sleep(3)
        
        # 3. Başlığı Kontrol Et
        # Streamlit sayfalarının başlığı genelde proje adıdır
        sayfa_basligi = self.driver.title
        print(f"Sitenin Başlığı: {sayfa_basligi}")
        
        # Başlığın içinde "Text Origin" veya "Streamlit" geçiyor mu?
        # (app.py içindeki page_title kısmına ne yazdıysanız o çıkar)
        self.assertTrue("Text Origin" in sayfa_basligi or "Streamlit" in sayfa_basligi)
        print(" Arayüz başarıyla açıldı ve başlık doğrulandı.")

    def tearDown(self):
        # 4. Tarayıcıyı Kapat
        self.driver.quit()

if __name__ == "__main__":
    unittest.main()