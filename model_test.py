import google.generativeai as genai
import os
from dotenv import load_dotenv
#KUllanabileceğimiz modelleri listeledik ai verisini çekmek için. 
# 1. .env dosyasını yükle
load_dotenv()

# 2. .env içindeki 'GOOGLE_API_KEY' değişkenini al
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("HATA: .env dosyasında GOOGLE_API_KEY bulunamadı veya dosya okunamadı.")
else:
    print("API Key başarıyla okundu. Modeller aranıyor...")
    
    # 3. Gemini'yi yapılandır
    genai.configure(api_key=api_key)

    try:
        print("\n--- KULLANILABİLİR MODELLER ---")
        found = False
        for m in genai.list_models():
            # Sadece içerik üretebilen modelleri listele
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ İsim: {m.name}")
                found = True
        
        if not found:
            print("Hiçbir uygun model bulunamadı. API Key yetkilerini kontrol edin.")
            
    except Exception as e:
        print(f"\nBir hata oluştu: {e}")
        print("İpucu: API Key yanlış olabilir veya internet bağlantısında sorun olabilir.")