Asystent Przepisów z RAG (Streamlit)

Aplikacja pomaga znaleźć przepisy na podstawie składników i oferuje okienko czatu wspierane przez Retrieval-Augmented Generation (RAG) z użyciem otwartego modelu generatywnego.

Instalacja (Windows):
1. Utwórz i aktywuj virtualenv dla Pythona.
2. pip install -r requirements.txt
3. Skonfiguruj model w pasku bocznym aplikacji. Opcje:
   - huggingface_inference: podaj token HF i opcjonalny endpoint modelu (np. https://api-inference.huggingface.co/models/your-model).
   - custom_endpoint: wskaż lokalny serwer LLM (text-generation-inference, Ollama itp.) akceptujący JSON {"prompt":...}.
   - gemini: użyj kompatybilnego endpointu dla Geminiego (np. odpowiedniego endpointu Google Cloud) i poświadczeń. Skonfiguruj prawidłowy endpoint w aplikacji.

Uruchomienie:
streamlit run app.py

Uwagi:
- Przepisy pobierane są z publicznego API TheMealDB (brak konieczności przechowywania lokalnego pliku CSV).
- Jeśli biblioteka sentence-transformers jest zainstalowana, aplikacja użyje semantycznych wektorów do lepszego dopasowania. W przeciwnym razie stosowany jest prosty ranking oparty na zgodności składników.
- Tłumaczenia na język polski: aplikacja domyślnie używa modelu Hugging Face (Helsinki-NLP/opus-mt-en-pl) jeśli dostarczysz token HF w pasku bocznym — to zapewnia lepsze, wierne tłumaczenia zachowujące liczby i miary. Jeśli nie podasz tokena HF, aplikacja spróbuje użyć skonfigurowanego modelu (Gemini / inny) lub fallbackowego słownika.
- Dla modelu otwartego zalecane jest uruchomienie lokalnego serwera lub użycie Hugging Face Inference API.
