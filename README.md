Asystent Przepisów (Streamlit)

Opis:
Aplikacja pomaga znaleźć przepisy na podstawie składników, pobiera przepisy z internetu (TheMealDB) i oferuje prosty czat. UI i komunikaty są po polsku.

Kluczowe zmiany i cechy (wykonane):
- Zamiast lokalnego pliku CSV przepisy pobierane są z TheMealDB API.
- Tłumaczenia pól przepisów na polski: kolejność fallbacków — Hugging Face (jeśli token), OpenAI (jeśli klucz), następnie prosty słownikowy fallback.
- Usunięto z nazwy aplikacji i widżetów odniesienia do "RAG" — aplikacja nazywa się teraz "Asystent Przepisów".
- Interfejs: wyniki wyświetlane są w trzech kolumnach; każdy przepis pokazuje tytuł, przybliżone makroskładniki na 100g, licznik składników (posiadane / łączna liczba) i pełną listę składników oraz jedną sekcję instrukcji (pierwsze niepuste pole: instructions / method / preparation).
- Brakujące składniki nie są kolorowane; listy składników są deduplikowane i wyświetlane raz.
- Przepisy bez tytułu są pomijane.
- Dodano prostą bazę NUTRITION_DB i heurystykę _estimate_macros do przybliżenia białka/węglowodanów/tłuszczu na 100g (heurystyczne — zależne od jakości danych i jednostek w źródle).
- Poprawione parsowanie składników i instrukcji (usuwanie powtarzających się linii, oczyszczanie tytułów).
- Usunięto komentarze programistyczne z plików źródłowych dla czystości kodu.

Uruchomienie:
1. Utwórz i aktywuj virtualenv dla Pythona.
2. pip install -r requirements.txt
3. streamlit run app.py

Konfiguracja modelu/kluczy:
- W pasku bocznym aplikacji można podać klucz OpenAI, aby włączyć generatywny czat (gpt-3.5-turbo domyślnie). Jeśli klucz nie jest podany, czat działa w trybie lokalnym (lista przepisów jako odpowiedzi).
- Token HF (opcjonalny) umożliwia użycie tłumaczeń Helsinki-NLP/opus-mt-en-pl przez Hugging Face Inference API.

Ograniczenia i uwagi:
- Estymacja makroskładników to uproszczona heurystyka oparta na lokalnym NUTRITION_DB; dla dokładnych wartości zalecane jest użycie zewnętrznego API (Edamam/USDA/OpenFoodFacts).
- W środowiskach z ograniczonym dostępem do internetu lub problemami DNS (np. brak dostępu do api-inference.huggingface.co) mechanizmy tłumaczeń zdalnych mogą się nie powieść — aplikacja użyje fallbacków.
- Gemeni/inne komercyjne modele wymagają prawidłowych poświadczeń i endpointów; aplikacja nie obejmuje obejścia autoryzacji.

Debug i rozwój:
- Jeśli występują problemy z wyświetlaniem lub tłumaczeniem, wklej log błędu i przykładowy obiekt przepisu (dictionary) do issue/bug reportu.

Autor:
- Wprowadzone zmiany wykonane przez dewelopera projektu (modyfikacje UI, tłumaczeń, estymacji makroskładników i obsługi modeli).
