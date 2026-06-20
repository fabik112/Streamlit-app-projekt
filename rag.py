import os
import json
import requests
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

NUTRITION_DB = {
    'tomato': (0.9, 3.9, 0.2),
    'pasta': (5.0, 75.0, 1.5),
    'olive oil': (0.0, 0.0, 100.0),
    'garlic': (6.4, 33.1, 0.5),
    'basil': (3.2, 2.7, 0.6),
    'egg': (13.0, 1.1, 11.0),
    'eggs': (13.0, 1.1, 11.0),
    'onion': (1.1, 9.3, 0.1),
    'pepper': (0.9, 6.0, 0.2),
    'chickpea': (19.3, 61.0, 6.0),
    'cucumber': (0.7, 3.6, 0.1),
    'bread': (8.0, 49.0, 3.2),
    'cheese': (25.0, 1.3, 33.0),
    'butter': (0.5, 0.1, 81.0),
    'broccoli': (2.8, 6.6, 0.4),
    'carrot': (0.9, 9.6, 0.2),
    'soy sauce': (8.0, 4.9, 0.1),
    'flour': (10.0, 76.0, 1.5),
    'milk': (3.4, 5.0, 3.6),
    'sugar': (0.0, 100.0, 0.0),
    'rice': (2.7, 28.0, 0.3),
    'potato': (2.0, 17.0, 0.1),
    'chicken': (27.0, 0.0, 3.6),
    'salt': (0.0, 0.0, 0.0),
}

POL_TO_EN = {
    'pomidor': 'tomato', 'pomidory': 'tomato', 'makaron': 'pasta', 'oliwa': 'olive oil', 'oliwa z oliwek': 'olive oil',
    'czosnek': 'garlic', 'bazylia': 'basil', 'jajko': 'egg', 'jajka': 'eggs', 'cebula': 'onion', 'papryka': 'pepper',
    'ciecierzyca': 'chickpea', 'ogórek': 'cucumber', 'chleb': 'bread', 'ser': 'cheese', 'masło': 'butter',
    'brokuły': 'broccoli', 'brokuł': 'broccoli', 'marchew': 'carrot', 'sos sojowy': 'soy sauce', 'soja': 'soy sauce',
    'mąka': 'flour', 'maka': 'flour', 'mleko': 'milk', 'cukier': 'sugar', 'ryż': 'rice', 'ziemniak': 'potato', 'ziemniaki': 'potatoes',
    'kurczak': 'chicken', 'sól': 'salt', 'pieprz': 'pepper'
}
EN_TO_POL = {v: k for k, v in POL_TO_EN.items()}

class RAGAssistant:
    """Asystent RAG pobierający przepisy z TheMealDB i tłumaczący je na polski. Składniki wejściowe można podać po polsku."""
    def __init__(self, model_type=None, api_token=None, endpoint=None, hf_token: str = None):
        self.model_type = model_type
        self.api_token = api_token
        self.endpoint = endpoint
        self.hf_token = hf_token
        self.embedder = None
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.embedder = None

    def _translate_ings_to_english(self, ingredients_polish: str) -> str:
        """Mapuje krótkie nazwy składników z polskiego na angielski używając słownika, fallback - zwraca oryginał."""
        parts = [p.strip().lower() for p in ingredients_polish.split(',') if p.strip()]
        translated = []
        for p in parts:
            if p in POL_TO_EN:
                translated.append(POL_TO_EN[p])
            else:
                words = p.split()
                mapped = []
                for w in words:
                    mapped.append(POL_TO_EN.get(w, w))
                translated.append(' '.join(mapped))
        return ','.join(translated)

    def _hf_translate(self, text: str) -> (str, str):
        """Tłumaczenie z użyciem modelu HF opus-mt-en-pl. Zwraca krotkę (translated_text, error_message). Jeśli brak błędu, error_message jest pusty."""
        if not self.hf_token:
            return text, 'Brak tokena HF'
        model = 'Helsinki-NLP/opus-mt-en-pl'
        url = f'https://api-inference.huggingface.co/models/{model}'
        headers = {'Authorization': f'Bearer {self.hf_token}'}
        payload = {"inputs": text}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
        except Exception as e:
            return text, f'Błąd sieciowy HF: {e}'
        if not resp.ok:
            return text, f'HTTP HF: {resp.status_code} {resp.text}'
        try:
            out = resp.json()
            if isinstance(out, list) and len(out)>0:
                first = out[0]
                if isinstance(first, dict):
                    return first.get('translation_text') or first.get('generated_text') or str(first), ''
                return str(first), ''
            if isinstance(out, dict):
                return out.get('translation_text') or out.get('generated_text') or json.dumps(out), ''
            return str(out), ''
        except Exception as e:
            return text, f'Błąd parsowania odpowiedzi HF: {e}'

    def _translate_to_polish(self, text: str) -> str:
        """Przetłumacz tekst na polski. Kolejność: Hugging Face (jeśli hf_token), OpenAI (jeśli model_type=='openai' i api_token), inny skonfigurowany model, słownikowy fallback."""
        if not text:
            return ''
        if self.hf_token:
            try:
                translated, err = self._hf_translate(text)
                if not err and translated:
                    return translated.strip()
            except Exception:
                pass
        if self.model_type == 'openai' and self.api_token:
            prompt = (
                "Przetłumacz poniższy tekst na język polski. Zachowaj formatowanie, wszystkie liczby i miary (np. '200 g', '1 cup', '2 tbsp'), "
                "oraz listy składników. Zwróć tylko przetłumaczony tekst bez dodatkowych objaśnień:\n\n" + text
            )
            out = self._call_model(prompt, override_type='openai', override_token=self.api_token)
            if isinstance(out, str) and not out.lower().startswith(('błąd', 'brak', 'wyjątek')):
                return out.strip()
        if self.model_type and (self.api_token or self.endpoint):
            prompt = f"Przetłumacz na język polski zachowując formatowanie i wszystkie liczby/miary:\n\n{text}"
            out = self._call_model(prompt)
            if isinstance(out, str) and not out.lower().startswith(('błąd', 'brak', 'wyjątek')):
                return out.strip()
        res = text
        for en, pl in EN_TO_POL.items():
            res = res.replace(en, pl)
            res = res.replace(en.capitalize(), pl)
        return res

    def fetch_recipes_online(self, ingredients: str, max_results: int = 5) -> List[dict]:
        """Pobiera przepisy z TheMealDB. Przyjmuje składniki w języku polskim lub angielskim."""
        if not ingredients or not ingredients.strip():
            return []
        eng_ings = self._translate_ings_to_english(ingredients)
        first_ing = eng_ings.split(',')[0].strip() if eng_ings else ingredients.split(',')[0].strip()
        if not first_ing:
            return []
        search_url = f"https://www.themealdb.com/api/json/v1/1/filter.php?i={requests.utils.quote(first_ing)}"
        resp = requests.get(search_url)
        if not resp.ok:
            return []
        data = resp.json()
        meals = data.get('meals')
        if not meals:
            return []
        results = []
        count = 0
        for m in meals:
            if count >= max_results:
                break
            meal_id = m.get('idMeal')
            detail_url = f"https://www.themealdb.com/api/json/v1/1/lookup.php?i={meal_id}"
            r2 = requests.get(detail_url)
            if not r2.ok:
                continue
            detail = r2.json().get('meals')
            if not detail:
                continue
            d = detail[0]
            ingredients_list = []
            for i in range(1,21):
                ing = d.get(f'strIngredient{i}')
                measure = d.get(f'strMeasure{i}')
                if ing and ing.strip():
                    if measure and measure.strip():
                        ingredients_list.append(f"{ing.strip()} ({measure.strip()})")
                    else:
                        ingredients_list.append(ing.strip())
            instructions = d.get('strInstructions') or ''
            title = d.get('strMeal') or ''
            title_pl = self._translate_to_polish(title)
            ings_pl = self._translate_to_polish(', '.join(ingredients_list))
            instr_pl = self._translate_to_polish(instructions)
            results.append({
                'id': meal_id,
                'title': title_pl,
                'ingredients': ings_pl,
                'instructions': instr_pl,
                'source': d.get('strSource') or d.get('strYoutube') or ''
            })
            count += 1
        return results

    def search_recipes(self, ingredients: str, top_k: int =5) -> List[dict]:
        recipes = self.fetch_recipes_online(ingredients, max_results=max(10, top_k))
        if not recipes:
            return []
        user_parts = [w.strip().lower() for w in ingredients.split(',') if w.strip()]
        eng_translated = [w.strip().lower() for w in self._translate_ings_to_english(ingredients).split(',') if w.strip()]
        user_norm = set(user_parts + eng_translated)

        def normalize_item(item: str) -> str:
            import re
            s = re.sub(r"\([^)]*\)", "", item)
            s = re.sub(r"\d+\s*(g|kg|ml|l|tbsp|tsp|cups?)", "", s)
            return s.strip().lower()

        enhanced = []
        for r in recipes:
            raw_items = [it.strip() for it in r.get('ingredients','').split(',') if it.strip()]
            matched = []
            missing = []
            for it in raw_items:
                norm = normalize_item(it)
                found = False
                for u in user_norm:
                    if not u:
                        continue
                    if u in norm or norm in u:
                        found = True
                        break
                if found:
                    matched.append(it)
                else:
                    missing.append(it)
            r_copy = dict(r)
            r_copy['matched_ingredients'] = matched
            r_copy['missing_ingredients'] = missing
            r_copy['missing_count'] = len(missing)
            r_copy['matched_count'] = len(matched)
            r_copy['macros'] = self._estimate_macros(r_copy)
            enhanced.append(r_copy)

        enhanced.sort(key=lambda x: (x['missing_count'], -x['matched_count']))

        if self.embedder is not None:
            try:
                candidates = enhanced[:max(top_k, 10)]
                texts = [(r['title'] + '. ' + r['ingredients'] + '. ' + r['instructions']) for r in candidates]
                vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                qvec = self.embedder.encode([ingredients], convert_to_numpy=True, normalize_embeddings=True)
                sims = (vecs @ qvec[0]).tolist()
                scored = list(zip(sims, candidates))
                scored.sort(key=lambda x: x[0], reverse=True)
                ranked = [r for _, r in scored]
                return ranked[:top_k]
            except Exception:
                pass

        return enhanced[:top_k]

    def _estimate_macros(self, recipe: dict) -> dict:
            """Estimate macros (protein, carbs, fat) per 100g of the dish using simple heuristics and NUTRITION_DB."""
            import re
            raw_items = [it.strip() for it in recipe.get('ingredients','').split(',') if it.strip()]
            total_weight = 0.0
            prot = carbs = fat = 0.0
            for it in raw_items:
                m = re.search(r"\(([^)]*)\)", it)
                measure = m.group(1) if m else ''
                weight_g = None
                if measure:
                    num = re.search(r"([0-9]+\,?[0-9]*)", measure)
                    unit = measure.lower()
                    if num:
                        val = float(num.group(1).replace(',','.'))
                        if 'kg' in unit:
                            weight_g = val * 1000
                        elif 'g' in unit:
                            weight_g = val
                        elif 'ml' in unit:
                            weight_g = val
                        elif 'cup' in unit:
                            weight_g = val * 240
                        elif 'tbsp' in unit or 'tablespoon' in unit:
                            weight_g = val * 15
                        elif 'tsp' in unit:
                            weight_g = val * 5
                        else:
                            weight_g = val
                if weight_g is None:
                    weight_g = 100.0
                name = re.sub(r"\([^)]*\)", "", it).strip().lower()
                key = None
                for k in NUTRITION_DB.keys():
                    if k in name:
                        key = k
                        break
                if key is None:
                    for pol, en in POL_TO_EN.items():
                        if pol in name:
                            if en in NUTRITION_DB:
                                key = en
                                break
                if key is None:
                    continue
                p100, c100, f100 = NUTRITION_DB.get(key, (0.0,0.0,0.0))
                prot += p100 * (weight_g / 100.0)
                carbs += c100 * (weight_g / 100.0)
                fat += f100 * (weight_g / 100.0)
                total_weight += weight_g
            if total_weight <= 0:
                return {'protein': 0.0, 'carbs': 0.0, 'fat': 0.0}
            factor = 100.0 / total_weight
            return {'protein': round(prot * factor, 1), 'carbs': round(carbs * factor, 1), 'fat': round(fat * factor, 1)}

    def _call_model(self, prompt: str, override_type: str = None, override_endpoint: str = None, override_token: str = None) -> str:
        """Wywołuje skonfigurowany model. Pozwala na nadpisanie typu/endpointu/tokenu dla fallbacków."""
        mtype = override_type or self.model_type
        token = override_token or self.api_token
        endpoint = override_endpoint or self.endpoint
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        try:
            if mtype == 'huggingface_inference':
                hf_endpoint = endpoint or 'https://api-inference.huggingface.co/models/gpt2'
                try:
                    resp = requests.post(hf_endpoint, headers=headers, json={"inputs": prompt, "options": {"wait_for_model": True}}, timeout=30)
                except requests.exceptions.RequestException as e:
                    return f"Błąd sieciowy HF: {e}"
                if not resp.ok:
                    return f"Błąd wywołania modelu HF: {resp.status_code} {resp.text}"
                out = resp.json()
                if isinstance(out, list) and len(out)>0 and isinstance(out[0], dict):
                    return out[0].get('generated_text') or out[0].get('translation_text') or str(out[0])
                return str(out)
            elif mtype == 'custom_endpoint':
                url = endpoint
                if not url:
                    return "Brak skonfigurowanego endpointu custom."
                try:
                    resp = requests.post(url, headers=headers, json={"prompt": prompt}, timeout=30)
                except requests.exceptions.RequestException as e:
                    return f"Błąd sieciowy custom endpoint: {e}"
                if not resp.ok:
                    return f"Błąd wywołania custom endpoint: {resp.status_code} {resp.text}"
                return resp.text
            elif mtype == 'gemini':
                url = endpoint
                if not url:
                    return "Brak skonfigurowanego endpointu dla Geminiego. Wprowadź adres endpointu w pasku bocznym aplikacji."
                try:
                    if 'generativelanguage' in url or 'googleapis.com' in url:
                        payload = {"prompt": {"text": prompt}}
                        resp = requests.post(url, headers=headers, json=payload, timeout=30)
                    else:
                        payload = {"model": "gemini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500}
                        resp = requests.post(url, headers=headers, json=payload, timeout=30)
                except requests.exceptions.RequestException as e:
                    return f"Błąd sieciowy Gemini: {e}"
                if not resp.ok:
                    return f"Błąd wywołania modelu Gemini: {resp.status_code} {resp.text}"
                data = resp.json()
                if isinstance(data, dict):
                    if 'candidates' in data and len(data['candidates'])>0:
                        cand = data['candidates'][0]
                        return cand.get('content') or cand.get('output') or cand.get('message') or json.dumps(cand)
                    if 'output' in data:
                        out = data.get('output')
                        if isinstance(out, list) and len(out)>0:
                            first = out[0]
                            if isinstance(first, dict):
                                return first.get('content') or first.get('text') or json.dumps(first)
                            return str(first)
                    if 'choices' in data and len(data['choices'])>0:
                        return data['choices'][0].get('message',{}).get('content') or data['choices'][0].get('text','')
                return json.dumps(data)
            elif mtype == 'openai':
                url = endpoint or 'https://api.openai.com/v1/chat/completions'
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role":"user","content":prompt}],
                    "max_tokens": 500
                }
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=30)
                except requests.exceptions.RequestException as e:
                    return f"Błąd sieciowy OpenAI: {e}"
                if resp.ok:
                    data = resp.json()
                    if 'choices' in data and len(data['choices'])>0:
                        choice = data['choices'][0]
                        if isinstance(choice, dict):
                            return choice.get('message',{}).get('content') or choice.get('text','')
                        return str(choice)
                    return json.dumps(data)
                return f"Błąd wywołania modelu OpenAI: {resp.status_code} {resp.text}"
            else:
                return "Brak poprawnej konfiguracji modelu."
        except Exception as e:
            return f"Wyjątek podczas wywołania modelu: {e}"


    def chat(self, user_message: str, context_ingredients: str = '', top_k: int = 3) -> str:
        contexts = self.search_recipes(context_ingredients, top_k=top_k) if context_ingredients else []
        if not self.model_type:
            if not contexts:
                return "Nie mam skonfigurowanego modelu generatywnego ani kontekstu przepisów. Podaj składniki lub zadaj pytanie, a wyszukam przepisy."
            parts = ["Nie mam skonfigurowanego modelu generatywnego. Oto znalezione przepisy:"]
            for i, r in enumerate(contexts[:top_k]):
                parts.append(f"{i+1}. {r.get('title','Brak tytułu')}")
                parts.append(f"Składniki: {r.get('ingredients','')}")
                instr = r.get('instructions','')
                parts.append(f"Instrukcje (skrócone): {instr[:400].strip() + ('...' if len(instr)>400 else '')}")
                parts.append("")
            parts.append("Jeśli chcesz pełną odpowiedź generowaną przez model, skonfiguruj endpoint lub token — obecnie działam lokalnie.")
            return "\n\n".join(parts)

        context_texts = []
        for c in contexts:
            context_texts.append(f"Tytuł: {c.get('title')}\nSkładniki: {c.get('ingredients')}\nInstrukcje: {c.get('instructions')}")
        prompt = "Jesteś pomocnym asystentem kuchennym. Użyj podanego kontekstu przepisów, aby odpowiedzieć na pytanie użytkownika. Jeśli kontekst jest pusty, odpowiedz ogólnie.\n\n"
        prompt += "\n\n--- Kontekst przepisów:\n" + "\n\n".join(context_texts) + "\n\n---\n"
        prompt += f"Pytanie użytkownika: {user_message}\nSzczegółowa, pomocna odpowiedź:" 
        res = self._call_model(prompt)
        if isinstance(res, str) and (res.startswith('Błąd') or res.startswith('Wyjątek') or res.startswith('Brak skonfigurowanego') or 'Failed to resolve' in res or 'NameResolutionError' in res):
            if self.endpoint and self.model_type != 'custom_endpoint':
                fb = self._call_model(prompt, override_type='custom_endpoint', override_endpoint=self.endpoint, override_token=self.api_token)
                if not (isinstance(fb, str) and (fb.startswith('Błąd') or fb.startswith('Wyjątek') or fb.startswith('Brak skonfigurowanego'))):
                    return fb
            if self.model_type != 'gemini' and self.api_token and self.endpoint:
                fb2 = self._call_model(prompt, override_type='gemini', override_endpoint=self.endpoint, override_token=self.api_token)
                if not (isinstance(fb2, str) and (fb2.startswith('Błąd') or fb2.startswith('Wyjątek') or fb2.startswith('Brak skonfigurowanego'))):
                    return fb2
            return res + '\n(Fallbacky nie powiodły się lub nie były skonfigurowane.)'
        return res
