import streamlit as st
import requests
from rag import RAGAssistant

st.set_page_config(page_title="Asystent Przepisów", layout="wide")
st.title("Asystent Przepisów")

st.sidebar.title("Asystent")
st.sidebar.write("Możesz wprowadzić klucz OpenAI, żeby włączyć czat generatywny (gpt-3.5-turbo). Jeśli nie podasz klucza, czat będzie działał w trybie lokalnym (tylko listowanie przepisów).")
openai_key = st.sidebar.text_input("OpenAI API Key (opcjonalnie)", type="password")

if openai_key and openai_key.strip():
    assistant = RAGAssistant(model_type='openai', api_token=openai_key.strip())
else:
    assistant = RAGAssistant()



st.header("Znajdź przepisy na podstawie składników")
ingredients = st.text_area("Wprowadź składniki które posiadasz (oddzielone przecinkami)")
cols = st.columns([1,3])
with cols[0]:
    top_k = st.number_input("Ile przepisów pokazać", min_value=1, max_value=10, value=5)
    search_btn = st.button("Szukaj przepisów")

if search_btn and ingredients.strip():
    with st.spinner("Wyszukiwanie przepisów..."):
        recipes = assistant.search_recipes(ingredients, top_k=top_k)
    st.subheader("Znalezione przepisy")
    if not recipes:
        st.info("Nie znaleziono przepisów. Spróbuj innych składników.")
    cols_per_row = 3
    row_cols = None
    valid_recipes = [r for r in recipes if (r.get('title') or '').strip()]
    if not valid_recipes:
        st.info("Brak znalezionych przepisów z nazwą.")
    else:
        cols_per_row = 3
        for idx, r in enumerate(valid_recipes):
            if idx % cols_per_row == 0:
                row_cols = st.columns(cols_per_row)
            col = row_cols[idx % cols_per_row]
            with col:
                title = (r.get('title') or '').strip()
                if '\n' in title:
                    title = title.split('\n')[0].strip()
                lower_title = title.lower()
                for marker in ['składniki', 'skladniki', 'sposób przygotowania', 'sposob przygotowania', 'instructions', 'ingredients']:
                    if marker in lower_title:
                        idx = lower_title.find(marker)
                        title = title[:idx].strip()
                        break

                macros = r.get('macros', {'protein':0,'carbs':0,'fat':0})
                raw_ings = [it.strip() for it in str(r.get('ingredients','')).split(',') if it.strip()]
                dedup_ings = []
                for it in raw_ings:
                    if it not in dedup_ings:
                        dedup_ings.append(it)
                total_ings = len(dedup_ings)
                mc = r.get('matched_count', 0)
                missc = r.get('missing_count', total_ings)

                instr_text = (r.get('instructions', '') or r.get('method', '') or r.get('preparation', ''))
                if instr_text:
                    instr_lines = [ln.strip() for ln in instr_text.splitlines() if ln.strip()]
                    filtered_lines = []
                    for ln in instr_lines:
                        ln_low = ln.lower()
                        skip = False
                        for it in dedup_ings:
                            if it.lower() in ln_low or ln_low in it.lower():
                                skip = True
                                break
                        if not skip:
                            filtered_lines.append(ln)
                    instr_text = '\n'.join(filtered_lines)
                instr_html = (instr_text.replace('\n','<br>')) if instr_text else ''

                ingredients_html = '<ul>' + ''.join([f'<li>{it}</li>' for it in dedup_ings]) + '</ul>' if dedup_ings else ''

                card_html = f"""
                    <div style='border:1px solid #ddd; padding:10px; border-radius:6px; margin-bottom:10px; background:#fff; color:#111;'>
                      <div style='font-size:18px; font-weight:700; margin-bottom:4px; color:#000;'>{title}</div>
                      <div style='font-size:12px; color:#444; margin-bottom:6px;'>B/W/T (na 100g): {macros.get('protein',0)}g / {macros.get('carbs',0)}g / {macros.get('fat',0)}g</div>
                      <div style='font-weight:600; margin-bottom:4px;'>Składniki ({mc} / {total_ings}):</div>
                      {ingredients_html}
                      <details style='margin-top:6px;'><summary>Sposób przygotowania</summary><div style='margin-top:6px'>{instr_html}</div></details>
                    </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

st.header("Czat z kuchennym asystentem")
if 'messages' not in st.session_state:
    st.session_state.messages = []

chat_col1, chat_col2 = st.columns([3,1])
with chat_col1:
    for i, m in enumerate(st.session_state.messages):
        if m['role']=='user':
            st.markdown(f"**Ty:** {m['text']}")
        else:
            st.markdown(f"**Asystent:** {m['text']}")

with chat_col2:
    st.write("")

with st.form(key='chat_form'):
    user_input = st.text_input("Zadaj pytanie o gotowanie lub swoje składniki:")
    submit = st.form_submit_button("Wyślij")
    if submit and user_input.strip():
        st.session_state.messages.append({'role':'user','text':user_input})
        with st.spinner("Generowanie odpowiedzi..."):
            response = assistant.chat(user_input, context_ingredients=ingredients)
        st.session_state.messages.append({'role':'assistant','text':response})
