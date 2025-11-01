"""
    streamlit run app.py
"""

import streamlit as st
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

from rag.data.loader import load_products
from rag.indexing import Embedder, DenseRetriever
from rag.retrieval import fuse_results_rrf, generate_rewrites
from rag.generation import generate_answer
from rag.utils import create_llm_client
from config import settings

st.set_page_config(
    page_title="RAG поиск",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Загрузка моделей и индексов"""
    with st.spinner("Загрузка моделей..."):
        df = load_products(rag_version=True)
        embedder = Embedder(settings.EMBEDDING_MODEL)
        dense = DenseRetriever(
            settings.FAISS_INDEX_PATH,
            settings.METADATA_PATH,
            embedder,
            df
        )

        # проверка LLM
        try:
            llm = create_llm_client()
            llm.chat([{"role": "user", "content": "test"}], max_tokens=1)
            llm_available = True
        except:
            llm = None
            llm_available = False

        return df, dense, llm, llm_available


def render_product_card(product, rank, show_score=True):
    title = product.get('title', 'Без названия')
    url = product.get('url', '#')
    score = product.get('score', 0)

    st.markdown(f"""
    <div class="product-card">
        <div class="product-title">{rank}. {title}</div>
        <a href="{url}" target="_blank" style="text-decoration: none; color: #667eea;">
            Посмотреть товар
        </a>
        {f'<div class="product-score">Релевантность: {score:.3f}</div>' if show_score else ''}
    </div>
    """, unsafe_allow_html=True)


def render_answer_with_sources(answer):
    """Отрисовка ответа с источниками"""
    answer_text = answer.get("answer_md", "")
    sources = answer.get("chosen", [])

    # Ответ
    st.markdown(f"""
    <div class="answer-box">
        <h3 style="margin-top: 0; color: #333;">💡 Рекомендация</h3>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #555;">{answer_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # Источники
    if sources:
        st.markdown("### Источники")
        for i, source in enumerate(sources, 1):
            render_product_card(source, i, show_score=False)


def main():
    st.markdown('<h1 class="main-header">RAG поиск </h1>', unsafe_allow_html=True)
    df, dense, llm, llm_available = load_models()
    query = st.text_input(
        "",
        placeholder="Например: найди белые кроссовки найк или подбери черную куртку...",
        key="search_query",
        label_visibility="collapsed"
    )
    with st.sidebar:
        st.header("Настройки")

        search_mode = st.radio(
            "Режим поиска",
            ["С AI генерацией" if llm_available else "Только поиск", "Только поиск"],
            index=0
        )

        top_k = st.slider("Количество результатов", 5, 20, 10)

        if llm_available and search_mode == "С AI генерацией":
            show_rewrites = st.checkbox("Показать перефразы", value=False)
        else:
            show_rewrites = False

    if st.button("Найти", use_container_width=True) or query:
        if not query:
            st.warning("Введите поисковый запрос")
            return

        with st.spinner("Ищем..."):
            try:
                if llm_available and search_mode == "С AI генерацией":
                    # С LLM: rewrites + генерация
                    rewrites = generate_rewrites(query, llm)

                    if show_rewrites:
                        st.info(f"Перефразы: {', '.join(rewrites)}")

                    results = fuse_results_rrf(
                        rewrites,
                        dense.search,
                        per_query_k=settings.HITS_PER_QUERY,
                        final_k=settings.FINAL_TOP_K
                    )

                    answer = generate_answer(query, results, df, llm, rewrites=rewrites)
                    render_answer_with_sources(answer)

                else:
                    # Без LLM: только поиск
                    results = dense.search(query, k=top_k)

                    st.markdown(f"### Найдено топ-{top_k}")
                    for i, result in enumerate(results, 1):
                        render_product_card(result, i)

            except Exception as e:
                st.error(f"Ошибка: {e}")
                if st.checkbox("Показать детали ошибки"):
                    st.exception(e)


if __name__ == "__main__":
    main()
