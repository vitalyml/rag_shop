"""
    streamlit run app.py
"""

import os

# Fix для macOS: отключаем многопоточность ДО импорта библиотек
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st

from rag.data.loader import load_products, get_available_shops
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
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(shop_name: str):
    """Загрузка моделей и индексов для конкретного магазина"""
    with st.spinner(f"Загрузка моделей для '{shop_name}'..."):
        df = load_products(rag_version=True, shop_name=shop_name)
        embedder = Embedder(settings.EMBEDDING_MODEL)

        shop_paths = settings.get_shop_paths(shop_name)
        faiss_index_path = shop_paths['artifacts'] / "faiss_index.index"

        dense = DenseRetriever(
            faiss_index_path,
            None,  # metadata больше не используется
            embedder,
            df
        )

        # проверка LLM
        try:
            llm_chat = create_llm_client(use_reasoner=False)  # Для перефразов
            llm_reasoner = create_llm_client(use_reasoner=True)  # Для генерации ответа
            # Проверяем доступность
            test_response = llm_chat.chat([{"role": "user", "content": "hi"}], max_tokens=5)
            llm_available = True
        except Exception as e:
            llm_chat = None
            llm_reasoner = None
            llm_available = False
            st.sidebar.warning(f"LLM недоступен: {str(e)[:50]}")

        return df, dense, llm_chat, llm_reasoner, llm_available


def render_product_card(product, rank, show_score=True):
    title = product.get('title', 'Без названия')
    url = product.get('url', '#')
    score = product.get('score', 0)
    image_url = product.get('image_url', '')
    price = product.get('price', '')
    old_price = product.get('old_price', '')
    description = product.get('description', '')

    # Создаём колонки для изображения и текста
    col1, col2 = st.columns([1, 2])

    with col1:
        if image_url and str(image_url) != 'nan':
            try:
                st.image(image_url, use_container_width=True)
            except:
                st.write("🖼️ Изображение недоступно")
        else:
            st.write("🖼️ Нет изображения")

    with col2:
        st.markdown(f"**{rank}. {title}**")

        if price and str(price) != 'nan':
            price_text = f"{price} ₽"
            if old_price and str(old_price) != 'nan':
                price_text = f"~~{old_price} ₽~~ **{price} ₽**"
            st.markdown(price_text)

        if description and str(description) != 'nan':
            # Ограничиваем описание до 150 символов
            short_desc = description[:150] + "..." if len(description) > 150 else description
            st.caption(short_desc)

        st.markdown(f"[Посмотреть товар →]({url})")

        if show_score:
            st.caption(f"Релевантность: {score:.3f}")

    st.divider()


def render_answer_with_sources(answer):
    """Отрисовка ответа с источниками"""
    answer_text = answer.get("answer_md", "")
    sources = answer.get("chosen", [])

    # Ответ
    st.markdown(f"""
    <div class="answer-box">
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

    # Получаем список доступных магазинов
    available_shops = get_available_shops()

    if not available_shops:
        st.error("Не найдено ни одного магазина с данными в папке data/")
        st.info("Создайте папку data/{shop_name}/processed с файлом products_rag.csv")
        return

    with st.sidebar:
        st.header("Настройки")

        # Выбор магазина
        selected_shop = st.selectbox(
            "Магазин",
            options=available_shops,
            index=available_shops.index(settings.DEFAULT_SHOP) if settings.DEFAULT_SHOP in available_shops else 0
        )

    # Загружаем модели для выбранного магазина
    try:
        df, dense, llm_chat, llm_reasoner, llm_available = load_models(selected_shop)
    except FileNotFoundError as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return

    query = st.text_input(
        "Поисковый запрос",
        placeholder="Например: найди белую рубашку",
        key="search_query",
        label_visibility="collapsed"
    )

    with st.sidebar:
        search_mode = st.radio(
            "Режим поиска",
            ["С AI генерацией" if llm_available else "Только поиск", "Только поиск"],
            index=0
        )

        # Константа количества результатов
        top_k = 25

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
                    # 1. Генерируем перефразы с помощью chat модели
                    rewrites = generate_rewrites(query, llm_chat)
                    print(f"DEBUG: Original query: {query}")
                    print(f"DEBUG: Rewrites returned: {rewrites}")

                    if show_rewrites:
                        st.info(f"Перефразы: {', '.join(rewrites)}")

                    # 2. Ищем по всем вариантам запроса (оригинал + перефразы)
                    results = fuse_results_rrf(
                        rewrites,
                        dense.search,
                        per_query_k=settings.HITS_PER_QUERY,
                        final_k=settings.FINAL_TOP_K
                    )

                    # 3. Генерируем ответ с помощью reasoner модели, передаем ТОЛЬКО оригинальный запрос
                    answer = generate_answer(query, results, df, llm_reasoner)

                    # DEBUG: Проверяем что в answer
                    print(f"DEBUG: Answer keys: {answer.keys()}")
                    print(f"DEBUG: Chosen count: {len(answer.get('chosen', []))}")
                    if answer.get('chosen'):
                        first = answer['chosen'][0]
                        print(f"DEBUG: First chosen keys: {first.keys()}")
                        print(f"DEBUG: First image_url: {first.get('image_url', 'NO IMAGE')}")

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
