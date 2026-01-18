import json
import os
import sqlite3
import streamlit as st

try:
    import importer
    HAS_IMPORTER = True
except ImportError:
    HAS_IMPORTER = False


DB_PATH_DEFAULT = "data/library.sqlite"


st.set_page_config(page_title="Random Dish Picker", page_icon="🍽️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');
    :root {
        --card: #ffffff;
        --ink: #1f2328;
        --accent: #2f6f5e;
        --muted: #5c646b;
        --line: #e4e7eb;
    }
    html, body, [class*="stApp"] {
        background: linear-gradient(180deg, #f3f4f2 0%, #f9faf8 100%);
        color: var(--ink);
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        overflow-x: hidden;
    }
    /* Reduce top padding */
    .stMainBlockContainer, [data-testid="stAppViewBlockContainer"] {
        padding-top: 0.5rem !important;
    }
    .block-container {
        padding-top: 0.5rem !important;
    }
    [data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
    }
    /* Hide Streamlit toolbar and menu */
    [data-testid="stToolbar"],
    .stDeployButton,
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    /* Align all controls on same row */
    .st-key-desktop_controls [data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }
    .st-key-desktop_controls [data-testid="column"] {
        display: flex !important;
        align-items: center !important;
    }
    .st-key-desktop_controls [data-testid="column"] > div {
        width: 100%;
    }
    /* Center the nav counter vertically */
    .nav-counter-desktop, .nav-counter-mobile {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 38px;
        color: #5c646b;
        font-size: 0.9rem;
    }
    /* Desktop multiselect - single line, no wrap */
    .st-key-desktop_controls .stMultiSelect [data-baseweb="select"] > div:first-child {
        flex-wrap: nowrap !important;
        overflow: hidden !important;
        max-height: 44px !important;
    }
    /* Disabled button styling */
    .stButton button:disabled {
        background: #cccccc !important;
        color: #888888 !important;
        cursor: not-allowed !important;
        opacity: 1 !important;
    }
    /* Responsive layouts - target containers by key */
    .st-key-desktop_controls {
        display: block !important;
    }
    .st-key-desktop_controls,
    .st-key-desktop_controls > div,
    .st-key-desktop_controls > div > div,
    .st-key-desktop_controls [data-testid="stVerticalBlock"],
    .st-key-desktop_controls [data-testid="stHorizontalBlock"],
    .st-key-desktop_controls [data-testid="stElementContainer"] {
        width: 100% !important;
        max-width: none !important;
        flex: 1 1 100% !important;
    }
    .st-key-mobile_controls {
        display: none !important;
    }
    @media (max-width: 768px) {
        .st-key-desktop_controls {
            display: none !important;
        }
        .st-key-mobile_controls {
            display: block !important;
        }
        /* Keep mobile controls horizontal */
        .st-key-mobile_controls [data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            gap: 8px !important;
            padding-right: 228px;
        }
        /* Nav buttons sizing */
        .st-key-mobile_controls [data-testid="stHorizontalBlock"] > div:nth-child(2),
        .st-key-mobile_controls [data-testid="stHorizontalBlock"] > div:nth-child(4) {
            flex: 0 0 40px !important;
            min-width: 40px !important;
            max-width: 40px !important;
        }
        /* Counter column */
        .st-key-mobile_controls [data-testid="stHorizontalBlock"] > div:nth-child(3) {
            flex: 0 0 50px !important;
            min-width: 50px !important;
        }
        /* Fix popover button text visibility */
        .st-key-mobile_controls [data-testid="stPopover"] button,
        .st-key-mobile_controls [data-testid="stPopover"] button span,
        .st-key-mobile_controls [data-testid="stPopover"] button p {
            background: #2f6f5e !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
        h2 {
            font-size: 1.5rem !important;
        }
    }
    h1, h2, h3, h4 {
        font-family: 'Libre Baskerville', serif;
        letter-spacing: 0.2px;
    }
    .badge {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 6px 4px 0;
        border-radius: 999px;
        font-size: 0.85rem;
        background: #eef3f1;
        color: #2f3b36;
        border: 1px solid #dde6e1;
    }
    .card {
        background: var(--card);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border: 1px solid var(--line);
    }
    .stButton > button {
        background: var(--accent);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
    }
    .stButton > button:hover {
        background: #275b4d;
        color: #ffffff;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--accent);
        color: #ffffff;
        border-radius: 999px;
    }
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: #ffffff;
    }
    .stImage > div {
        position: relative !important;
    }
    .stImage img {
        pointer-events: none !important;
    }
    .fade-in {
        animation: fadeIn 280ms ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .section-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
        margin-bottom: 10px;
    }
    /* Reduce spacing between sections */
    .stMarkdown, .stCaption {
        margin-bottom: 0 !important;
    }
    h3 {
        margin-bottom: 4px !important;
    }
    /* Reduce default Streamlit block gaps */
    .stElementContainer, .stVerticalBlock {
        gap: 0.5rem !important;
    }
    /* Vertically align the select and button */
    [data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }
    /* Center only the library row */
    .stMainBlockContainer > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]:first-of-type {
        align-items: center !important;
    }
    /* Make streamlit header and controls more visible */
    # .stApp > header {
    #     background: rgba(255, 255, 255, 0.9);
    #     backdrop-filter: blur(10px);
    #     border-bottom: 1px solid var(--line);
    # }
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--line);
    }
    .stApp > header[data-testid="stHeader"] {
        height: auto;
        min-height: 60px;
    }
    /* Make streamlit toolbar more visible */
    # [data-testid="stToolbar"] {
    #     background: rgba(255, 255, 255, 0.95);
    #     backdrop-filter: blur(10px);
    #     border: 1px solid var(--line);
    #     border-radius: 8px;
    #     margin: 8px;
    # }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_image_bytes(book_path: str, image_path: str) -> bytes:
    if not HAS_IMPORTER:
        raise ImportError("importer module not available")
    reader = importer.EpubReader(book_path)
    try:
        return reader.open_bytes(image_path)
    finally:
        reader.close()


def _format_ingredients(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            lines.append(f"**{line.strip('[]')}**")
        else:
            lines.append(f"- {line}")
    return "\n".join(lines)


def _format_steps(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    return "\n".join(lines)


st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown("## The Random Roasting Tin")
st.markdown('</div>', unsafe_allow_html=True)

db_path = DB_PATH_DEFAULT

# Load book options
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT title FROM books ORDER BY title")
        book_options = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
else:
    book_options = []

# Initialize session state
if "selected_books" not in st.session_state:
    st.session_state.selected_books = book_options.copy()
if "recipe_history" not in st.session_state:
    st.session_state.recipe_history = []
if "history_index" not in st.session_state:
    st.session_state.history_index = -1

# Desktop layout - Book select | [← x/x →] | Random dish
can_go_prev = st.session_state.history_index > 0
can_go_next = st.session_state.history_index < len(st.session_state.recipe_history) - 1
counter_text = f"{st.session_state.history_index + 1} / {len(st.session_state.recipe_history)}" if st.session_state.recipe_history else "0 / 0"

# Desktop controls
with st.container(key="desktop_controls"):
    desk_left, desk_right = st.columns([4, 1.17])
    with desk_left:
        sel_col, nav1, nav2, nav3 = st.columns([4, 0.5, 0.6, 0.5])
        with sel_col:
            desktop_selected = st.multiselect(
                "Books",
                options=book_options,
                default=st.session_state.selected_books,
                placeholder="Choose one or more",
                label_visibility="collapsed",
                key="desktop_books"
            )
        with nav1:
            desktop_prev = st.button("←", key="desktop_prev", disabled=not can_go_prev)
        with nav2:
            st.markdown(f'<div class="nav-counter-desktop">{counter_text}</div>', unsafe_allow_html=True)
        with nav3:
            desktop_next = st.button("→", key="desktop_next", disabled=not can_go_next)
    with desk_right:
        desktop_submitted = st.button("Random dish", key="desktop_btn")

# Mobile layout - Books | ← | x/x | → | Random dish
with st.container(key="mobile_controls"):
    mob_col1, mob_col2, mob_col3, mob_col4, mob_col5 = st.columns([1.2, 0.3, 0.5, 0.3, 1.2])
    with mob_col1:
        with st.popover(f"Books ({len(st.session_state.selected_books)})"):
            mobile_selected = []
            for book in book_options:
                checked = st.checkbox(
                    book,
                    value=book in st.session_state.selected_books,
                    key=f"mobile_cb_{book}"
                )
                if checked:
                    mobile_selected.append(book)
    with mob_col2:
        can_go_prev_m = st.session_state.history_index > 0
        mobile_prev = st.button("←", key="mobile_prev", disabled=not can_go_prev_m)
    with mob_col3:
        st.markdown(f'<div class="nav-counter-mobile">{counter_text}</div>', unsafe_allow_html=True)
    with mob_col4:
        can_go_next_m = st.session_state.history_index < len(st.session_state.recipe_history) - 1
        mobile_next = st.button("→", key="mobile_next", disabled=not can_go_next_m)
    with mob_col5:
        mobile_submitted = st.button("Random dish", key="mobile_btn")

# Sync selections
if desktop_selected != st.session_state.selected_books:
    st.session_state.selected_books = desktop_selected
if mobile_selected != st.session_state.selected_books:
    st.session_state.selected_books = mobile_selected

selected_books = st.session_state.selected_books
submitted = desktop_submitted or mobile_submitted

# Handle navigation
if (desktop_prev or mobile_prev) and st.session_state.history_index > 0:
    st.session_state.history_index -= 1
    st.rerun()
if (desktop_next or mobile_next) and st.session_state.history_index < len(st.session_state.recipe_history) - 1:
    st.session_state.history_index += 1
    st.rerun()

# Fetch new random recipe
if submitted:
    if not os.path.exists(db_path):
        st.error("Database not found. Run importer first.")
    else:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            if selected_books:
                placeholders = ",".join(["?"] * len(selected_books))
                cur.execute(
                    f"""
                    SELECT recipes.title, books.title, recipes.href, recipes.image_paths,
                           recipes.ingredients, recipes.steps, recipes.meta, books.source_path
                    FROM recipes
                    JOIN books ON books.id = recipes.book_id
                    WHERE books.title IN ({placeholders})
                    ORDER BY RANDOM()
                    LIMIT 1
                    """,
                    selected_books,
                )
            else:
                cur.execute(
                    """
                    SELECT recipes.title, books.title, recipes.href, recipes.image_paths,
                           recipes.ingredients, recipes.steps, recipes.meta, books.source_path
                    FROM recipes
                    JOIN books ON books.id = recipes.book_id
                    ORDER BY RANDOM()
                    LIMIT 1
                    """
                )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            st.warning("No recipes found in the selected books.")
        else:
            # Add to history and update index
            st.session_state.recipe_history.append(row)
            st.session_state.history_index = len(st.session_state.recipe_history) - 1
            st.rerun()

# Display current recipe from history
if st.session_state.recipe_history and st.session_state.history_index >= 0:
    row = st.session_state.recipe_history[st.session_state.history_index]
    recipe_title, book_title, href, image_paths, ingredients, steps, meta, source_path = row

    # Display recipe
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"### {recipe_title}")
        st.caption(f"Book: {book_title}")
        if meta:
            badges = "".join(f"<span class='badge'>{line}</span>" for line in meta.splitlines() if line.strip())
            st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: -8px; margin-bottom: 16px;">{badges}</div>', unsafe_allow_html=True)

        if ingredients:
            st.markdown('<div class="section-title">Ingredients</div>', unsafe_allow_html=True)
            st.markdown(_format_ingredients(ingredients))

        if steps:
            st.markdown('<div class="section-title">Method</div>', unsafe_allow_html=True)
            st.markdown(_format_steps(steps))

    with col2:
        if image_paths:
            try:
                images = json.loads(image_paths)
            except Exception:
                images = []
            if images:
                for image_path in images:
                    if not image_path or image_path.startswith("http"):
                        continue
                    try:
                        import base64
                        # Check if it's an extracted file or still in EPUB
                        if os.path.isfile(image_path):
                            with open(image_path, "rb") as f:
                                data = f.read()
                        else:
                            data = _load_image_bytes(source_path, image_path)
                        encoded = base64.b64encode(data).decode()
                        st.markdown(
                            f'<img src="data:image/jpeg;base64,{encoded}" style="width: 100%; border-radius: 8px; margin-top: 18px;">',
                            unsafe_allow_html=True
                        )
                        break
                    except Exception:
                        continue
    st.markdown("</div>", unsafe_allow_html=True)
