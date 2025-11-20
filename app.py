import time
import io
import random

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas

import google.generativeai as genai


# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.5rem;
        color: #555555;
    }
    .keyword-box {
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        padding: 0.8rem;
        border-radius: 0.8rem;
        background-color: #fff8e1;
        border: 2px solid #ffca28;
        margin-bottom: 0.8rem;
    }
    .timer-box {
        font-size: 1.2rem;
        font-weight: 700;
        padding: 0.5rem 0.8rem;
        border-radius: 0.8rem;
        background-color: #e3f2fd;
        display: inline-block;
    }
    .result-card {
        border-radius: 1rem;
        padding: 1rem;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 유틸 함수 ----------
@st.cache_data
def load_keywords(csv_path: str = "keyword.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["카테고리", "키워드"])
    return df


def init_session_state():
    defaults = {
        "page": "start",
        "category": None,
        "problems": [],
        "round_index": 0,
        "user_images": [],
        "ai_answers": [],
        "correct_answers": [],
        "start_time": None,
        "last_snapshot_bytes": None,
        "submitting": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_game():
    keys = [
        "page", "category", "problems", "round_index",
        "user_images", "ai_answers", "correct_answers",
        "start_time", "last_snapshot_bytes", "submitting"
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    init_session_state()


def prepare_problems(category: str, n_rounds: int = 5):
    df = load_keywords()
    df_cat = df[df["카테고리"] == category]

    if df_cat.empty:
        st.error(f"'{category}' 카테고리에 키워드가 없습니다.")
        return

    replace = len(df_cat) < n_rounds
    sampled = df_cat.sample(n=n_rounds, replace=replace)

    st.session_state.problems = [{"keyword": row["키워드"]} for _, row in sampled.iterrows()]
    st.session_state.correct_answers = [p["keyword"] for p in st.session_state.problems]
    st.session_state.round_index = 0
    st.session_state.user_images = []
    st.session_state.ai_answers = []
    st.session_state.start_time = time.time()
    st.session_state.last_snapshot_bytes = None
    st.session_state.submitting = False
    st.session_state.page = "game"


def pil_from_canvas(image_data: np.ndarray) -> Image.Image:
    img = Image.fromarray(image_data.astype("uint8")).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def call_gemini(image_bytes: bytes, category: str) -> str:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("GEMINI_API_KEY가 없습니다. secrets.toml을 확인하세요.")
        return "모름"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(io.BytesIO(image_bytes))

    prompt = f"""
너는 초등학생이 그린 그림을 보고 단어를 맞추는 AI야.

규칙:
- 반드시 카테고리와 관련된 '한국어 한 단어'만 대답해.
- 카테고리: {category}
- 예: 사과, 연필, 고양이, 토마토, 당근 등
- 문장, 설명, 이모지, 기호, 따옴표 금지.
- 한 단어 명사만 출력.
"""

    try:
        response = model.generate_content([prompt, img])
        text = response.text.strip()
        text = text.replace("정답:", "").replace("정답은", "")
        first = text.split()[0].strip(" .,!?:;\"'()[]{}")
        return first if first else "모름"
    except Exception as e:
        st.error(f"AI 오류: {e}")
        return "오류"


# ---------- 메인 ----------
init_session_state()

st.markdown('<div class="main-title">🎨 AI 캐치마인드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">그림을 그리고, AI가 단어를 맞추는 게임!</div>', unsafe_allow_html=True)

page = st.session_state.page


# ---------- 시작 화면 ----------
def render_start_page():
    st.markdown("### 1) 카테고리를 선택하세요")
    categories = ["동물", "과일", "채소", "사물", "교통수단"]

    cols = st.columns(5)
    for i, cat in enumerate(categories):
        with cols[i]:
            if st.button(cat, use_container_width=True):
                st.session_state.category = cat

    if st.session_state.category:
        st.info(f"선택됨: **{st.session_state.category}**")

    st.markdown("---")

    if st.button("🚀 게임 시작하기", type="primary", use_container_width=True):
        if not st.session_state.category:
            st.warning("카테고리를 선택하세요!")
        else:
            prepare_problems(st.session_state.category)
            st.rerun()


# ---------- 게임 화면 ----------
def render_game_page():
    # 끝났으면 결과 페이지
    if st.session_state.round_index >= len(st.session_state.problems):
        st.session_state.page = "result"
        st.rerun()

    round_idx = st.session_state.round_index
    keyword = st.session_state.problems[round_idx]["keyword"]
    category = st.session_state.category

    # 제출 중이면 AI 처리 페이지
    if st.session_state.submitting:
        st.markdown(f"### 문제 {round_idx+1}")
        if st.session_state.last_snapshot_bytes:
            st.image(st.session_state.last_snapshot_bytes)
        st.info("🧠 AI가 생각중입니다...")

        with st.spinner("AI 분석 중..."):
            ai_answer = call_gemini(st.session_state.last_snapshot_bytes, category)

        st.session_state.user_images.append(st.session_state.last_snapshot_bytes)
        st.session_state.ai_answers.append(ai_answer)

        st.session_state.round_index += 1
        st.session_state.start_time = time.time()
        st.session_state.last_snapshot_bytes = None
        st.session_state.submitting = False

        st.rerun()

    # 일반 게임 화면
    elapsed = time.time() - st.session_state.start_time
    remaining = max(0, int(60 - elapsed))
    time_over = remaining <= 0

    st.markdown(f"### 문제 {round_idx+1} / 5")
    st.markdown(f'<div class="keyword-box">제시어: {keyword}</div>', unsafe_allow_html=True)
    st.markdown(f"⏱ 남은 시간: **{remaining}초**")

    left, right = st.columns([3, 2])

    with left:
        if not time_over:
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=6,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=500,
                height=500,
                drawing_mode="freedraw",
                key=f"canvas_{round_idx}",
            )

            if canvas_result.image_data is not None:
                img_pil = pil_from_canvas(canvas_result.image_data)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.session_state.last_snapshot_bytes = buf.getvalue()

        else:
            st.warning("⏰ 시간 종료! 마지막 그림을 사용합니다.")
            if st.session_state.last_snapshot_bytes:
                st.image(st.session_state.last_snapshot_bytes)

    with right:
        st.markdown("### 제출하기")

        if st.button("✅ 제출", use_container_width=True):
            # 그림이 아예 없다면 빈 캔버스 제공
            if st.session_state.last_snapshot_bytes is None:
                blank = Image.new("RGB", (500, 500), "white")
                buf = io.BytesIO()
                blank.save(buf, format="PNG")
                st.session_state.last_snapshot_bytes = buf.getvalue()

            st.session_state.submitting = True
            st.rerun()


# ---------- 결과 화면 ----------
def render_result_page():
    st.success("🎉 게임 완료! 결과를 확인해요!")

    for i in range(5):
        st.markdown(f"## 문제 {i+1}")

        col1, col2 = st.columns([2, 3])
        with col1:
            st.image(st.session_state.user_images[i], caption="사용자 그림")

        with col2:
            ai = st.session_state.ai_answers[i]
            correct = st.session_state.correct_answers[i]
            st.write(f"**AI 응답:** `{ai}`")
            st.write(f"**정답:** `{correct}`")

            if ai == correct:
                st.success("정답!")
            else:
                st.info("AI가 조금 다르게 생각했어요!")

        st.markdown("---")

    if st.button("↩ 처음 화면으로"):
        reset_game()
        st.rerun()


# ---------- 실행 ----------
if page == "start":
    render_start_page()
elif page == "game":
    render_game_page()
elif page == "result":
    render_result_page()
