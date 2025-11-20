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
    """
    keyword.csv 파일을 불러오는 함수
    - 컬럼: 카테고리, 키워드
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["카테고리", "키워드"])
    return df


def init_session_state():
    """게임에 필요한 세션 상태 초기화"""
    defaults = {
        "page": "start",          # start, game, result
        "category": None,
        "problems": [],           # [{"keyword": str}, ...]
        "round_index": 0,
        "user_images": [],        # [bytes, ...]
        "ai_answers": [],         # [str, ...]
        "correct_answers": [],    # [str, ...]
        "start_time": None,
        "last_snapshot_bytes": None,
        "submitting": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_game():
    """전체 게임 리셋"""
    for key in [
        "page", "category", "problems", "round_index",
        "user_images", "ai_answers", "correct_answers",
        "start_time", "last_snapshot_bytes", "submitting"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


def prepare_problems(category: str, n_rounds: int = 5):
    """선택한 카테고리에서 n_rounds개의 제시어 생성"""
    df = load_keywords()
    df_cat = df[df["카테고리"] == category]

    if df_cat.empty:
        st.error(f"'{category}' 카테고리에 해당하는 키워드가 keyword.csv에 없습니다.")
        return

    # 키워드가 5개 미만이면 중복 허용하여 샘플링
    replace = len(df_cat) < n_rounds
    sampled = df_cat.sample(n=n_rounds, replace=replace, random_state=random.randint(0, 99999))

    st.session_state.problems = [{"keyword": row["키워드"]} for _, row in sampled.iterrows()]
    st.session_state.correct_answers = [p["keyword"] for p in st.session_state.problems]
    st.session_state.round_index = 0
    st.session_state.user_images = []
    st.session_state.ai_answers = []
    st.session_state.start_time = time.time()
    st.session_state.last_snapshot_bytes = None
    st.session_state.page = "game"


def pil_from_canvas(image_data: np.ndarray) -> Image.Image:
    """캔버스의 RGBA numpy 배열을 흰 배경의 RGB PIL 이미지로 변환"""
    img = Image.fromarray(image_data.astype("uint8")).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img_white = Image.alpha_composite(bg, img).convert("RGB")
    return img_white


def call_gemini(image_bytes: bytes, category: str) -> str:
    """
    Gemini-2.5-flash 호출하여 그림을 분석하고 한 단어로 정답 추론
    - 응답은 카테고리와 관련된 '한국어 한 단어'만 허용
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets에 API 키를 추가해주세요.")
        return "알수없음"

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")

    img = Image.open(io.BytesIO(image_bytes))

    prompt = f"""
너는 초등학생이 그린 그림을 보고 단어를 맞추는 '캐치마인드' 게임용 AI야.

규칙:
- 반드시 카테고리와 관련된 '한국어 한 단어'만 대답해.
- 카테고리: {category}
- 예시: 사과, 원숭이, 연필, 자동차, 비행기, 토마토, 당근 등
- 문장, 설명, 이모지, 기호, 따옴표를 절대 쓰지 마.
- 조사(을, 를, 이, 가 등)를 붙이지 말고 순수한 명사 한 단어만 답해.
- "정답은 ~~입니다" 같은 말은 하지 마.
- 초등학생의 그림이기 때문에 형태와 윤곽에 집중해서 추론해.

출력 형식:
- 한 단어만 출력.
"""

    try:
        response = model.generate_content([prompt, img])
        text = response.text.strip()

        # 혹시 여러 단어가 온 경우 첫 번째 단어만 사용 & 특수문자 제거
        text = text.replace("정답:", "").replace("정답은", "")
        text = text.strip()
        first = text.split()[0]
        first = first.strip(" .,!?:;\"'()[]{}")
        if not first:
            first = "모름"
        return first
    except Exception as e:
        st.error(f"AI 호출 중 오류가 발생했습니다: {e}")
        return "오류"


# ---------- 메인 앱 ----------
init_session_state()

st.markdown('<div class="main-title">🎨 AI 캐치마인드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">태블릿으로 그림을 그리고, AI가 단어를 맞춰보는 게임이에요!</div>', unsafe_allow_html=True)

# ---- 페이지 라우팅 ----
page = st.session_state.page


# ---------- 시작 화면 ----------
def render_start_page():
    st.markdown("### 1단계 · 카테고리를 선택하세요")

    categories = ["동물", "과일", "채소", "사물", "교통수단"]

    cols = st.columns(5)
    selected = None
    for i, cat in enumerate(categories):
        with cols[i]:
            if st.button(cat, use_container_width=True, type="primary" if st.session_state.get("category") == cat else "secondary"):
                st.session_state.category = cat
                selected = cat

    if st.session_state.category:
        st.info(f"선택된 카테고리: **{st.session_state.category}**")

    st.markdown("---")
    st.markdown("### 2단계 · 게임 시작")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("선택한 카테고리의 제시어 5개가 랜덤으로 출제됩니다.")
        st.write("- 제한 시간: **각 문제당 60초**")
        st.write("- AI는 당신의 그림만 보고 한 단어로 정답을 맞춰요!")
    with col2:
        if st.button("🚀 게임 시작하기", use_container_width=True, type="primary"):
            if not st.session_state.category:
                st.warning("먼저 카테고리를 선택해주세요!")
            else:
                prepare_problems(st.session_state.category)


# ---------- 게임 화면 ----------
def render_game_page():
    # 모든 문제를 다 풀었으면 결과 화면으로 이동
    if st.session_state.round_index >= len(st.session_state.problems):
        st.session_state.page = "result"
        st.experimental_rerun()
        return

    round_idx = st.session_state.round_index
    current_keyword = st.session_state.problems[round_idx]["keyword"]
    category = st.session_state.category
    submitting = st.session_state.submitting

    # 타이머 설정
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    elapsed = time.time() - st.session_state.start_time
    remaining = max(0, int(60 - elapsed))
    time_over = elapsed >= 60

    top_col1, top_col2, top_col3 = st.columns([2, 2, 1])

    with top_col1:
        st.markdown(f"#### 문제 {round_idx + 1} / {len(st.session_state.problems)}")
        st.markdown(f'<div class="keyword-box">제시어: <span style="color:#e65100;">{current_keyword}</span></div>', unsafe_allow_html=True)
        st.caption(f"카테고리: {category}")

    with top_col2:
        st.markdown("#### 남은 시간")
        st.markdown(
            f'<div class="timer-box">⏱ {remaining}초 남았어요!</div>',
            unsafe_allow_html=True,
        )
        progress = remaining / 60
        st.progress(progress if progress >= 0 else 0)

    with top_col3:
        if st.button("↩ 처음으로", use_container_width=True):
            reset_game()
            st.experimental_rerun()
            return

    st.markdown("---")

    # 그림판 + 제출 버튼 영역
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### 1) 그림을 그려요")

        # 시간 초과 또는 제출 중에는 캔버스 잠금
        canvas_disabled = time_over or submitting

        # 캔버스 그리기
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=8,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=500,
            height=500,
            drawing_mode="freedraw",
            key=f"canvas_{round_idx}",
            disabled=canvas_disabled,
            update_streamlit=True,
        )

        # 캔버스에서 이미지 데이터가 있을 때마다 스냅샷 저장
        if canvas_result.image_data is not None:
            img_pil = pil_from_canvas(canvas_result.image_data)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            buf.seek(0)
            st.session_state.last_snapshot_bytes = buf.getvalue()

        if time_over:
            st.warning("⏰ 제한 시간이 끝났어요! 그려진 마지막 그림으로 AI가 정답을 맞춰볼게요.")

    with right:
        st.markdown("#### 2) AI에게 제출해요")

        if submitting:
            st.info("🧠 AI가 생각중입니다...")
            # 마지막 스냅샷 이미지 고정 표시
            if st.session_state.last_snapshot_bytes:
                st.image(
                    st.session_state.last_snapshot_bytes,
                    caption="AI가 보는 마지막 그림",
                    use_column_width=True,
                )
            return

        if st.session_state.last_snapshot_bytes is None:
            st.info("그림을 먼저 그리고 나서 제출 버튼을 눌러주세요.")

        submit_disabled = st.session_state.last_snapshot_bytes is None

        if st.button("✅ 제출하기", use_container_width=True, disabled=submit_disabled):
            if st.session_state.last_snapshot_bytes is None:
                st.warning("제출할 그림이 없습니다. 그림을 그려주세요!")
                return

            # 제출 상태로 전환
            st.session_state.submitting = True
            st.experimental_rerun()
            return

        # 제출 버튼 아래 도움말
        st.caption("- 그림을 다 그렸다면 제출 버튼을 눌러보세요.\n- 시간 안에 제출하지 않아도 마지막 그림으로 AI가 맞춰요.")


    # 제출 상태 처리 (별도 rerun에서 처리)
    if st.session_state.submitting:
        # AI 호출
        with st.spinner("AI가 그림을 보고 단어를 떠올리고 있어요..."):
            ai_answer = call_gemini(st.session_state.last_snapshot_bytes, category)

        # 결과 저장
        st.session_state.user_images.append(st.session_state.last_snapshot_bytes)
        st.session_state.ai_answers.append(ai_answer)

        # 다음 라운드로 이동
        st.session_state.round_index += 1
        st.session_state.start_time = time.time()
        st.session_state.last_snapshot_bytes = None
        st.session_state.submitting = False

        # 모든 문제를 풀었다면 결과로 이동
        if st.session_state.round_index >= len(st.session_state.problems):
            st.session_state.page = "result"

        st.experimental_rerun()


# ---------- 결과 화면 ----------
def render_result_page():
    st.success("🎉 모든 문제를 다 풀었어요! 결과를 확인해볼까요?")

    n_rounds = len(st.session_state.correct_answers)

    for i in range(n_rounds):
        st.markdown(f"### 🔎 문제 {i + 1}")

        col1, col2 = st.columns([2, 3])

        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("**사용자가 그린 그림**")
            if i < len(st.session_state.user_images) and st.session_state.user_images[i] is not None:
                st.image(st.session_state.user_images[i], use_column_width=True)
            else:
                st.write("저장된 그림이 없습니다.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            ai_ans = st.session_state.ai_answers[i] if i < len(st.session_state.ai_answers) else "응답 없음"
            correct = st.session_state.correct_answers[i] if i < len(st.session_state.correct_answers) else "정답 없음"

            st.markdown(f"**AI 응답:** `{ai_ans}`")
            st.markdown(f"**정답(제시어):** `{correct}`")

            if ai_ans == correct:
                st.success("✅ AI가 정답을 맞췄어요!")
            else:
                st.info("🤔 AI의 생각과 정답이 조금 달랐네요.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

    st.markdown("### 다시 해볼까요?")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔁 같은 카테고리로 다시 하기", use_container_width=True):
            cat = st.session_state.category
            reset_game()
            st.session_state.category = cat
            prepare_problems(cat)
            st.experimental_rerun()
    with col2:
        if st.button("🏠 처음 화면으로 돌아가기", use_container_width=True):
            reset_game()
            st.experimental_rerun()


# ---------- 페이지 렌더링 ----------
if page == "start":
    render_start_page()
elif page == "game":
    render_game_page()
elif page == "result":
    render_result_page()
else:
    reset_game()
    render_start_page()
