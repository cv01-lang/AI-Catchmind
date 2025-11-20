import io
import time
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from google import genai
from google.genai import types


# ---------------- 기본 설정 ---------------- #
st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* 전체 폰트 조금 키우기 (태블릿 고려) */
    html, body, [class*="css"]  {
        font-size: 18px;
    }
    .stButton>button {
        font-size: 20px;
        padding: 0.6em 1.2em;
        border-radius: 0.8em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CATEGORIES = ["동물", "과일", "음식", "물건", "탈것"]
TOTAL_ROUNDS = 5
TIME_LIMIT_SECONDS = 60


# ---------------- 유틸 함수 ---------------- #
@st.cache_data
def load_keywords():
    try:
        df = pd.read_csv("keyword.csv")
    except FileNotFoundError:
        st.error("⚠️ `Keyword.csv` 파일을 찾을 수 없습니다. 같은 폴더에 파일을 넣어주세요.")
        st.stop()

    expected_cols = {"카테고리", "키워드"}
    if not expected_cols.issubset(set(df.columns)):
        st.error("⚠️ `Keyword.csv` 파일의 컬럼은 반드시 `카테고리`, `키워드` 여야 합니다.")
        st.stop()

    return df


def get_client():
    """Streamlit secrets에서 API Key를 읽어 Gemini 클라이언트 생성."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "⚠️ `GEMINI_API_KEY`가 설정되어 있지 않습니다.\n\n"
            "`.streamlit/secrets.toml` 파일에 다음처럼 설정해주세요.\n\n"
            "[[secrets]]\nGEMINI_API_KEY = \"YOUR_API_KEY\""
        )
        st.stop()
    return genai.Client(api_key=api_key)


def image_array_to_png_bytes(image_array):
    """canvas의 image_data(numpy array)를 PNG 바이트로 변환."""
    if image_array is None:
        return None
    img = Image.fromarray(image_array.astype("uint8"), "RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def call_gemini(category: str, image_bytes: bytes) -> str:
    """Gemini-2.5-flash를 호출해 그림에 대한 한 단어 추론을 수행."""
    client = get_client()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    system_instruction = (
        "너는 초등학생이 그린 단순한 그림을 보고 정답을 맞추는 게임의 AI야. "
        "항상 한국어로 답하고, 반드시 '한 단어'로만 대답해. "
        "색깔이나 수식어는 쓰지 말고, 대상의 이름만 명사 한 단어로 말해. "
        "예: 사과, 원숭이, 연필 등."
    )

    user_prompt = (
        f"카테고리: {category}\n"
        "주어진 그림을 보고 이 카테고리에 속하는 대상이 무엇인지 추론해.\n"
        "형태와 윤곽에 집중해서 가장 가능성이 높은 대상 한 가지를 선택해.\n"
        "정답은 한국어 명사 한 단어만 출력해. 예: '사과', '원숭이', '연필'\n"
        "문장, 설명, 두 단어 이상(예: '빨간 사과')은 절대 쓰지 마."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[user_prompt, img],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            max_output_tokens=16,
        ),
    )

    text = (response.text or "").strip()

    # 후처리: 첫 줄, 첫 토큰만 남겨 강제로 '한 단어'로 맞추기
    if "\n" in text:
        text = text.split("\n")[0].strip()

    # 공백/쉼표 기준 첫 토큰만
    token = re.split(r"[,\s]+", text)[0].strip()

    # 혹시 비어 있으면 원문 반환
    return token or text or "모름"


def reset_game():
    for key in [
        "page",
        "selected_category",
        "keywords",
        "round_index",
        "results",
        "round_start_time",
        "current_snapshot",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.page = "start"


def start_game(selected_category: str):
    df = load_keywords()
    cat_df = df[df["카테고리"] == selected_category]

    if cat_df.empty:
        st.error(f"⚠️ `{selected_category}` 카테고리의 키워드가 없습니다. Keyword.csv를 확인해주세요.")
        st.stop()

    # 5개를 뽑되, 키워드가 부족하면 중복 허용
    if len(cat_df) >= TOTAL_ROUNDS:
        sampled = cat_df.sample(TOTAL_ROUNDS, replace=False, random_state=int(time.time()))
    else:
        sampled = cat_df.sample(TOTAL_ROUNDS, replace=True, random_state=int(time.time()))

    st.session_state.selected_category = selected_category
    st.session_state.keywords = sampled["키워드"].tolist()
    st.session_state.round_index = 0
    st.session_state.results = []
    st.session_state.round_start_time = time.time()
    st.session_state.current_snapshot = None
    st.session_state.page = "game"


# ---------------- 화면 구성 함수 ---------------- #
def draw_start_page():
    st.title("🎨 AI 캐치마인드")
    st.write("초등학생용 그림 퀴즈 게임입니다. 제시어를 보고 그림을 그리면 AI가 정답을 맞춰봐요!")

    st.markdown("### 1. 카테고리를 선택하세요")
    category = st.radio(
        "카테고리",
        CATEGORIES,
        horizontal=True,
        index=0,
    )

    st.markdown("### 2. 게임 설명")
    st.markdown(
        """
        - 선택한 카테고리에서 **랜덤으로 5개의 제시어**가 나와요.  
        - 제한시간 **60초 동안 그림판에 그림을 그려보세요.**  
        - 시간이 지나면 그림판은 잠기고, 그려진 그림을 가지고 **AI가 정답을 한 단어로 추론**해요.  
        - 모든 문제(5문제)를 풀면 **결과 화면**에서 라운드별로 정답과 AI의 답을 확인할 수 있어요.
        """
    )

    if st.button("게임 시작하기 ▶", use_container_width=True):
        start_game(category)


def draw_game_page():
    # 모든 라운드를 다 풀었으면 결과 페이지로 이동
    if st.session_state.round_index >= TOTAL_ROUNDS:
        st.session_state.page = "result"
        st.experimental_rerun()

    round_idx = st.session_state.round_index
    keyword = st.session_state.keywords[round_idx]
    category = st.session_state.selected_category

    # 남은 시간 계산
    elapsed = time.time() - st.session_state.round_start_time
    remaining = max(0, TIME_LIMIT_SECONDS - int(elapsed))
    drawing_disabled = remaining <= 0

    st.header("🖌️ 그림 그리기 (게임 화면)")
    col_title, col_timer = st.columns([3, 1])

    with col_title:
        st.subheader(f"라운드 {round_idx + 1} / {TOTAL_ROUNDS}")
        st.markdown(f"**카테고리:** {category}")
        st.markdown(f"**제시어:** `{keyword}`")

    with col_timer:
        st.metric("남은 시간(초)", remaining)
        if drawing_disabled:
            st.error("⏰ 시간 종료! 이제 그림을 더 그릴 수 없어요.")

    st.markdown("---")

    # 좌측: 캔버스 / 우측: 스냅샷 & 안내
    col_canvas, col_side = st.columns([2, 1])

    with col_canvas:
        st.markdown("#### 1️⃣ 그림판에 제시어를 그려보세요")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=8,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"canvas_round_{round_idx}",
            disabled=drawing_disabled,
        )

        # 현재 그림을 스냅샷으로 저장
        if canvas_result.image_data is not None:
            png_bytes = image_array_to_png_bytes(canvas_result.image_data)
            if png_bytes:
                st.session_state.current_snapshot = png_bytes

    with col_side:
        st.markdown("#### 2️⃣ 제출하면 AI가 맞춰봐요")

        if st.session_state.get("current_snapshot"):
            st.image(
                st.session_state.current_snapshot,
                caption="현재 스냅샷 (제출 시 이 그림이 사용됩니다)",
                use_column_width=True,
            )
        else:
            st.info("아직 스냅샷이 없습니다. 그림을 그리면 여기에서 미리보기를 볼 수 있어요.")

        if drawing_disabled:
            st.info("⏰ 시간이 끝났어요! **제출하기** 버튼을 눌러 AI에게 정답을 물어보세요.")

        st.markdown("")

        submit = st.button("제출하기 (AI에게 맞춰보기) 🚀", use_container_width=True)

        if submit:
            if not st.session_state.get("current_snapshot"):
                st.warning("먼저 그림을 그려주세요!")
                st.stop()

            snapshot_bytes = st.session_state.current_snapshot

            with st.spinner("🤖 AI가 생각중입니다..."):
                st.image(
                    snapshot_bytes,
                    caption="내가 그린 그림 (제출 스냅샷)",
                    use_column_width=True,
                )
                ai_answer = call_gemini(category, snapshot_bytes)

            # 라운드 결과 저장
            st.session_state.results.append(
                {
                    "round": round_idx + 1,
                    "keyword": keyword,
                    "ai_answer": ai_answer,
                    "image": snapshot_bytes,
                }
            )

            # 다음 라운드로
            st.session_state.round_index += 1
            if st.session_state.round_index >= TOTAL_ROUNDS:
                st.session_state.page = "result"
            else:
                st.session_state.round_start_time = time.time()
                st.session_state.current_snapshot = None

            st.experimental_rerun()


def draw_result_page():
    st.header("📊 게임 결과")

    results = st.session_state.get("results", [])

    if not results:
        st.info("아직 결과가 없습니다. 먼저 게임을 진행해주세요.")
        if st.button("처음으로 돌아가기"):
            reset_game()
        return

    # 간단하게 정답/오답 카운트
    correct_count = sum(1 for r in results if r["ai_answer"] == r["keyword"])
    st.subheader(f"총 {TOTAL_ROUNDS}문제 중 {correct_count}개 정답 (단순 일치 기준)")

    st.markdown("---")

    for r in results:
        st.markdown(f"### 라운드 {r['round']}")
        col_img, col_text = st.columns([2, 2])

        with col_img:
            st.image(
                r["image"],
                caption="학생이 그린 그림",
                use_column_width=True,
            )

        with col_text:
            st.markdown(f"**제시어(정답)**: `{r['keyword']}`")
            st.markdown(f"**AI의 답변**: `{r['ai_answer']}`")
            if r["ai_answer"] == r["keyword"]:
                st.success("✅ 일치!")
            else:
                st.warning("❌ 다르게 예측했어요.")

        st.markdown("---")

    if st.button("다시 하기 🔁", use_container_width=True):
        reset_game()


# ---------------- 메인 로직 ---------------- #
if "page" not in st.session_state:
    st.session_state.page = "start"

page = st.session_state.page

if page == "start":
    draw_start_page()
elif page == "game":
    draw_game_page()
elif page == "result":
    draw_result_page()
else:
    reset_game()
