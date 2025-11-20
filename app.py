import time
import io
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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
    """keyword.csv 파일을 불러오는 함수 (카테고리, 키워드)"""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["카테고리", "키워드"])
    return df


def init_session_state():
    """게임에 필요한 세션 상태 초기화"""
    defaults = {
        "page": "start",             # start, game, result
        "category": None,
        "problems": [],              # 준비된 전체 문제 (문항수 + 2, 중복 없는 키워드)
        "round_index": 0,            # 현재 problems 인덱스 (패스 포함 진행)
        "user_images": [],           # 실제로 푼 문제에 대한 그림 bytes (최종)
        "ai_answers": [],            # 실제로 푼 문제에 대한 AI 답
        "correct_answers": [],       # 실제로 푼 문제에 대한 정답(키워드)
        "start_time": None,
        "last_snapshot_bytes": None, # 현재 문제에 대해 마지막 스냅샷
        "submitting": False,         # True: AI 호출 중(생각중 화면)
        "grading": False,            # True: 방금 푼 문제에 대한 채점 화면
        "target_questions": 5,       # 사용자가 설정한 문항 수
        "max_passes": 2,             # 패스 최대 횟수
        "passes_used": 0,            # 이미 사용한 패스 수
        "answered_count": 0,         # 실제로 푼(제출한) 문제 수

        # 중간 채점 화면용 임시 저장
        "last_user_image": None,
        "last_ai_answer": None,
        "last_correct_answer": None,
        "last_is_correct": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_game():
    """전체 게임 리셋"""
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    init_session_state()


def prepare_problems(category: str, n_questions: int):
    """
    선택한 카테고리에서 '문항수 + 2' 개의 키워드를 준비
    - 한 게임 동안 같은 키워드는 다시 나오지 않도록 '키워드' 기준으로 중복 제거 후 샘플링
    """
    df = load_keywords()
    df_cat = df[df["카테고리"] == category]

    # 같은 키워드는 한 번만 사용하기 위해 키워드 기준으로 중복 제거
    df_cat_unique = df_cat.drop_duplicates(subset=["키워드"])

    total_needed = n_questions + 2  # 패스 2회 대비
    if len(df_cat_unique) < total_needed:
        st.error(
            f"'{category}' 카테고리에는 최소 {total_needed}개의 서로 다른 키워드가 필요합니다.\n"
            f"현재 keyword.csv에는 {len(df_cat_unique)}개의 고유 키워드만 존재합니다. 키워드를 더 추가해주세요."
        )
        return

    sampled = df_cat_unique.sample(n=total_needed, replace=False, random_state=random.randint(0, 99999))

    st.session_state.problems = [{"keyword": row["키워드"]} for _, row in sampled.iterrows()]
    st.session_state.round_index = 0
    st.session_state.user_images = []
    st.session_state.ai_answers = []
    st.session_state.correct_answers = []
    st.session_state.start_time = time.time()
    st.session_state.last_snapshot_bytes = None
    st.session_state.submitting = False
    st.session_state.grading = False
    st.session_state.max_passes = 2
    st.session_state.passes_used = 0
    st.session_state.answered_count = 0

    st.session_state.last_user_image = None
    st.session_state.last_ai_answer = None
    st.session_state.last_correct_answer = None
    st.session_state.last_is_correct = None

    st.session_state.target_questions = n_questions
    st.session_state.page = "game"


def pil_from_canvas(image_data: np.ndarray) -> Image.Image:
    """캔버스의 RGBA numpy 배열을 흰 배경의 RGB PIL 이미지로 변환"""
    img = Image.fromarray(image_data.astype("uint8")).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def call_gemini(image_bytes: bytes, category: str) -> str:
    """
    Gemini-2.5-flash 호출하여 그림을 분석하고 한 단어로 정답 추론
    - 응답은 카테고리와 관련된 '한국어 한 단어'만 허용
    - 네트워크 문제 시 '통신에 실패했습니다' 반환
    """
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets에 API 키를 추가해주세요.")
        return "통신에 실패했습니다"

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
        if not response or not getattr(response, "text", "").strip():
            st.error("통신에 실패했습니다. 잠시 후 다시 시도해주세요.")
            return "통신에 실패했습니다"

        text = response.text.strip()
        text = text.replace("정답:", "").replace("정답은", "")
        text = text.strip()
        first = text.split()[0] if text.split() else ""
        first = first.strip(" .,!?:;\"'()[]{}")
        return first if first else "모름"
    except Exception:
        st.error("통신에 실패했습니다. 잠시 후 다시 시도해주세요.")
        return "통신에 실패했습니다"


def generate_results_image() -> bytes:
    """결과 요약 PNG 이미지 생성 후 bytes 반환"""
    user_images = st.session_state.user_images
    ai_answers = st.session_state.ai_answers
    correct_answers = st.session_state.correct_answers
    n = len(correct_answers)

    if n == 0:
        img = Image.new("RGB", (800, 300), "white")
        draw = ImageDraw.Draw(img)
        try:
            title_font = ImageFont.truetype("arial.ttf", 40)
        except Exception:
            title_font = ImageFont.load_default()
        draw.text((40, 120), "결과가 없습니다.", font=title_font, fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    width = 1100
    thumb_w, thumb_h = 140, 140
    margin = 40
    row_h = thumb_h + 40
    height = margin * 2 + 60 + n * row_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 40)
        subtitle_font = ImageFont.truetype("arial.ttf", 28)
        main_font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        main_font = ImageFont.load_default()

    y = margin
    draw.text((margin, y), "AI 캐치마인드 결과", font=title_font, fill=(15, 23, 42))
    y += 60

    for i in range(n):
        top = y + i * row_h
        draw.rectangle(
            [(margin - 10, top - 10), (width - margin, top + row_h - 20)],
            outline=(209, 213, 219),
            width=2,
        )

        # 썸네일
        if user_images[i] is not None:
            try:
                thumb = Image.open(io.BytesIO(user_images[i])).convert("RGB")
                thumb.thumbnail((thumb_w, thumb_h))
                img.paste(thumb, (margin, top))
            except Exception:
                pass

        x_text = margin + thumb_w + 20
        ai = ai_answers[i]
        correct = correct_answers[i]

        is_correct = ai == correct
        color_ai = (22, 163, 74) if is_correct else (220, 38, 38)
        emoji = "✅" if is_correct else "❌"

        draw.text((x_text, top), f"{i+1}번 문제", font=subtitle_font, fill=(55, 65, 81))
        draw.text(
            (x_text, top + 40),
            f"{emoji} AI: {ai}",
            font=main_font,
            fill=color_ai,
        )
        draw.text(
            (x_text, top + 80),
            f"🎯 정답: {correct}",
            font=main_font,
            fill=(37, 99, 235),
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ---------- 메인 ----------
init_session_state()

st.markdown('<div class="main-title">🎨 AI 캐치마인드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">태블릿으로 그림을 그리고, AI가 단어를 맞춰보는 게임이에요!</div>', unsafe_allow_html=True)

page = st.session_state.page


# ---------- 시작 화면 ----------
def render_start_page():
    st.markdown("### 1단계 · 카테고리를 고르세요")
    categories = ["동물", "과일", "채소", "사물", "교통수단"]

    cols = st.columns(5)
    for i, cat in enumerate(categories):
        with cols[i]:
            if st.button(cat, use_container_width=True):
                st.session_state.category = cat

    if st.session_state.category:
        st.info(f"선택된 카테고리: **{st.session_state.category}**")

    st.markdown("---")
    st.markdown("### 2단계 · 문항 수를 정하세요")

    st.session_state.target_questions = st.slider(
        "문항 수",
        min_value=3,
        max_value=10,
        value=st.session_state.target_questions,
        step=1,
    )
    st.caption("패스를 고려해 실제로는 ‘문항 수 + 2’개의 문제가 준비됩니다.")

    # 게임 안내는 최소화: 보고 싶은 학생만 펼쳐서 보도록
    with st.expander("게임 방법이 궁금하면 눌러보세요"):
        st.write("- 제한 시간: 각 문제당 **60초**")
        st.write("- 그림을 그리고 **제출**을 누르면 AI가 한 단어로 답해요.")
        st.write("- **패스**는 한 게임에 최대 2번, 패스한 문제는 점수에 들어가지 않아요.")

    st.markdown("---")
    if st.button("🚀 게임 시작하기", type="primary", use_container_width=True):
        if not st.session_state.category:
            st.warning("먼저 카테고리를 선택해주세요!")
        else:
            prepare_problems(st.session_state.category, st.session_state.target_questions)
            if st.session_state.problems:  # 키워드 부족 등 오류 없을 때만 진행
                st.rerun()


# ---------- 게임 화면 ----------
def render_game_page():
    # 종료 조건: 실제 푼 문제 수가 target_questions에 도달했거나, 준비된 문제를 다 소진했을 때
    if (
        st.session_state.answered_count >= st.session_state.target_questions
        or st.session_state.round_index >= len(st.session_state.problems)
    ):
        st.session_state.page = "result"
        st.rerun()

    round_idx = st.session_state.round_index
    current_problem = st.session_state.problems[round_idx]
    current_keyword = current_problem["keyword"]
    category = st.session_state.category

    # ----- '처음으로' 버튼 (모든 상태에서 공통 사용) -----
    top_bar = st.columns([5, 1])
    with top_bar[1]:
        if st.button("↩ 처음으로", use_container_width=True):
            reset_game()
            st.rerun()

    # ----- 제출 후: AI 생각중 화면 -----
    if st.session_state.submitting:
        st.markdown(
            f"### 문제 {st.session_state.answered_count + 1} / {st.session_state.target_questions}"
        )
        st.markdown(
            f'<div class="keyword-box">제시어: <span style="color:#e65100;">{current_keyword}</span></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.session_state.last_snapshot_bytes:
                st.image(
                    st.session_state.last_snapshot_bytes,
                    caption="AI가 보는 마지막 그림",
                    width=320,
                )
        with col2:
            st.info("🧠 AI가 생각중입니다...")

        # 여기서 실제로 AI 호출
        with st.spinner("AI가 그림을 보고 단어를 떠올리고 있어요..."):
            ai_answer = call_gemini(st.session_state.last_snapshot_bytes, category)

        # 방금 푼 문제 정보를 임시 저장 (중간 채점 화면용)
        st.session_state.last_user_image = st.session_state.last_snapshot_bytes
        st.session_state.last_ai_answer = ai_answer
        st.session_state.last_correct_answer = current_keyword
        st.session_state.last_is_correct = ai_answer == current_keyword

        # 다음 단계: 채점 화면
        st.session_state.submitting = False
        st.session_state.grading = True

        st.rerun()
        return

    # ----- 중간 채점 화면 -----
    if st.session_state.grading:
        st.markdown(
            f"### 문제 {st.session_state.answered_count + 1} / {st.session_state.target_questions}"
        )
        st.markdown(
            f'<div class="keyword-box">제시어: <span style="color:#e65100;">{st.session_state.last_correct_answer}</span></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("#### 내가 그린 그림")
            if st.session_state.last_user_image:
                st.image(st.session_state.last_user_image, width=260)
            else:
                st.write("그림이 저장되지 않았어요.")

        with col2:
            st.markdown("#### AI 채점 결과")

            ai_ans = st.session_state.last_ai_answer or "응답 없음"
            correct = st.session_state.last_correct_answer or "정답 없음"
            is_correct = st.session_state.last_is_correct

            # ✅ / ❌ 한 줄로 눈에 확 들어오게
            if is_correct:
                st.markdown(
                    "<div style='font-size:1.6rem; font-weight:700; color:#16a34a; margin-bottom:0.5rem;'>"
                    "✅ 정답!"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='font-size:1.6rem; font-weight:700; color:#dc2626; margin-bottom:0.5rem;'>"
                    "❌ 오답!"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # AI 답 / 정답은 짧고 선명하게
            st.markdown(
                f"<div style='font-size:1.3rem; margin-bottom:0.2rem;'>"
                f"🤖 <b>AI 답:</b> <span style='color:#b91c1c;'>{ai_ans}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:1.3rem;'>"
                f"🎯 <b>정답:</b> <span style='color:#1d4ed8;'>{correct}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            if st.button("➡ 다음 문제로", use_container_width=True):
                # 최종 결과 리스트에 확정 반영
                st.session_state.user_images.append(st.session_state.last_user_image)
                st.session_state.ai_answers.append(ai_ans)
                st.session_state.correct_answers.append(correct)
                st.session_state.answered_count += 1

                # 다음 문제로 이동
                st.session_state.round_index += 1
                st.session_state.start_time = time.time()
                st.session_state.last_snapshot_bytes = None

                # 중간 채점 상태 초기화
                st.session_state.grading = False
                st.session_state.last_user_image = None
                st.session_state.last_ai_answer = None
                st.session_state.last_correct_answer = None
                st.session_state.last_is_correct = None

                # 다 풀었으면 결과 페이지로
                if (
                    st.session_state.answered_count >= st.session_state.target_questions
                    or st.session_state.round_index >= len(st.session_state.problems)
                ):
                    st.session_state.page = "result"

                st.rerun()
                return

        return

    # ----- 일반 게임 화면 (그리기 + 타이머 + 패스/제출) -----
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    elapsed = time.time() - st.session_state.start_time
    remaining = max(0, int(60 - elapsed))
    time_over = remaining <= 0

    top1, top2 = st.columns([2, 2])

    with top1:
        st.markdown(
            f"#### 문제 {st.session_state.answered_count + 1} / {st.session_state.target_questions}"
        )
        st.markdown(
            f'<div class="keyword-box">제시어: <span style="color:#e65100;">{current_keyword}</span></div>',
            unsafe_allow_html=True,
        )

    with top2:
        st.markdown("#### 남은 시간")
        st.markdown(
            f'<div class="timer-box">⏱ {remaining}초 남았어요!</div>',
            unsafe_allow_html=True,
        )
        progress = remaining / 60
        st.progress(progress if progress >= 0 else 0)
        st.markdown(
            f"- 사용한 패스: **{st.session_state.passes_used} / {st.session_state.max_passes}**"
        )

    st.markdown("---")

    left, right = st.columns([3, 2])

    # ----- 왼쪽: 팔레트 + 캔버스 + 제출/패스 -----
    with left:
        st.markdown("#### 1) 팔레트 & 그림 그리기")

        # 색상 팔레트 (radio)
        color_label = st.radio(
            "자주 쓰는 색상",
            options=["⚫ 검정", "🔴 빨강", "🔵 파랑", "🟢 초록"],
            horizontal=True,
        )

        if "검정" in color_label:
            stroke_color = "#000000"
        elif "빨강" in color_label:
            stroke_color = "#ef4444"
        elif "파랑" in color_label:
            stroke_color = "#3b82f6"
        else:
            stroke_color = "#22c55e"

        if not time_over:
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=8,
                stroke_color=stroke_color,
                background_color="#FFFFFF",
                width=600,   # 캔버스 넓게
                height=450,
                drawing_mode="freedraw",
                key=f"canvas_{round_idx}",
            )

            if canvas_result.image_data is not None:
                img_pil = pil_from_canvas(canvas_result.image_data)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.session_state.last_snapshot_bytes = buf.getvalue()
        else:
            st.warning("⏰ 제한 시간이 끝났어요! 더 이상 그림을 그릴 수 없어요.")
            if st.session_state.last_snapshot_bytes:
                st.image(
                    st.session_state.last_snapshot_bytes,
                    caption="마지막으로 그린 그림",
                    width=320,
                )
            else:
                st.info("시간 안에 그린 그림이 없어요.")

        st.markdown("#### 2) 제출 / 패스")

        # 제출 / 패스 버튼 한 줄 배치
        bcol1, bcol2, _ = st.columns([1, 1, 1])

        # 제출 버튼 활성화 조건
        # - 시간 안엔 그림이 있어야 제출 가능
        # - 시간이 지나면 그림이 없어도 제출 가능(빈 그림 생성)
        if time_over and st.session_state.last_snapshot_bytes is None:
            submit_disabled = False
        else:
            submit_disabled = st.session_state.last_snapshot_bytes is None

        with bcol1:
            if st.button("✅ 제출", use_container_width=True, disabled=submit_disabled):
                if st.session_state.last_snapshot_bytes is None:
                    # 완전히 빈 그림인 경우 흰 이미지 생성 (캔버스와 동일 크기)
                    blank = Image.new("RGB", (600, 450), "white")
                    buf = io.BytesIO()
                    blank.save(buf, format="PNG")
                    st.session_state.last_snapshot_bytes = buf.getvalue()

                st.session_state.submitting = True
                st.rerun()

        with bcol2:
            pass_disabled = st.session_state.passes_used >= st.session_state.max_passes
            if st.button("⏭ 패스", use_container_width=True, disabled=pass_disabled):
                if st.session_state.passes_used >= st.session_state.max_passes:
                    st.warning("패스는 한 게임에 최대 2번까지만 사용할 수 있어요.")
                else:
                    st.session_state.passes_used += 1
                    st.session_state.round_index += 1
                    st.session_state.start_time = time.time()
                    st.session_state.last_snapshot_bytes = None
                    st.rerun()

    # ----- 오른쪽: 간단한 현재 상태 요약 -----
    with right:
        st.markdown("#### 현재 진행 상황")
        st.write(f"- 푼 문제 수: **{st.session_state.answered_count}** / {st.session_state.target_questions}")
        st.write(f"- 남은 패스: **{st.session_state.max_passes - st.session_state.passes_used}** 회")


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
                # 결과 화면에서도 한 눈에 들어오도록 크기 조정
                st.image(st.session_state.user_images[i], width=260)
            else:
                st.write("저장된 그림이 없습니다.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            ai_ans = st.session_state.ai_answers[i] if i < len(st.session_state.ai_answers) else "응답 없음"
            correct = (
                st.session_state.correct_answers[i]
                if i < len(st.session_state.correct_answers)
                else "정답 없음"
            )

            is_correct = ai_ans == correct
            if is_correct:
                st.markdown(
                    f"<div style='font-size:1.4rem; color:#15803d; margin-bottom:0.5rem;'>"
                    f"✅ <b>AI 응답:</b> {ai_ans}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:1.4rem; color:#1d4ed8;'>"
                    f"🎯 <b>정답:</b> {correct}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='font-size:1.4rem; color:#dc2626; margin-bottom:0.5rem;'>"
                    f"❌ <b>AI 응답:</b> {ai_ans}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:1.4rem; color:#1d4ed8;'>"
                    f"🎯 <b>정답:</b> {correct}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

    st.markdown("### 📥 결과 저장")

    png_bytes = generate_results_image()
    st.download_button(
        label="🖼 PNG로 다운",
        data=png_bytes,
        file_name="catchmind_results.png",
        mime="image/png",
        use_container_width=True,
    )

    st.markdown("### 다시 해볼까요?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 같은 설정으로 다시 하기", use_container_width=True):
            cat = st.session_state.category
            n_questions = st.session_state.target_questions
            reset_game()
            st.session_state.category = cat
            st.session_state.target_questions = n_questions
            prepare_problems(cat, n_questions)
            if st.session_state.problems:
                st.rerun()
    with col2:
        if st.button("🏠 처음 화면으로 돌아가기", use_container_width=True):
            reset_game()
            st.rerun()


# ---------- 실행 ----------
if page == "start":
    render_start_page()
elif page == "game":
    render_game_page()
elif page == "result":
    render_result_page()
else:
    reset_game()
    render_start_page()
