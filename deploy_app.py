import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
import requests
import urllib.parse

# ==========================================
# [설정] API 키 및 페이지 설정
# ==========================================
st.set_page_config(
    page_title="스마트 팜 AI 진단",
    page_icon="🌿",
    layout="wide",  # ★ 중요: 2단 레이아웃을 위해 넓게 쓰기
    initial_sidebar_state="collapsed"
)

NAVER_CLIENT_ID = "2wR8x89ky2cwkwIspEyj"
NAVER_CLIENT_SECRET = "uw_h22JCJR"
WEATHER_API_KEY = "f9408d1bd75131dddadd813aaa4809b4"

# ==========================================
# [스타일] CSS 주입 (보내주신 HTML 디자인 적용)
# ==========================================
st.markdown("""
<style>
    /* 전체 배경색 */
    .stApp {
        background-color: #f4f6f8;
    }

    /* 상단 헤더 숨기기 (Streamlit 기본 헤더) */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* 커스텀 헤더 스타일 */
    .custom-header {
        background: #27ae60;
        color: white;
        padding: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        border-radius: 0 0 10px 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* 카드 스타일 (흰색 박스) */
    .css-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* 제목 스타일 */
    .section-title {
        color: #2c3e50;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 1.2rem;
        font-weight: bold;
    }

    /* 뉴스 아이템 스타일 */
    .news-item {
        display: flex;
        gap: 15px;
        padding: 15px 0;
        border-bottom: 1px solid #f1f1f1;
        text-decoration: none;
        color: inherit;
        transition: 0.2s;
    }
    .news-item:hover { background-color: #fafafa; }
    .news-thumb {
        min-width: 80px; height: 80px;
        background: #eee; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        color: #999; font-weight: bold; font-size: 0.8rem;
    }
    .news-content { flex: 1; }
    .news-title { font-weight: bold; font-size: 1rem; color: #333; display: block; margin-bottom: 5px;}
    .news-desc { font-size: 0.85rem; color: #666; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .news-date { font-size: 0.75rem; color: #999; margin-top: 5px; }

    /* 날씨 카드 스타일 */
    .weather-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin-top: 15px;
    }

    /* 버튼 스타일 조정 */
    .stButton > button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# [설정] 6개 작물별 모델 및 클래스
# ==========================================
CROP_CONFIG = {
    "고추": {"file": "pepper_model.pth", "classes": ['고추 (정상)', '고추 (마일드모틀바이러스)', '고추 (점무늬병)']},
    "토마토": {"file": "tomato_model.pth", "classes": ['토마토 (정상)', '토마토 (잎곰팡이병)', '토마토 (황화잎말이바이러스)']},
    "딸기": {"file": "strawberry_model.pth", "classes": ['딸기 (정상)', '딸기 (잿빛곰팡이병)', '딸기 (흰가루병)']},
    "상추": {"file": "lettuce_model.pth", "classes": ['상추 (정상)', '상추 (노균병)', '상추 (균핵병)']},
    "오이": {"file": "cucumber_model.pth", "classes": ['오이 (정상)', '오이 (모자이크바이러스)', '오이 (녹반모자이크바이러스)']},
    "포도": {"file": "grape_model.pth", "classes": ['포도 (정상)', '포도 (노균병)']}
}


# ==========================================
# [함수] 로직 (모델, 날씨, 뉴스)
# ==========================================
@st.cache_resource
def load_model_for_crop(crop_name):
    config = CROP_CONFIG[crop_name]
    model_path = config["file"]
    num_classes = len(config["classes"])

    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    if os.path.exists(model_path):
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model.eval()
        return model
    return None


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def get_weather_info():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Seoul&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") != 200: return None
        return {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "desc": res["weather"][0]["description"]
        }
    except:
        return None


def get_naver_news(keyword):
    try:
        encText = urllib.parse.quote(keyword)
        url = "https://openapi.naver.com/v1/search/news?query=" + encText + "&display=5&sort=sim"
        headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
        response = requests.get(url, headers=headers)
        if response.status_code == 200: return response.json()['items']
        return []
    except:
        return []


# ==========================================
# [UI] 화면 구성 (2단 레이아웃)
# ==========================================

# 1. 커스텀 헤더 출력
st.markdown('<div class="custom-header">🌿 스마트 팜 AI 플랫폼</div>', unsafe_allow_html=True)

# 2. 메인 레이아웃 분할 (왼쪽: 진단 60% / 오른쪽: 뉴스 40%)
col_left, col_right = st.columns([1.5, 1])

# === [왼쪽 컬럼] AI 진단 섹션 ===
with col_left:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 작물 AI 진단</div>', unsafe_allow_html=True)

    # 1. 작물 선택 (가로형 라디오 버튼 느낌)
    selected_crop = st.radio("작물을 선택하세요", list(CROP_CONFIG.keys()), horizontal=True)

    # 2. 이미지 업로드
    st.write("---")
    uploaded_file = st.file_uploader("사진을 업로드하거나 촬영하세요", type=["jpg", "png", "jpeg"])

    result_placeholder = st.empty()  # 결과가 들어갈 자리

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='선택된 이미지', use_column_width=True)

        if st.button("🚀 진단 시작"):
            with st.spinner("AI가 잎사귀를 분석 중입니다..."):
                model = load_model_for_crop(selected_crop)
                if model:
                    # 추론
                    input_tensor = preprocess_image(image)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        top_prob, top_idx = torch.max(probabilities, 0)

                        class_names = CROP_CONFIG[selected_crop]["classes"]
                        predicted_class = class_names[top_idx]
                        confidence = top_prob.item() * 100

                    # 진단 결과 저장
                    st.session_state['last_pred'] = predicted_class
                    st.session_state['last_conf'] = confidence
                else:
                    st.error("모델 파일이 없습니다.")

    # 진단 결과 표시 영역
    if 'last_pred' in st.session_state:
        pred = st.session_state['last_pred']
        conf = st.session_state['last_conf']

        st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: #e74c3c; margin: 0;">{pred}</h2>
            <p style="color: #7f8c8d;">신뢰도: {conf:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(conf))

        # 날씨 카드
        weather = get_weather_info()
        if weather:
            st.markdown(f"""
            <div class="weather-box">
                <strong style="color: #1565c0;">🌤️ 실시간 환경 분석 (Seoul)</strong><br>
                기온: <b>{weather['temp']}°C</b> / 습도: <b>{weather['humidity']}%</b><br>
                <span style="font-size: 0.9rem; color: #555;">습도가 70% 이상이면 곰팡이병에 주의하세요.</span>
            </div>
            """, unsafe_allow_html=True)

        # 챗봇 (Streamlit ChatInput 사용)
        st.write("---")
        st.subheader("💬 AI 농업 챗봇")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 이전 대화 출력
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 채팅 입력
        if prompt := st.chat_input("질문을 입력하세요 (예: 예방법은?)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 간단한 규칙 기반 답변
            response = f"'{pred}'에 대해 문의하셨군요. 가까운 농약사를 방문하여 전문가의 처방을 받는 것을 추천드립니다."
            if "예방" in prompt:
                response = "환기를 자주 시키고 적정 습도를 유지하는 것이 가장 좋은 예방법입니다."

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown('</div>', unsafe_allow_html=True)  # 카드 닫기

# === [오른쪽 컬럼] 뉴스 섹션 ===
with col_right:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📰 관련 농업 뉴스</div>', unsafe_allow_html=True)

    # 검색어 결정
    search_keyword = st.session_state.get('last_pred', f"{selected_crop} 병해충")
    search_keyword = search_keyword.split('(')[0] + " 방제"  # 검색어 정제

    news_items = get_naver_news(search_keyword)

    if news_items:
        for item in news_items:
            title = item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
            desc = item['description'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
            link = item['link']
            date = item['pubDate'][:16]

            # HTML로 뉴스 아이템 렌더링
            st.markdown(f"""
            <a href="{link}" target="_blank" class="news-item">
                <div class="news-thumb">NEWS</div>
                <div class="news-content">
                    <span class="news-title">{title}</span>
                    <span class="news-desc">{desc}</span>
                    <div class="news-date">{date}</div>
                </div>
            </a>
            """, unsafe_allow_html=True)
    else:
        st.info("관련 뉴스를 찾을 수 없습니다.")

    st.markdown('</div>', unsafe_allow_html=True)