import streamlit as st
from streamlit_js_eval import get_geolocation
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
import requests
import urllib.parse
import json

# ==========================================
# [설정] 페이지 및 API 키
# ==========================================
st.set_page_config(
    page_title="스마트 팜 AI 진단",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NAVER_CLIENT_ID = "2wR8x89ky2cwkwIspEyj"
NAVER_CLIENT_SECRET = "uw_h22JCJR"
WEATHER_API_KEY = "f9408d1bd75131dddadd813aaa4809b4"

# ==========================================
# [스타일] CSS (상단 여백 완벽 제거 버전)
# ==========================================
st.markdown("""
<style>
    /* 전체 배경색 */
    .stApp { background-color: #f4f6f8; }

    /* 1. 메인 컨테이너 위쪽 여백 강제 제거 */
    .block-container {
        padding-top: 1rem !important; /* 0으로 하면 너무 붙어서 1rem 정도 줌 */
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }

    /* 2. Streamlit 기본 헤더(햄버거 메뉴 등) 숨기기 */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* 3. 커스텀 헤더 스타일 (위로 끌어올리기) */
    .custom-header {
        background: #27ae60; 
        color: white; 
        padding: 15px 20px; 
        font-size: 1.5rem;
        font-weight: bold; 
        border-radius: 0 0 10px 10px; 
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        display: flex; 
        align-items: center; 
        gap: 10px;

        /* ★ [핵심] 음수 마진으로 강제로 위로 올림 */
        margin-top: -60px !important; 
        z-index: 999;
    }

    /* 카드 스타일 (높이 고정 및 스크롤) */
    .css-card {
        background: white; border-radius: 15px; padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px;
        height: 80vh;       
        overflow-y: auto;   
    }

    /* 스크롤바 디자인 */
    .css-card::-webkit-scrollbar { width: 8px; }
    .css-card::-webkit-scrollbar-thumb { background-color: #bdc3c7; border-radius: 4px; }

    /* 기타 폰트 및 버튼 스타일 */
    .section-title {
        color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px;
        margin-bottom: 20px; font-size: 1.2rem; font-weight: bold;
    }
    .weather-box {
        background: #e3f2fd; padding: 15px; border-radius: 8px;
        border-left: 5px solid #2196f3; margin-top: 15px;
    }
    .news-item { display: flex; gap: 15px; padding: 15px 0; border-bottom: 1px solid #f1f1f1; text-decoration: none; color: inherit; transition: background 0.2s; }
    .news-item:hover { background-color: #fafafa; }
    .news-thumb { min-width: 80px; height: 80px; background: #eee; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #999; font-weight: bold; font-size: 0.8rem; }
    .news-content { flex: 1; }
    .news-title { font-weight: bold; font-size: 1rem; color: #333; display: block; margin-bottom: 5px;}
    .news-desc { font-size: 0.85rem; color: #666; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .news-date { font-size: 0.75rem; color: #999; margin-top: 5px; }
    .stButton > button { width: 100%; background-color: #3498db; color: white; border-radius: 8px; font-weight: bold; border: none; }
    .stButton > button:hover { background-color: #2980b9; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# [설정] 6개 작물별 모델
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
# [함수] 로직
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


def get_weather_by_coords(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") != 200: return None
        return {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "desc": res["weather"][0]["description"],
            "city": res.get("name", "Unknown Location")
        }
    except:
        return None


def get_weather_by_city(city="Seoul"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") != 200: return None
        return {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "desc": res["weather"][0]["description"],
            "city": res.get("name", city)
        }
    except:
        return None


def get_naver_news(keyword):
    try:
        encText = urllib.parse.quote(keyword)
        # 뉴스 10개로 제한
        url = "https://openapi.naver.com/v1/search/news?query=" + encText + "&display=10&sort=sim"
        headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
        response = requests.get(url, headers=headers)
        if response.status_code == 200: return response.json()['items']
        return []
    except:
        return []


# ==========================================
# [UI] 화면 구성
# ==========================================
# ★ [핵심] 커스텀 헤더 (margin-top: -60px 적용됨)
st.markdown('<div class="custom-header">🌿 스마트 팜 AI 플랫폼</div>', unsafe_allow_html=True)

# GPS 요청
location = get_geolocation()

col_left, col_right = st.columns([1.5, 1])

# === 왼쪽 컬럼 ===
with col_left:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 작물 AI 진단</div>', unsafe_allow_html=True)

    selected_crop = st.radio("작물을 선택하세요", list(CROP_CONFIG.keys()), horizontal=True)
    st.write("---")

    tab1, tab2 = st.tabs(["📸 카메라 촬영", "📂 앨범 선택"])
    image = None

    with tab1:
        cam_file = st.camera_input("작물을 촬영하세요")
        if cam_file: image = Image.open(cam_file).convert('RGB')

    with tab2:
        up_file = st.file_uploader("사진 파일 선택", type=["jpg", "png", "jpeg"])
        if up_file: image = Image.open(up_file).convert('RGB')

    if image:
        st.image(image, caption='분석할 이미지', use_column_width=True)

        if st.button("🚀 진단 시작"):
            with st.spinner("AI가 분석 중입니다..."):
                model = load_model_for_crop(selected_crop)
                if model:
                    input_tensor = preprocess_image(image)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        top_prob, top_idx = torch.max(probabilities, 0)

                        class_names = CROP_CONFIG[selected_crop]["classes"]
                        predicted_class = class_names[top_idx]
                        confidence = top_prob.item() * 100

                    st.session_state['last_pred'] = predicted_class
                    st.session_state['last_conf'] = confidence
                else:
                    st.error("모델 파일이 없습니다.")

    # 결과 표시
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

        # 날씨
        weather = None
        if location and 'coords' in location:
            lat = location['coords']['latitude']
            lon = location['coords']['longitude']
            weather = get_weather_by_coords(lat, lon)
            loc_label = f"{weather['city']} (내 위치)"
        else:
            weather = get_weather_by_city("Seoul")
            loc_label = "Seoul (위치 권한 없음)"

        if weather:
            st.markdown(f"""
            <div class="weather-box">
                <strong style="color: #1565c0;">🌤️ 실시간 환경 분석 - {loc_label}</strong><br>
                기온: <b>{weather['temp']}°C</b> / 습도: <b>{weather['humidity']}%</b><br>
                <span style="font-size: 0.9rem; color: #555;">습도가 70% 이상이면 곰팡이병에 주의하세요.</span>
            </div>
            """, unsafe_allow_html=True)

        # 챗봇
        st.write("---")
        st.subheader("💬 AI 농업 챗봇")
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            reply = f"'{pred}'에 대한 답변: 전문가와 상담하세요."
            if "예방" in prompt: reply = "통풍과 배수가 가장 중요합니다."
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    st.markdown('</div>', unsafe_allow_html=True)

# === 오른쪽 컬럼: 뉴스 ===
with col_right:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📰 관련 농업 뉴스</div>', unsafe_allow_html=True)

    keyword = st.session_state.get('last_pred', f"{selected_crop} 병해충")
    keyword = keyword.split('(')[0] + " 방제"
    news_items = get_naver_news(keyword)

    if news_items:
        # 뉴스 중복 제거
        seen_links = set()
        unique_news = []
        for item in news_items:
            if item['link'] not in seen_links:
                seen_links.add(item['link'])
                unique_news.append(item)

        for item in unique_news:
            title = item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
            desc = item['description'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
            link = item['link']
            date = item['pubDate'][:16]
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