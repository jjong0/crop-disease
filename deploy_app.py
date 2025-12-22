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
    page_title="스마트 팜",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NAVER_CLIENT_ID = "2wR8x89ky2cwkwIspEyj"
NAVER_CLIENT_SECRET = "uw_h22JCJR"
WEATHER_API_KEY = "f9408d1bd75131dddadd813aaa4809b4"

# ==========================================
# [스타일] CSS (다크모드 글씨 안보임 해결 완벽 버전)
# ==========================================
st.markdown("""
<style>
    /* 1. 전체 배경색 및 기본 폰트 색상 강제 지정 */
    .stApp { 
        background-color: #f4f6f8;
        color: #000000 !important; /* 기본 글자 검은색 */
    }

    /* 2. 라디오 버튼, 체크박스 등 위젯 라벨 강제 검은색 (★중요★) */
    .stRadio label p {
        color: #000000 !important;
        font-weight: bold;
    }
    .stRadio div[role='radiogroup'] {
        color: #000000 !important;
    }

    /* 3. 일반 텍스트(p), 제목(h) 강제 검은색 */
    p, h1, h2, h3, h4, h5, h6, span, label {
        color: #000000 !important;
    }

    /* 4. 탭(Tabs) 글씨 색상 */
    button[data-baseweb="tab"] div {
        color: #000000 !important;
    }

    /* 5. 상단 여백 제거 */
    .block-container {
        padding-top: 0px !important; 
        padding-bottom: 2rem !important;
    }

    /* 6. Streamlit 기본 헤더 숨기기 */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* 7. 커스텀 헤더 스타일 */
    .custom-header {
        background: #27ae60; 
        color: white !important; /* 헤더 글씨는 흰색 유지 */
        padding: 20px; 
        font-size: 1.5rem; 
        font-weight: bold; 
        border-radius: 0 0 10px 10px; 
        margin-bottom: 20px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        display: flex; 
        align-items: center; 
        gap: 10px;
        margin-top: 0px !important; 
    }
    /* 헤더 내부 텍스트는 흰색이어야 하므로 재지정 */
    .custom-header span, .custom-header div {
        color: white !important;
    }

    /* 8. 컬럼 스타일 (카드 형태) */
    [data-testid="column"] {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }

    /* 9. 제목 스타일 */
    .section-title {
        color: #2c3e50 !important; 
        border-bottom: 2px solid #eee; 
        padding-bottom: 10px;
        margin-bottom: 20px; 
        font-size: 1.2rem; 
        font-weight: bold;
    }

    /* 10. 날씨 박스 */
    .weather-box {
        background: #e3f2fd; 
        padding: 15px; 
        border-radius: 8px;
        border-left: 5px solid #2196f3; 
        margin-top: 15px;
        color: #000000 !important;
    }

    /* 11. 뉴스 아이템 */
    .news-item { display: flex; gap: 15px; padding: 15px 0; border-bottom: 1px solid #f1f1f1; text-decoration: none; color: inherit; transition: background 0.2s; }
    .news-item:hover { background-color: #fafafa; }
    .news-thumb { min-width: 80px; height: 80px; background: #eee; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #999 !important; font-weight: bold; font-size: 0.8rem; }
    .news-content { flex: 1; }
    .news-title { font-weight: bold; font-size: 1rem; color: #333 !important; display: block; margin-bottom: 5px;}
    .news-desc { font-size: 0.85rem; color: #666 !important; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .news-date { font-size: 0.75rem; color: #999 !important; margin-top: 5px; }

    /* 12. 버튼 스타일 */
    .stButton > button { width: 100%; background-color: #3498db; color: white !important; border-radius: 8px; font-weight: bold; border: none; }
    .stButton > button:hover { background-color: #2980b9; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# [설정] 모델 정보
# ==========================================
CROP_CONFIG = {
    "고추": {"file": "pepper_model.pth", "classes": ['고추 (정상)', '고추 (마일드모틀바이러스)', '고추 (점무늬병)']},
    "토마토": {"file": "tomato_model.pth", "classes": ['토마토 (정상)', '토마토 (잎곰팡이병)', '토마토 (황화잎말이바이러스)']},
    "딸기": {"file": "strawberry_model.pth", "classes": ['딸기 (정상)', '딸기 (잿빛곰팡이병)', '딸기 (흰가루병)']},
    "상추": {"file": "lettuce_model.pth", "classes": ['상추 (정상)', '상추 (노균병)', '상추 (균핵병)']},
    "오이": {"file": "cucumber_model.pth", "classes": ['오이 (정상)', '오이 (모자이크바이러스)', '오이 (녹반모자이크바이러스)']},
    "포도": {"file": "grape_model.pth", "classes": ['포도 (정상)', '포도 (노균병)']}
}


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
            "temp": res["main"]["temp"], "humidity": res["main"]["humidity"],
            "desc": res["weather"][0]["description"], "city": res.get("name", "Unknown")
        }
    except:
        return None


def get_weather_by_city(city="Seoul"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") != 200: return None
        return {
            "temp": res["main"]["temp"], "humidity": res["main"]["humidity"],
            "desc": res["weather"][0]["description"], "city": res.get("name", city)
        }
    except:
        return None


def get_naver_news(keyword):
    try:
        encText = urllib.parse.quote(keyword)
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
st.markdown('<div class="custom-header">🌿 스마트 팜</div>', unsafe_allow_html=True)
location = get_geolocation()

col_left, col_right = st.columns([1.5, 1], gap="medium")

# === 왼쪽 컬럼: 진단 ===
with col_left:
    st.markdown('<div class="section-title">🩺 작물 진단</div>', unsafe_allow_html=True)

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
                        # 상위 2개 클래스 확률 추출 (불확실성 계산용)
                        top2 = torch.topk(probabilities, 2)
                        confidence_gap = (top2.values[0] - top2.values[1]).item() * 100

                        # 불확실성 레벨 정의 (설명용)
                        if confidence_gap >= 30:
                            certainty_level = "높음"
                        elif confidence_gap >= 15:
                            certainty_level = "보통"
                        else:
                            certainty_level = "낮음"

                        # 세션 저장
                        st.session_state['confidence_gap'] = confidence_gap
                        st.session_state['certainty_level'] = certainty_level
                        st.session_state['top2_classes'] = [
                            CROP_CONFIG[selected_crop]["classes"][top2.indices[0]],
                            CROP_CONFIG[selected_crop]["classes"][top2.indices[1]]
                        ]
                        class_names = CROP_CONFIG[selected_crop]["classes"]
                        predicted_class = class_names[top_idx]
                        confidence = top_prob.item() * 100
                    st.session_state['last_pred'] = predicted_class
                    st.session_state['last_conf'] = confidence
                else:
                    st.error("모델 파일이 없습니다.")

    # ----------------------------------------------------
    # ★ 결과 출력 카드 (검은색 글씨 고정) ★
    # ----------------------------------------------------
    if 'last_pred' in st.session_state:
        pred = st.session_state['last_pred']
        conf = st.session_state['last_conf']

        # HTML 코드는 들여쓰기 없이 작성해야 텍스트 노출 방지됨
        html_code = f"""
<div style="background-color: #FFFFFF; padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; border: 1px solid #e0e0e0;">
<p style="color: #000000; font-size: 14px; margin-bottom: 5px; font-weight: bold;">분석 결과</p>
<h2 style="color: #000000; font-weight: bold; margin: 0; margin-bottom: 10px;">{pred}</h2>
<p style="color: #4CAF50; font-weight: bold; font-size: 16px; margin: 0;">신뢰도: {conf:.2f}%</p>
</div>
"""
        st.markdown(html_code, unsafe_allow_html=True)

        # ==============================
        # 🧠 모델 예측 신뢰성 설명
        # ==============================
        gap = st.session_state['confidence_gap']
        level = st.session_state['certainty_level']
        top2_cls = st.session_state['top2_classes']

        st.markdown(f"""
        <div style="background:#f1f8e9; padding:15px; border-radius:12px; margin-top:10px; border-left:5px solid #8bc34a;">
        <b>🧠 모델 예측 신뢰성 설명</b><br>
        예측 확실성 수준: <b>{level}</b><br>
        1순위–2순위 예측 확률 차이: <b>{gap:.1f}%</b><br>
        <span style="font-size:0.9rem;">
        모델은 <b>{top2_cls[0]}</b>와 <b>{top2_cls[1]}</b> 사이에서 상대적으로 더 높은 확률을 보였습니다.
        </span>
        </div>
        """, unsafe_allow_html=True)

        st.caption("※ 본 정보는 모델 출력 분포를 설명하기 위한 것으로, 최종 진단을 대체하지 않습니다.")

        st.progress(int(conf))

        weather = None
        if location and 'coords' in location:
            weather = get_weather_by_coords(location['coords']['latitude'], location['coords']['longitude'])
            loc_label = f"{weather['city']} (내 위치)"
        else:
            weather = get_weather_by_city("Seoul")
            loc_label = "Seoul (위치 권한 없음)"

        if weather:
            st.markdown(f"""
            <div class="weather-box">
                <strong style="color: #1565c0;">🌤️ 실시간 환경 분석 - {loc_label}</strong><br>
                <span style="color: #000000;">기온: <b>{weather['temp']}°C</b> / 습도: <b>{weather['humidity']}%</b></span><br>
                <span style="font-size: 0.9rem; color: #333333;">습도가 70% 이상이면 곰팡이병에 주의하세요.</span>
            </div>
            """, unsafe_allow_html=True)
        # ===============================
        # 병해 + 기상 기반 위험 추세 분석
        # ===============================

        top1_class = st.session_state["predicted_class"]
        top1_prob = st.session_state["predicted_prob"] * 100
        temp = st.session_state["temperature"]
        humidity = st.session_state["humidity"]

        model_confident = top1_prob >= 70
        high_risk_weather = (humidity >= 80) and (temp >= 25)

        if model_confident and high_risk_weather:
            risk_level = "높음"
            color = "#ffebee"
            border = "#f44336"
        elif model_confident or high_risk_weather:
            risk_level = "중간"
            color = "#fff8e1"
            border = "#ff9800"
        else:
            risk_level = "낮음"
            color = "#e8f5e9"
            border = "#4caf50"

        st.markdown(f"""
        <div style="background:{color}; padding:18px; border-radius:14px;
                    border-left:6px solid {border}; margin-top:15px;">
        <b>📈 병해 확산 위험 추세 분석</b><br><br>

        <b>• 모델 예측 결과</b><br>
        - 주요 병해 유형: <b>{top1_class}</b><br>
        - 모델 분류 신뢰도: <b>{top1_prob:.1f}%</b><br><br>

        <b>• 환경 조건 분석</b><br>
        - 평균 기온: {temp}℃<br>
        - 평균 습도: {humidity}%<br><br>

        <b>▶ 종합 판단</b><br>
        병해 확산 위험 추세: <b>{risk_level}</b>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "※ 본 결과는 이미지 분류 모델 출력과 기상 조건을 "
            "종합한 관리 참고 지표이며, 실제 병 발생 확률을 의미하지 않습니다."
        )

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

# === 오른쪽 컬럼: 뉴스 ===
with col_right:
    st.markdown('<div class="section-title">📰 관련 농업 뉴스</div>', unsafe_allow_html=True)

    keyword = st.session_state.get('last_pred', f"{selected_crop} 병해충")
    keyword = keyword.split('(')[0] + " 방제"
    news_items = get_naver_news(keyword)

    # 뉴스 스크롤 컨테이너
    with st.container(height=600, border=False):
        if news_items:
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