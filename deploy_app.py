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
# [스타일] CSS
# ==========================================
st.markdown("""
<style>
    /* 전체 폰트 및 배경 설정 */
    .stApp { background-color: #f4f6f8; color: #000000 !important; }

    /* 글자색 강제 검정 */
    p, h1, h2, h3, h4, h5, h6, span, label, div[role='radiogroup'] { color: #000000 !important; }
    .stRadio label p { color: #000000 !important; font-weight: bold; }

    /* 상단 여백 제거 */
    .block-container { padding-top: 0px !important; padding-bottom: 2rem !important; }
    header[data-testid="stHeader"] { display: none !important; }

    /* 커스텀 헤더 */
    .custom-header {
        background: #27ae60; color: white !important; padding: 20px; 
        font-size: 1.5rem; font-weight: bold; border-radius: 0 0 10px 10px; 
        margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        display: flex; align-items: center; gap: 10px; margin-top: 0px !important; 
    }
    .custom-header span, .custom-header div { color: white !important; }

    /* 카드 스타일 */
    [data-testid="column"] {
        background-color: white; border-radius: 15px; padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #eee;
    }

    /* 섹션 제목 */
    .section-title {
        color: #2c3e50 !important; border-bottom: 2px solid #eee; 
        padding-bottom: 10px; margin-bottom: 20px; font-size: 1.2rem; font-weight: bold;
    }

    /* 날씨 박스 */
    .weather-box {
        background: #e3f2fd; padding: 15px; border-radius: 8px;
        border-left: 5px solid #2196f3; margin-top: 15px; color: #000000 !important;
    }

    /* 뉴스 스타일 */
    .news-item { display: flex; gap: 15px; padding: 15px 0; border-bottom: 1px solid #f1f1f1; text-decoration: none; color: inherit; transition: background 0.2s; }
    .news-item:hover { background-color: #fafafa; }
    .news-thumb { min-width: 80px; height: 80px; background: #eee; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #999 !important; font-weight: bold; font-size: 0.8rem; }
    .news-content { flex: 1; }
    .news-title { font-weight: bold; font-size: 1rem; color: #333 !important; display: block; margin-bottom: 5px;}
    .news-desc { font-size: 0.85rem; color: #666 !important; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .news-date { font-size: 0.75rem; color: #999 !important; margin-top: 5px; }

    /* 버튼 */
    .stButton > button { width: 100%; background-color: #3498db; color: white !important; border-radius: 8px; font-weight: bold; border: none; }
    .stButton > button:hover { background-color: #2980b9; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# [설정] 모델 및 질병 정보
# ==========================================
CROP_CONFIG = {
    "고추": {"file": "pepper_model.pth", "classes": ['고추 (정상)', '고추 (마일드모틀바이러스)', '고추 (점무늬병)'],
           "risk_env": {
               "점무늬병": {"습도": "80% 이상", "기온": "20~30℃", "특징": "장마철, 통풍 불량 시 급속 확산"},
               "마일드모틀바이러스": {"습도": "영향 적음", "기온": "20~28℃", "특징": "작업 도구, 토양 전염"}
           },
           "causes": {
               "점무늬병": ["곰팡이 감염", "고온다습 환경", "연작 피해"],
               "마일드모틀바이러스": ["감염 종자", "진딧물 매개", "작업자 손 접촉"]
           }},
    "토마토": {"file": "tomato_model.pth", "classes": ['토마토 (정상)', '토마토 (잎곰팡이병)', '토마토 (황화잎말이바이러스)'],
            "risk_env": {
                "잎곰팡이병": {"습도": "85% 이상", "기온": "18~25℃", "특징": "시설 내 과습 시 발생"},
                "황화잎말이바이러스": {"습도": "영향 적음", "기온": "20~30℃", "특징": "담배가루이 매개"}
            },
            "causes": {
                "잎곰팡이병": ["곰팡이 포자 비산", "환기 부족", "밀식 재배"],
                "황화잎말이바이러스": ["담배가루이", "감염된 모종", "주변 잡초 서식"]
            }},
    "딸기": {"file": "strawberry_model.pth", "classes": ['딸기 (정상)', '딸기 (잿빛곰팡이병)', '딸기 (흰가루병)'],
           "risk_env": {
               "잿빛곰팡이병": {"습도": "90% 이상", "기온": "15~23℃", "특징": "저온 다습 환경"},
               "흰가루병": {"습도": "건조~다습 반복", "기온": "18~25℃", "특징": "일교차 클 때 발생"}
           },
           "causes": {
               "잿빛곰팡이병": ["수정 후 꽃잎 잔재", "과습", "병든 과실 방치"],
               "흰가루병": ["통풍 불량", "질소 과다 시비", "건조한 환경"]
           }},
    "상추": {"file": "lettuce_model.pth", "classes": ['상추 (정상)', '상추 (노균병)', '상추 (균핵병)'],
           "risk_env": {
               "상추 (노균병)": {"습도": "85% 이상", "기온": "15~23℃", "특징": "저온다습 시 급속 확산"},
               "상추 (균핵병)": {"습도": "80% 이상", "기온": "15~25℃", "특징": "연작지 토양 전염"}
           },
           "causes": {
               "상추 (노균병)": ["잎 표면 결로", "배수 불량", "밀식"],
               "상추 (균핵병)": ["토양 내 월동 균핵", "이어짓기(연작)", "시설 내 과습"]
           }},
    "오이": {"file": "cucumber_model.pth", "classes": ['오이 (정상)', '오이 (모자이크바이러스)', '오이 (녹반모자이크바이러스)'],
           "risk_env": {
               "모자이크바이러스": {"습도": "영향 적음", "기온": "20~30℃", "특징": "진딧물 매개"},
               "녹반모자이크바이러스": {"습도": "영향 적음", "기온": "22~30℃", "특징": "토양, 종자 전염"}
           },
           "causes": {
               "모자이크바이러스": ["진딧물", "주변 잡초", "전정 가위 소독 미흡"],
               "녹반모자이크바이러스": ["오염된 토양", "접목 도구", "감염 종자"]
           }},
    "포도": {"file": "grape_model.pth", "classes": ['포도 (정상)', '포도 (노균병)'],
           "risk_env": {
               "노균병": {"습도": "85% 이상", "기온": "18~25℃", "특징": "비 온 뒤 급격 확산"}
           },
           "causes": {
               "노균병": ["잦은 강우", "배수 불량", "잎 뒷면 기공 침입"]
           }}
}


# ==========================================
# [함수] 모델 로드 및 유틸리티
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
        return {"temp": res["main"]["temp"], "humidity": res["main"]["humidity"], "city": res.get("name", "Unknown")}
    except:
        return None


def get_weather_by_city(city="Seoul"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") != 200: return None
        return {"temp": res["main"]["temp"], "humidity": res["main"]["humidity"], "city": res.get("name", city)}
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


# ★ [수정됨] 함수 위치를 상단으로 이동하여 에러 해결
def generate_prescription(disease, humidity, temp):
    prescription = {
        "risk_score": 0, "risk_label": "안전", "color": "green",
        "action_plan": [], "chemical": "-", "eco_friendly": "-"
    }

    # 1. 곰팡이류 (습도 민감)
    if any(x in disease for x in ['탄저', '곰팡이', '노균', '무늬']):
        if humidity >= 80:
            prescription['risk_score'] = 90
            prescription['risk_label'] = "🚨 심각 (즉시 방제)"
            prescription['color'] = "red"
            prescription['action_plan'] = ["습도가 매우 높아(80%↑) 포자가 급속 확산 중입니다.", "병든 잎과 과실을 즉시 제거하여 소각하세요.",
                                           "하우스 측창과 환기팬을 최대로 가동하세요."]
        elif humidity >= 60:
            prescription['risk_score'] = 60
            prescription['risk_label'] = "⚠️ 주의 (예방 필요)"
            prescription['color'] = "orange"
            prescription['action_plan'] = ["습도가 높아질 조짐이 보입니다.", "예방 차원에서 친환경 제제를 살포하세요.", "밀식된 부위 잎을 정리하여 통풍을 확보하세요."]
        else:
            prescription['risk_score'] = 20
            prescription['risk_label'] = "✅ 관찰 (위험 낮음)"
            prescription['action_plan'] = ["현재 환경은 건조하여 병 확산이 억제됩니다.", "3일 간격으로 작물을 예찰하세요."]
        prescription['chemical'] = "플루아지남 수화제, 아족시스트로빈 액상"
        prescription['eco_friendly'] = "석회보르도액, 난황유 도포"

    # 2. 바이러스/해충류 (온도 민감)
    elif any(x in disease for x in ['바이러스', '모자이크', '벌레']):
        if temp >= 25:
            prescription['risk_score'] = 85
            prescription['risk_label'] = "🚨 위험 (매개충 활성)"
            prescription['color'] = "red"
            prescription['action_plan'] = [f"현재 기온({temp}도)은 매개충(진딧물 등) 활동 최적기입니다.", "황색 끈끈이 트랩을 설치하여 밀도를 낮추세요.",
                                           "주변 잡초를 제거하여 서식지를 없애세요."]
        else:
            prescription['risk_score'] = 40
            prescription['risk_label'] = "⚠️ 경계"
            prescription['color'] = "orange"
            prescription['action_plan'] = ["아직 기온이 낮아 해충 활동이 적습니다.", "초기 방제로 확산을 막으세요."]
        prescription['chemical'] = "이미다클로프리드 수화제 (진딧물용)"
        prescription['eco_friendly'] = "님 오일(Neem Oil), 고삼 추출물"

    # 3. 정상
    elif '정상' in disease:
        prescription['risk_label'] = "✨ 매우 건강"
        prescription['action_plan'] = ["현재 작물 상태가 아주 좋습니다.", "현행 관리 방법을 유지하세요."]

    return prescription


# ==========================================
# [UI] 메인 화면
# ==========================================
st.markdown('<div class="custom-header">🌿 스마트 팜</div>', unsafe_allow_html=True)
location = get_geolocation()

col_left, col_right = st.columns([1.5, 1], gap="medium")

# === [왼쪽] 진단 및 분석 ===
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
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        top_prob, top_idx = torch.max(probs, 0)

                        # 상위 2개 추출 (불확실성 계산)
                        top2 = torch.topk(probs, 2)
                        gap = (top2.values[0] - top2.values[1]).item() * 100

                        class_names = CROP_CONFIG[selected_crop]["classes"]
                        pred = class_names[top_idx]
                        conf = top_prob.item() * 100

                    # 세션에 결과 저장
                    st.session_state['last_pred'] = pred
                    st.session_state['last_conf'] = conf
                    st.session_state['confidence_gap'] = gap
                    st.session_state['top2_classes'] = [class_names[top2.indices[0]], class_names[top2.indices[1]]]
                else:
                    st.error("모델 파일이 없습니다.")

    # 결과 출력
    if 'last_pred' in st.session_state:
        pred = st.session_state['last_pred']
        conf = st.session_state['last_conf']

        # 1. 결과 카드
        html_code = f"""
        <div style="background-color: #FFFFFF; padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; border: 1px solid #e0e0e0;">
            <p style="color: #000000; font-size: 14px; margin-bottom: 5px; font-weight: bold;">분석 결과</p>
            <h2 style="color: #000000; font-weight: bold; margin: 0; margin-bottom: 10px;">{pred}</h2>
            <p style="color: #4CAF50; font-weight: bold; font-size: 16px; margin: 0;">신뢰도: {conf:.2f}%</p>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

        # 2. 신뢰성 설명
        gap = st.session_state.get('confidence_gap', 0)
        top2 = st.session_state.get('top2_classes', [])
        level = "높음" if gap >= 30 else "보통" if gap >= 15 else "낮음"

        st.markdown(f"""
        <div style="background:#f1f8e9; padding:15px; border-radius:12px; margin-top:10px; border-left:5px solid #8bc34a;">
            <b>🧠 모델 예측 신뢰성: {level}</b> (차이: {gap:.1f}%)<br>
            <span style="font-size:0.9rem;">AI는 <b>{top2[0]}</b>일 확률이 <b>{top2[1]}</b>보다 확실히 높다고 판단했습니다.</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(conf))

        # 3. 날씨 정보
        weather = None
        if location and 'coords' in location:
            weather = get_weather_by_coords(location['coords']['latitude'], location['coords']['longitude'])
            loc_label = f"{weather['city']} (내 위치)"
        else:
            weather = get_weather_by_city("Seoul")
            loc_label = "Seoul (기본)"

        if weather:
            # 세션에 날씨 저장 (오른쪽 컬럼 처방전용)
            st.session_state['temp'] = weather['temp']
            st.session_state['humid'] = weather['humidity']

            st.markdown(f"""
            <div class="weather-box">
                <strong style="color: #1565c0;">🌤️ 실시간 환경 분석 - {loc_label}</strong><br>
                <span style="color: #000000;">기온: <b>{weather['temp']}°C</b> / 습도: <b>{weather['humidity']}%</b></span>
            </div>
            """, unsafe_allow_html=True)

        # 4. 상세 원인 정보
        disease_name = pred.split("(")[-1].replace(")", "").strip()
        risk_info = CROP_CONFIG[selected_crop].get("risk_env", {}).get(disease_name)
        cause_info = CROP_CONFIG[selected_crop].get("causes", {}).get(disease_name)

        if risk_info or cause_info:
            cause_html = "".join([f"<li>{c}</li>" for c in cause_info]) if cause_info else ""
            st.markdown(f"""
            <div style="background:#fff8e1; padding:16px; border-radius:14px; border-left:6px solid #ffeb3b; margin-top:15px;">
                <b>📊 병해 취약 환경 & 원인</b><br>
                <ul style="margin-left:20px; font-size:0.9rem;">{cause_html}</ul>
                <div style="font-size:0.85rem; color:#555;">출처: 농촌진흥청 농사로</div>
            </div>
            """, unsafe_allow_html=True)

# === [오른쪽] 뉴스 & 처방전 ===
with col_right:
    st.markdown('<div class="section-title">📰 관련 농업 뉴스</div>', unsafe_allow_html=True)
    keyword = st.session_state.get('last_pred', f"{selected_crop} 병해충").split('(')[0] + " 방제"
    news_items = get_naver_news(keyword)

    with st.container(height=300, border=False):
        if news_items:
            for item in news_items:
                title = item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
                link = item['link']
                st.markdown(
                    f"<a href='{link}' target='_blank' class='news-item'><span class='news-title'>📄 {title}</span></a>",
                    unsafe_allow_html=True)
        else:
            st.info("관련 뉴스를 찾을 수 없습니다.")

    # ---------------------------------------------------------
    # AI 스마트 처방전 (DSS)
    # ---------------------------------------------------------
    st.write("---")
    st.subheader("📋 AI 스마트 방제 처방전")

    if 'last_pred' in st.session_state and 'temp' in st.session_state:
        # 처방전 생성 실행
        rx = generate_prescription(st.session_state['last_pred'], st.session_state['humid'], st.session_state['temp'])

        # 1. 위험도 게이지
        st.write(f"**전염 확산 위험도: {rx['risk_label']}**")
        st.progress(rx['risk_score'])

        # 2. 행동 요령 & 약제 추천
        st.info("**🛠️ 환경 제어 및 행동 요령**")
        for action in rx['action_plan']:
            st.write(f"- {action}")

        if rx['chemical'] != "-":
            st.success("**💊 추천 약제 및 농법**")
            st.write(f"**[화학]** {rx['chemical']}")
            st.write(f"**[친환경]** {rx['eco_friendly']}")
            st.caption("※ 농약 안전 사용 기준을 준수하세요.")

        # 3. 종합 판단 (DSS 핵심 멘트)
        bg_color = "#ffebee" if rx['risk_score'] >= 50 else "#e8f5e9"
        border_color = "red" if rx['risk_score'] >= 50 else "green"
        msg = "즉각적인 방제가 필요합니다." if rx['risk_score'] >= 50 else "현재 환경은 안전하나 예찰이 필요합니다."

        st.markdown(f"""
        <div style="margin-top:10px; padding:15px; background-color:{bg_color}; border-left: 5px solid {border_color}; border-radius:5px;">
            <b>🤖 AI 종합 판단 Report</b><br>
            현재 기상(습도 {st.session_state['humid']}%)과 병해 특성을 종합 분석한 결과, 
            <b>{msg}</b>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 왼쪽에서 작물 진단을 먼저 완료해주세요.\nAI가 진단 결과와 날씨를 분석하여 처방전을 발행합니다.")