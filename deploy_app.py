import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
import requests
import urllib.parse

# ==========================================
# [설정] API 키 (main.py에서 가져옴)
# ==========================================
# 실제 배포 시에는 Streamlit Secrets 기능을 쓰는 것이 보안상 좋습니다.
NAVER_CLIENT_ID = "2wR8x89ky2cwkwIspEyj"
NAVER_CLIENT_SECRET = "uw_h22JCJR"
WEATHER_API_KEY = "f9408d1bd75131dddadd813aaa4809b4"

# ==========================================
# [설정] 6개 작물별 모델 및 클래스 정의
# ==========================================
CROP_CONFIG = {
    "고추 (Pepper)": {
        "model_file": "pepper_model.pth",
        "classes": ['고추 (정상)', '고추 (마일드모틀바이러스)', '고추 (점무늬병)']
    },
    "딸기 (Strawberry)": {
        "model_file": "strawberry_model.pth",
        "classes": ['딸기 (정상)', '딸기 (잿빛곰팡이병)', '딸기 (흰가루병)']
    },
    "상추 (Lettuce)": {
        "model_file": "lettuce_model.pth",
        "classes": ['상추 (정상)', '상추 (노균병)', '상추 (균핵병)']
    },
    "오이 (Cucumber)": {
        "model_file": "cucumber_model.pth",
        "classes": ['오이 (정상)', '오이 (모자이크바이러스)', '오이 (녹반모자이크바이러스)']
    },
    "토마토 (Tomato)": {
        "model_file": "tomato_model.pth",
        "classes": ['토마토 (정상)', '토마토 (잎곰팡이병)', '토마토 (황화잎말이바이러스)']
    },
    "포도 (Grape)": {
        "model_file": "grape_model.pth",
        "classes": ['포도 (정상)', '포도 (노균병)']
    }
}


# ==========================================
# [함수] 기능 구현 (모델, 날씨, 뉴스)
# ==========================================

@st.cache_resource
def load_model_for_crop(crop_name):
    """선택한 작물에 맞는 모델 로드"""
    config = CROP_CONFIG[crop_name]
    model_path = config["model_file"]
    num_classes = len(config["classes"])

    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    if os.path.exists(model_path):
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model.eval()
        return model
    else:
        return None


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def get_weather_info(city_name="Seoul"):
    """OpenWeatherMap API 호출"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        if res.get("cod") != 200:
            return None
        return {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "desc": res["weather"][0]["description"],
            "city": res.get("name", city_name)
        }
    except:
        return None


def get_naver_news(keyword):
    """네이버 뉴스 검색 API 호출"""
    try:
        encText = urllib.parse.quote(keyword)
        url = "https://openapi.naver.com/v1/search/news?query=" + encText + "&display=3&sort=sim"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['items']
        else:
            return []
    except:
        return []


# ==========================================
# [UI] 화면 구성
# ==========================================
st.set_page_config(page_title="농작물 병해 진단 플랫폼", page_icon="🌿")

st.title("🌿 농작물 병해 진단 통합 플랫폼")
st.markdown("---")

# 사이드바: 설정
with st.sidebar:
    st.header("⚙️ 환경 설정")
    selected_crop = st.selectbox("진단할 작물 선택", list(CROP_CONFIG.keys()))
    city_name = st.text_input("현재 지역 (영문)", value="Seoul")

# 메인 화면
st.subheader(f"1️⃣ {selected_crop} 사진 업로드")
uploaded_file = st.file_uploader("사진을 선택하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    if st.button("🔍 병해 진단 및 환경 분석 시작"):

        # 1. 병해 진단
        with st.spinner("AI가 작물을 분석하고 있습니다..."):
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

                st.success("✅ 분석 완료!")
                st.metric(label="진단 결과", value=predicted_class)
                st.progress(int(confidence))
                st.caption(f"신뢰도: {confidence:.2f}%")

                st.markdown("---")

                # 2. 날씨 정보 (2단 컬럼)
                st.subheader("2️⃣ 실시간 재배 환경 분석")
                weather = get_weather_info(city_name)

                if weather:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("지역", weather['city'])
                    col2.metric("기온", f"{weather['temp']}°C")
                    col3.metric("습도", f"{weather['humidity']}%")

                    # 간단한 조언 로직
                    if weather['humidity'] > 70:
                        st.warning("습도가 높습니다! 곰팡이병 예방을 위해 환기가 필요합니다.")
                    else:
                        st.info("현재 습도는 적정 수준입니다.")
                else:
                    st.error("날씨 정보를 불러올 수 없습니다. 지역명을 확인하세요.")

                st.markdown("---")

                # 3. 관련 뉴스 (진단된 병명으로 검색)
                st.subheader(f"3️⃣ '{predicted_class}' 관련 최신 방제 뉴스")

                # 검색 키워드 정제 (괄호 제거 등)
                search_keyword = predicted_class.split('(')[0] + " " + predicted_class.split('(')[-1].replace(')', '')
                if "정상" in search_keyword:
                    search_keyword = f"{selected_crop.split()[0]} 재배 기술"

                news_items = get_naver_news(search_keyword)

                if news_items:
                    for item in news_items:
                        title = item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"')
                        link = item['link']
                        st.markdown(f"- [{title}]({link})")
                else:
                    st.info("관련된 최신 뉴스가 없습니다.")

            else:
                st.error("모델 파일을 찾을 수 없습니다.")