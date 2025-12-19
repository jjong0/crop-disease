import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

# ==========================================
# [설정] 6개 작물별 모델 및 클래스 정의
# ==========================================
# 사용자가 제공한 정렬 순서("10"이 "9"보다 먼저 오는 문자열 정렬)에 맞춤
CROP_CONFIG = {
    "고추 (Pepper)": {
        "model_file": "pepper_model.pth",
        "classes": ['고추 (정상)', '고추 (마일드모틀바이러스)', '고추 (점무늬병)']  # 2_0, 2_3, 2_4
    },
    "딸기 (Strawberry)": {
        "model_file": "strawberry_model.pth",
        "classes": ['딸기 (정상)', '딸기 (잿빛곰팡이병)', '딸기 (흰가루병)']  # 4_0, 4_7, 4_8
    },
    "상추 (Lettuce)": {
        "model_file": "lettuce_model.pth",
        "classes": ['상추 (정상)', '상추 (노균병)', '상추 (균핵병)']  # 5_0, 5_10, 5_9 (순서 중요!)
    },
    "오이 (Cucumber)": {
        "model_file": "cucumber_model.pth",
        "classes": ['오이 (정상)', '오이 (모자이크바이러스)', '오이 (녹반모자이크바이러스)']  # 8_0, 8_15, 8_8
    },
    "토마토 (Tomato)": {
        "model_file": "tomato_model.pth",
        "classes": ['토마토 (정상)', '토마토 (잎곰팡이병)', '토마토 (황화잎말이바이러스)']  # 11_0, 11_18, 11_19
    },
    "포도 (Grape)": {
        "model_file": "grape_model.pth",
        "classes": ['포도 (정상)', '포도 (노균병)']  # 12_0, 12_20
    }
}


# ==========================================
# [함수] 모델 로드 및 전처리
# ==========================================
@st.cache_resource
def load_model_for_crop(crop_name):
    """선택한 작물에 맞는 모델을 로드합니다."""
    config = CROP_CONFIG[crop_name]
    model_path = config["model_file"]
    num_classes = len(config["classes"])

    # 1. 모델 구조 생성 (MobileNetV3 Small)
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    # 2. 가중치 로드 (CPU 모드)
    if os.path.exists(model_path):
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        model.eval()
        return model
    else:
        st.error(f"모델 파일({model_path})을 찾을 수 없습니다. GitHub에 파일을 업로드했는지 확인하세요.")
        return None


def preprocess_image(image):
    """이미지 전처리 (224x224, 정규화)"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# ==========================================
# [UI] 화면 구성
# ==========================================
st.set_page_config(page_title="농작물 병해 진단 통합 플랫폼", page_icon="🌿")

st.title("🌿 농작물 병해 진단 통합 플랫폼")
st.markdown("---")

# 1. 작물 선택하기
st.subheader("1️⃣ 진단할 작물을 선택하세요")
selected_crop = st.selectbox("작물 목록", list(CROP_CONFIG.keys()))

# 2. 사진 업로드
st.subheader("2️⃣ 사진을 업로드하세요")
uploaded_file = st.file_uploader(f"{selected_crop} 사진 선택", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_column_width=True)

    # 진단 버튼
    if st.button("병해 진단 시작"):
        with st.spinner(f"{selected_crop} 전용 AI 모델을 불러오는 중..."):

            # 모델 로드
            model = load_model_for_crop(selected_crop)

            if model:
                try:
                    # 추론 실행
                    input_tensor = preprocess_image(image)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                        # 결과 해석
                        top_prob, top_idx = torch.max(probabilities, 0)
                        class_names = CROP_CONFIG[selected_crop]["classes"]
                        predicted_class = class_names[top_idx]
                        confidence = top_prob.item() * 100

                    # 결과 출력
                    st.success("✅ 진단이 완료되었습니다!")
                    st.metric(label="진단 결과", value=predicted_class)
                    st.write(f"신뢰도(확률): **{confidence:.2f}%**")

                    # 확률 막대 그래프
                    st.progress(int(confidence))

                    # (선택) 추가 조언
                    if "정상" in predicted_class:
                        st.info("작물이 건강해 보입니다! 주기적인 물 주기와 환기를 잊지 마세요.")
                    else:
                        st.warning("병해가 의심됩니다. 가까운 농업기술센터나 전문가에게 상담을 권장합니다.")

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")