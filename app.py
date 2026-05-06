import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation,
    Concatenate, GlobalAveragePooling2D, Reshape, UpSampling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from huggingface_hub import hf_hub_download # Thêm thư viện tải model từ HF

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="MSC Segmentation App", layout="wide")

IMAGE_SIZE = 512

# ==========================================
# 1. HÀM XÂY DỰNG KIẾN TRÚC MÔ HÌNH (Từ Code của bạn)
# ==========================================
def conv_bn_relu(x, filters, k=3, dilation=1, name=None):
    x = Conv2D(filters, k, padding="same", use_bias=False, dilation_rate=dilation,
               name=None if name is None else name+"_conv")(x)
    x = BatchNormalization(name=None if name is None else name+"_bn")(x)
    x = Activation("relu", name=None if name is None else name+"_relu")(x)
    return x

def conv_block(x, filters, name):
    x = conv_bn_relu(x, filters, 3, name=name+"_1")
    x = conv_bn_relu(x, filters, 3, name=name+"_2")
    return x

def aspp(x, filters=256, name="aspp"):
    # 1x1
    b0 = conv_bn_relu(x, filters, k=1, name=name+"_b0")

    # atrous conv (multi-scale)
    b1 = conv_bn_relu(x, filters, k=3, dilation=6,  name=name+"_b1")
    b2 = conv_bn_relu(x, filters, k=3, dilation=12, name=name+"_b2")
    b3 = conv_bn_relu(x, filters, k=3, dilation=18, name=name+"_b3")

    # image pooling
    c = x.shape[-1]
    pool = GlobalAveragePooling2D(name=name+"_gap")(x)
    pool = Reshape((1,1,c), name=name+"_reshape")(pool)
    pool = conv_bn_relu(pool, filters, k=1, name=name+"_poolproj")

    factor = IMAGE_SIZE // 32
    pool = UpSampling2D(size=(factor, factor), interpolation="bilinear", name=name+"_poolup")(pool)

    y = Concatenate(name=name+"_cat")([b0, b1, b2, b3, pool])
    y = conv_bn_relu(y, filters, k=1, name=name+"_proj")
    return y

def build_resunet_aspp(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), freeze_encoder=True):
    inputs = Input(input_shape)

    x = tf.keras.layers.Lambda(lambda t: t * 255.0, name="to_255")(inputs)
    x = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
                               name="resnet_preprocess")(x)

    encoder = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    encoder.trainable = not freeze_encoder

    s1 = encoder.get_layer("conv1_relu").output
    s2 = encoder.get_layer("conv2_block3_out").output
    s3 = encoder.get_layer("conv3_block4_out").output
    s4 = encoder.get_layer("conv4_block6_out").output
    b  = encoder.get_layer("conv5_block3_out").output

    b = aspp(b, filters=256, name="aspp")

    d4 = Conv2DTranspose(256, 3, strides=2, padding="same", name="up4")(b)
    d4 = Concatenate(name="cat4")([d4, s4])
    d4 = conv_block(d4, 256, name="dec4")

    d3 = Conv2DTranspose(128, 3, strides=2, padding="same", name="up3")(d4)
    d3 = Concatenate(name="cat3")([d3, s3])
    d3 = conv_block(d3, 128, name="dec3")

    d2 = Conv2DTranspose(64, 3, strides=2, padding="same", name="up2")(d3)
    d2 = Concatenate(name="cat2")([d2, s2])
    d2 = conv_block(d2, 64, name="dec2")

    d1 = Conv2DTranspose(32, 3, strides=2, padding="same", name="up1")(d2)
    d1 = Concatenate(name="cat1")([d1, s1])
    d1 = conv_block(d1, 32, name="dec1")

    d0 = Conv2DTranspose(16, 3, strides=2, padding="same", name="up0")(d1)
    d0 = conv_block(d0, 16, name="dec0")

    outputs = Conv2D(1, 1, activation="sigmoid", name="mask")(d0)
    model = Model(inputs, outputs, name="ResUNet50_ASPP")
    return model

# ==========================================
# 2. HÀM TẢI MÔ HÌNH VÀ TRỌNG SỐ TỪ HUGGING FACE
# ==========================================
@st.cache_resource(show_spinner="Đang tải Model từ Hugging Face (Chỉ tải lần đầu tiên)...")
def load_model():
    # TODO: THAY ĐỔI DÒNG NÀY THÀNH REPO CỦA BẠN
    REPO_ID = "ducthang1703/msc-segmentation" 
    FILENAME = "resunet50_aspp_best.weights.h5"
    
    # Tải file weights từ Hugging Face về (sẽ được cache lại trên server Streamlit)
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # Khởi tạo kiến trúc và nạp weights
    model = build_resunet_aspp(freeze_encoder=True) 
    model.load_weights(weights_path)
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.warning("Không thể tải model từ Hugging Face. Vui lòng kiểm tra lại REPO_ID hoặc kết nối mạng.")
    st.error(f"Chi tiết lỗi: {e}")
    model_loaded = False

# ==========================================
# 3. GIAO DIỆN CHÍNH (UI)
# ==========================================
st.title("🔬 Ứng dụng Phân đoạn ảnh MSC")
st.write("Sử dụng mô hình **ResUNet50 + ASPP** để tự động phân vùng đối tượng từ ảnh tải lên.")

threshold = st.sidebar.slider("Ngưỡng phân đoạn (Threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
st.sidebar.write("*Mẹo: Dựa vào Notebook của bạn, bạn có thể chỉnh threshold về `best_dice_threshold` để tối ưu.*")

uploaded_file = st.file_uploader("Chọn một tệp hình ảnh (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    if st.button("Phân đoạn ảnh ngay"):
        with st.spinner('Mô hình đang xử lý... vui lòng đợi!'):
            resized_img = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            input_tensor = resized_img.astype(np.float32) / 255.0 
            input_tensor = np.expand_dims(input_tensor, axis=0) 
            
            pred_probs = model.predict(input_tensor)[0] 
            
            pred_mask = (pred_probs > threshold).astype(np.uint8) * 255
            pred_mask_squeeze = np.squeeze(pred_mask, axis=-1)
            
            original_h, original_w = image_np.shape[:2]
            final_mask = cv2.resize(pred_mask_squeeze, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            colored_mask = np.zeros_like(image_np)
            colored_mask[final_mask == 255] = [0, 255, 0]
            overlay_img = cv2.addWeighted(image_np, 0.7, colored_mask, 0.3, 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Ảnh Gốc")
                st.image(image_np, use_container_width=True)
                
            with col2:
                st.subheader("Mask Dự Đoán")
                st.image(final_mask, use_container_width=True)
                
            with col3:
                st.subheader("Lớp Phủ (Overlay)")
                st.image(overlay_img, use_container_width=True)
                
        st.balloons()