# app.py

# ==============================================================================
# 1. 导入必要的库 (Import Libraries)
# ==============================================================================
import os

# 下面这行代码可以屏蔽掉 TensorFlow 的一些信息级日志，让终端输出更干净
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ==============================================================================
# 2. 设置网页布局和标题 (Setup Webpage Layout and Titles)
# ==============================================================================
st.set_page_config(
    page_title="肺炎X光片智能检测",
    page_icon="🩺",
    layout="centered"
)
st.title("肺炎X光片智能检测系统 🩺")
st.write("""
这是一个基于深度学习的Web应用，能够根据您上传的胸部X光片，辅助判断是否存在肺炎迹象。
**请注意：** 本应用结果仅供参考，不能替代专业医师的诊断。
""")
st.markdown("---")  # 添加一条分割线


# ==============================================================================
# 3. 加载训练好的模型 (Load the Trained Model)
# ==============================================================================
# @st.cache_resource 装饰器会缓存模型加载的结果，避免每次操作都重复加载
@st.cache_resource
def load_my_model():
    """加载我们训练和优化好的模型"""
    model = tf.keras.models.load_model('pneumonia_detection_model_optimized.keras')
    return model


model = load_my_model()

# ==============================================================================
# 4. 创建文件上传组件 (Create the File Uploader)
# ==============================================================================
uploaded_file = st.file_uploader(
    "请在此处上传一张胸部X光片图片...",
    type=["jpeg", "jpg", "png"]  # 限制可接受的文件类型
)

# ==============================================================================
# 5. 核心预测逻辑 (Core Prediction Logic)
# ==============================================================================
if uploaded_file is not None:
    # 使用 PIL 打开用户上传的图片
    image = Image.open(uploaded_file)

    # 在网页上显示用户上传的图片
    st.image(image, caption='您上传的X光片', use_column_width=True)

    # 使用 st.spinner 来显示一个加载提示，提升用户体验
    # 所有耗时的操作都应放在 with 代码块内部
    with st.spinner('模型正在全力分析中，请稍候...'):
        # a. 图片预处理：必须和训练时的预处理步骤完全一致
        img_resized = image.resize((150, 150))
        img_rgb = img_resized.convert('RGB')
        img_array = np.array(img_rgb)
        img_normalized = img_array / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # b. 进行预测 (这是最耗时的一步)
        prediction = model.predict(img_batch)

    # c. 显示预测结果 (预测完成后，spinner会自动消失)
    st.subheader("分析结果：")

    # Sigmoid 输出的是一个0到1之间的概率值，我们以0.5为阈值
    if prediction[0][0] > 0.5:
        confidence = prediction[0][0] * 100
        st.error(f"**高风险：** 模型判断为 **肺炎** 的可能性为: **{confidence:.2f}%**")
        st.warning("建议尽快咨询专业医师进行进一步检查。")
    else:
        confidence = (1 - prediction[0][0]) * 100
        st.success(f"**低风险：** 模型判断为 **正常** 的可能性为: **{confidence:.2f}%**")
        st.info("这表明未发现明显的肺炎迹象，但仍请以专业医师诊断为准。")