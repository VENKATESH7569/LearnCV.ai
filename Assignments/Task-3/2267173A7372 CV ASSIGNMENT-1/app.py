
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="Image Processing & Analysis Toolkit", layout="wide")

def load_image(uploaded_file):
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if opencv_image is not None:
            return cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return None

def convert_to_bytes(image, format_type="PNG", quality=95):
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    if format_type.upper() == "JPG" or format_type.upper() == "JPEG":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode('.jpg', image_bgr, encode_param)
    elif format_type.upper() == "PNG":
        success, encoded_img = cv2.imencode('.png', image_bgr)
    else:
        success, encoded_img = cv2.imencode('.bmp', image_bgr)

    if success:
        return encoded_img.tobytes()
    return None

def get_image_info(image):
    if image is None:
        return {}

    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    return {
        "Height": height,
        "Width": width,
        "Channels": channels,
        "Shape": f"{height} x {width} x {channels}" if channels > 1 else f"{height} x {width}",
        "Data Type": str(image.dtype)
    }

st.title("🖼️ Image Processing & Analysis Toolkit")
st.markdown("Upload an image and apply various computer vision operations")

uploaded_file = st.sidebar.file_uploader(
    "📁 Open Image", 
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="Upload an image file to begin processing"
)

original_image = load_image(uploaded_file)
processed_image = original_image.copy() if original_image is not None else None

if original_image is not None:
    st.sidebar.markdown("---")

    operation_category = st.sidebar.selectbox(
        "🔧 Select Operation Category",
        [
            "Image Info",
            "Color Conversions", 
            "Transformations",
            "Filtering & Morphology",
            "Enhancement",
            "Edge Detection",
            "Compression"
        ]
    )

    if operation_category == "Image Info":
        st.sidebar.markdown("### 📊 Image Information")
        info = get_image_info(original_image)
        for key, value in info.items():
            st.sidebar.text(f"{key}: {value}")

    elif operation_category == "Color Conversions":
        st.sidebar.markdown("### 🎨 Color Space Conversions")
        conversion_type = st.sidebar.selectbox(
            "Select Conversion",
            ["Original", "RGB to BGR", "RGB to HSV", "RGB to YCbCr", "RGB to Grayscale"]
        )

        if conversion_type == "RGB to BGR":
            processed_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        elif conversion_type == "RGB to HSV":
            processed_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
        elif conversion_type == "RGB to YCbCr":
            processed_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2YCrCb)
        elif conversion_type == "RGB to Grayscale":
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            processed_image = original_image.copy()

    elif operation_category == "Transformations":
        st.sidebar.markdown("### 📐 Geometric Transformations")
        transform_type = st.sidebar.selectbox(
            "Select Transformation",
            ["None", "Rotation", "Scaling", "Translation", "Affine Transform"]
        )

        if transform_type == "Rotation":
            angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
            height, width = original_image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed_image = cv2.warpAffine(original_image, rotation_matrix, (width, height))

        elif transform_type == "Scaling":
            scale_factor = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)
            processed_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor)

        elif transform_type == "Translation":
            tx = st.sidebar.slider("Translate X", -200, 200, 0)
            ty = st.sidebar.slider("Translate Y", -200, 200, 0)
            height, width = original_image.shape[:2]
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            processed_image = cv2.warpAffine(original_image, translation_matrix, (width, height))

        elif transform_type == "Affine Transform":
            shear_x = st.sidebar.slider("Shear X", -0.5, 0.5, 0.0, 0.1)
            shear_y = st.sidebar.slider("Shear Y", -0.5, 0.5, 0.0, 0.1)
            height, width = original_image.shape[:2]

            pts1 = np.float32([[0, 0], [width, 0], [0, height]])
            pts2 = np.float32([[shear_x * height, shear_y * width], 
                              [width + shear_x * height, shear_y * width], 
                              [shear_x * height, height + shear_y * width]])

            affine_matrix = cv2.getAffineTransform(pts1, pts2)
            processed_image = cv2.warpAffine(original_image, affine_matrix, (width, height))
        else:
            processed_image = original_image.copy()

    elif operation_category == "Filtering & Morphology":
        st.sidebar.markdown("### 🔍 Filters & Morphological Operations")
        filter_type = st.sidebar.selectbox(
            "Select Filter/Operation",
            ["None", "Gaussian Blur", "Mean Filter", "Median Filter", "Sobel Filter", "Laplacian Filter", "Dilation", "Erosion", "Opening", "Closing"]
        )

        if filter_type in ["Gaussian Blur", "Mean Filter", "Median Filter"]:
            kernel_size = st.sidebar.slider("Kernel Size", 3, 31, 5, step=2)

            if filter_type == "Gaussian Blur":
                processed_image = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0)
            elif filter_type == "Mean Filter":
                processed_image = cv2.blur(original_image, (kernel_size, kernel_size))
            else:
                processed_image = cv2.medianBlur(original_image, kernel_size)

        elif filter_type in ["Sobel Filter", "Laplacian Filter"]:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

            if filter_type == "Sobel Filter":
                direction = st.sidebar.selectbox("Direction", ["X", "Y", "Both"])
                if direction == "X":
                    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                elif direction == "Y":
                    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                else:
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    sobel = np.sqrt(sobelx**2 + sobely**2)

                sobel = np.uint8(np.absolute(sobel))
                processed_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
            else:
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian = np.uint8(np.absolute(laplacian))
                processed_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)

        elif filter_type in ["Dilation", "Erosion", "Opening", "Closing"]:
            kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            if filter_type == "Dilation":
                processed_image = cv2.dilate(original_image, kernel, iterations=1)
            elif filter_type == "Erosion":
                processed_image = cv2.erode(original_image, kernel, iterations=1)
            elif filter_type == "Opening":
                processed_image = cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel)
            else:
                processed_image = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel)
        else:
            processed_image = original_image.copy()

    elif operation_category == "Enhancement":
        st.sidebar.markdown("### ✨ Image Enhancement")
        enhancement_type = st.sidebar.selectbox(
            "Select Enhancement",
            ["None", "Histogram Equalization", "CLAHE", "Contrast Stretching", "Sharpening"]
        )

        if enhancement_type == "Histogram Equalization":
            if len(original_image.shape) == 3:
                yuv = cv2.cvtColor(original_image, cv2.COLOR_RGB2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                processed_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            else:
                processed_image = cv2.equalizeHist(original_image)

        elif enhancement_type == "CLAHE":
            clip_limit = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0, 0.1)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))

            if len(original_image.shape) == 3:
                lab = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                processed_image = clahe.apply(original_image)

        elif enhancement_type == "Contrast Stretching":
            alpha = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
            beta = st.sidebar.slider("Brightness", -100, 100, 0)
            processed_image = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)

        elif enhancement_type == "Sharpening":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed_image = cv2.filter2D(original_image, -1, kernel)
        else:
            processed_image = original_image.copy()

    elif operation_category == "Edge Detection":
        st.sidebar.markdown("### 🔲 Edge Detection")
        edge_method = st.sidebar.selectbox(
            "Select Method",
            ["None", "Canny", "Sobel", "Laplacian", "Prewitt"]
        )

        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        if edge_method == "Canny":
            threshold1 = st.sidebar.slider("Lower Threshold", 0, 300, 50)
            threshold2 = st.sidebar.slider("Upper Threshold", 0, 300, 150)
            edges = cv2.Canny(gray, threshold1, threshold2)
            processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        elif edge_method == "Sobel":
            direction = st.sidebar.selectbox("Direction", ["X", "Y", "Both"])
            if direction == "X":
                edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            elif direction == "Y":
                edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            else:
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2)

            edges = np.uint8(np.absolute(edges))
            processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        elif edge_method == "Laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
            processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        else:
            processed_image = original_image.copy()

    elif operation_category == "Compression":
        st.sidebar.markdown("### 💾 Image Compression & Save")
        save_format = st.sidebar.selectbox("Save Format", ["PNG", "JPG", "BMP"])

        if save_format == "JPG":
            quality = st.sidebar.slider("JPEG Quality", 10, 100, 95)
        else:
            quality = 95

        processed_image = original_image.copy()

        st.sidebar.markdown("**File Size Comparison:**")
        for fmt in ["PNG", "JPG", "BMP"]:
            img_bytes = convert_to_bytes(processed_image, fmt, quality if fmt=="JPG" else 95)
            if img_bytes:
                size_kb = len(img_bytes) / 1024
                st.sidebar.text(f"{fmt}: {size_kb:.1f} KB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(original_image, use_column_width=True)

        original_info = get_image_info(original_image)
        st.markdown("**Original Image Info:**")
        for key, value in original_info.items():
            st.text(f"{key}: {value}")

    with col2:
        st.subheader("🔄 Processed Image")
        if processed_image is not None:
            st.image(processed_image, use_column_width=True)

            processed_info = get_image_info(processed_image)
            st.markdown("**Processed Image Info:**")
            for key, value in processed_info.items():
                st.text(f"{key}: {value}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save as PNG"):
            img_bytes = convert_to_bytes(processed_image, "PNG")
            if img_bytes:
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="processed_image.png",
                    mime="image/png"
                )

    with col2:
        if st.button("💾 Save as JPG"):
            img_bytes = convert_to_bytes(processed_image, "JPG", 95)
            if img_bytes:
                st.download_button(
                    label="Download JPG",
                    data=img_bytes,
                    file_name="processed_image.jpg",
                    mime="image/jpeg"
                )

    with col3:
        if st.button("💾 Save as BMP"):
            img_bytes = convert_to_bytes(processed_image, "BMP")
            if img_bytes:
                st.download_button(
                    label="Download BMP",
                    data=img_bytes,
                    file_name="processed_image.bmp",
                    mime="image/bmp"
                )

else:
    st.info("👆 Please upload an image from the sidebar to start processing")

    st.markdown("""
    ## 🚀 Features Available:

    ### 📊 Image Information
    - Display image dimensions, channels, and data type

    ### 🎨 Color Conversions
    - RGB ↔ BGR, HSV, YCbCr, Grayscale

    ### 📐 Transformations  
    - Rotation, Scaling, Translation, Affine Transform

    ### 🔍 Filtering & Morphology
    - Gaussian, Mean, Median filters
    - Sobel, Laplacian edge filters  
    - Dilation, Erosion, Opening, Closing

    ### ✨ Enhancement
    - Histogram Equalization, CLAHE
    - Contrast Stretching, Sharpening

    ### 🔲 Edge Detection
    - Canny, Sobel, Laplacian edge detection

    ### 💾 Compression
    - Save in PNG, JPG, BMP formats
    - File size comparison
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("**📚 Image Processing & Analysis Toolkit**")
st.sidebar.markdown("*Built with Streamlit & OpenCV*")
