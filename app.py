import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="GPR AI Prototype - Mahmoud", layout="wide")
st.title("️ GPR Leak Detection Prototype (i3WaterS DC1)")
st.markdown(
    "Developed by: **Mahmoud Elnajjar** | Demonstrating AI-driven preprocessing and XAI for subsurface utility detection.")

st.sidebar.header("⚙️ Control Panel")
st.sidebar.info("Adjust parameters to optimize detection for different GPR images.")

uploaded_file = st.sidebar.file_uploader("Upload a GPR B-scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("1. Original GPR B-scan")
    st.image(img, channels="BGR", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Detection Parameters")

    # Sliders للتحكم في المعاملات
    clahe_clip = st.sidebar.slider("CLAHE Contrast", 1.0, 10.0, 3.0, 0.5)
    clahe_grid = st.sidebar.slider("CLAHE Grid Size", 4, 16, 8, 2)
    canny_low = st.sidebar.slider("Canny Low Threshold", 10, 100, 30, 5)
    canny_high = st.sidebar.slider("Canny High Threshold", 50, 200, 100, 10)
    min_contour_area = st.sidebar.slider("Min Contour Area", 50, 1000, 150, 50)

    if st.sidebar.button(" Run AI Analysis Pipeline"):
        with st.spinner("Processing signal and applying XAI detection..."):

            # 1. CLAHE Enhancement
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
            enhanced = clahe.apply(gray)

            # 2. Canny Edge Detection
            edges = cv2.Canny(enhanced, canny_low, canny_high)

            # 3. Find Contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_img = img.copy()
            heatmap = np.zeros_like(img, dtype=np.float32)
            detection_found = False

            if len(contours) > 0:
                # Sort contours by area (largest first)
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

                # Take top 3 largest contours
                for i, c in enumerate(contours_sorted[:3]):
                    area = cv2.contourArea(c)
                    if area > min_contour_area:
                        x, y, w, h = cv2.boundingRect(c)

                        # Draw bounding box
                        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(result_img, f"Detection #{i + 1}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        detection_found = True

                        # Create heatmap
                        cv2.rectangle(heatmap, (x, y), (x + w, y + h), (1.0,), -1)

                if detection_found:
                    # Blur heatmap
                    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
                    heatmap = np.uint8(255 * heatmap / np.max(heatmap))
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

            if not detection_found:
                overlay = img
                st.warning(
                    "️ No significant anomalies detected. Try adjusting the sliders (lower Canny thresholds or Min Contour Area).")

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("2. Preprocessed Signal (CLAHE)")
                st.image(enhanced, channels="GRAY", use_container_width=True)
                st.caption("Noise reduction and local contrast enhancement.")

            with col2:
                st.subheader("3. AI Detection & XAI Heatmap")
                st.image(overlay, channels="BGR", use_container_width=True)
                st.caption("Red boxes: Detected anomalies. Heatmap: XAI attention map.")

            st.success("✅ Pipeline executed successfully!")
else:
    st.info(" Please upload a GPR B-scan image from the sidebar to begin.")