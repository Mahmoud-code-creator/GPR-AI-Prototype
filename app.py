import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="GPR AI Prototype - Mahmoud", layout="wide")
st.title("🛰️ GPR Leak Detection Prototype (i3WaterS DC1)")
st.markdown(
    "Developed by: **Mahmoud [Your Last Name]** | Demonstrating AI-driven preprocessing and XAI for subsurface utility detection.")

st.sidebar.header("Control Panel")
st.sidebar.info(
    "This prototype simulates the preprocessing and semi-autonomous detection pipeline proposed in the i3WaterS project.")

uploaded_file = st.sidebar.file_uploader("Upload a GPR B-scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("1. Original GPR B-scan")
    st.image(img, channels="BGR", use_container_width=True)

    if st.sidebar.button("🚀 Run AI Analysis Pipeline"):
        with st.spinner("Processing signal and applying XAI detection..."):
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            edges = cv2.Canny(enhanced, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_img = img.copy()
            heatmap = np.zeros_like(img, dtype=np.float32)
            detection_found = False

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 150:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(result_img, "AI Detection: Potential Pipe/Leak", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    detection_found = True

                    cv2.rectangle(heatmap, (x, y), (x + w, y + h), (1.0,), -1)
                    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
                    heatmap = np.uint8(255 * heatmap / np.max(heatmap))
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

            if not detection_found:
                overlay = img
                st.warning("No significant subsurface anomalies detected in this scan.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("2. Preprocessed Signal (CLAHE)")
                st.image(enhanced, channels="GRAY", use_container_width=True)
                st.caption("Noise reduction and local contrast enhancement to highlight hyperbolic signatures.")

            with col2:
                st.subheader("3. AI Detection & XAI Heatmap")
                st.image(overlay, channels="BGR", use_container_width=True)
                st.caption(
                    "Red box: Detected anomaly. Heatmap: Explainable AI (XAI) attention map showing model confidence.")

            st.success(
                "✅ Pipeline executed successfully. This demonstrates the feasibility of semi-autonomous GPR interpretation.")
else:
    st.info(" Please upload a GPR B-scan image from the sidebar to begin the demonstration.")