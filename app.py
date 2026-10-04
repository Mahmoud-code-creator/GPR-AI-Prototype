import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="GPR AI Prototype - Mahmoud", layout="wide")
st.title("🛰️ GPR Leak Detection Prototype (i3WaterS DC1)")
st.markdown(
    "Developed by: **Mahmoud Elnajjar** | Demonstrating AI-driven preprocessing and XAI for subsurface utility detection.")

st.sidebar.header("⚙️ Detection Parameters")
st.sidebar.info("Adjust parameters to optimize hyperbola detection.")

uploaded_file = st.sidebar.file_uploader("Upload a GPR B-scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("1. Original GPR B-scan")
    st.image(img, channels="BGR", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Hyperbola Detection Settings")

    # Sliders
    clahe_clip = st.sidebar.slider("CLAHE Contrast", 1.0, 10.0, 4.0, 0.5)
    blur_kernel = st.sidebar.slider("Gaussian Blur", 3, 15, 5, 2)
    threshold_val = st.sidebar.slider("Threshold Value", 50, 200, 120, 10)
    min_area = st.sidebar.slider("Min Detection Area", 500, 5000, 1500, 500)

    if st.sidebar.button(" Run AI Analysis Pipeline"):
        with st.spinner("Processing signal and detecting hyperbolas..."):

            # 1. Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 2. Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)

            # 3. Threshold to isolate bright regions (hyperbolas are bright)
            _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

            # 4. Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_img = img.copy()
            heatmap = np.zeros_like(img, dtype=np.float32)
            detection_count = 0

            if len(contours) > 0:
                # Sort by area
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

                for i, c in enumerate(contours_sorted[:5]):  # Top 5 detections
                    area = cv2.contourArea(c)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(c)

                        # Check aspect ratio (hyperbolas are wider than tall)
                        aspect_ratio = float(w) / h
                        if 0.5 < aspect_ratio < 3.0:  # Reasonable for hyperbolas
                            # Draw bounding box
                            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            cv2.putText(result_img, f"Hyperbola #{detection_count + 1}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            # Add to heatmap
                            cv2.rectangle(heatmap, (x, y), (x + w, y + h), (1.0,), -1)
                            detection_count += 1

                if detection_count > 0:
                    # Create smooth heatmap
                    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
                    heatmap = np.uint8(255 * heatmap / np.max(heatmap))
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

                    st.sidebar.success(f"✅ Detected {detection_count} hyperbola(s)!")
                else:
                    overlay = img
                    st.sidebar.warning(
                        "️ No hyperbolas detected. Try adjusting Threshold Value (lower it) or Min Detection Area.")
            else:
                overlay = img
                st.sidebar.warning("⚠️ No contours found. Lower the Threshold Value.")

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("2. Preprocessed (CLAHE + Threshold)")
                st.image(thresh, channels="GRAY", use_container_width=True)
                st.caption("Binary threshold isolates bright hyperbolic regions.")

            with col2:
                st.subheader("3. Detection & XAI Heatmap")
                st.image(overlay, channels="BGR", use_container_width=True)
                st.caption(f"Red boxes: {detection_count} hyperbola(s) detected. Heatmap: XAI confidence map.")

            st.success("✅ Pipeline executed successfully!")
else:
    st.info("👈 Please upload a GPR B-scan image from the sidebar to begin.")