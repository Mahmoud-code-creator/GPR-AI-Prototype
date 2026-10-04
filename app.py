import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="GPR AI Prototype - Mahmoud", layout="wide")
st.title("🛰️ GPR Leak Detection Prototype (i3WaterS DC1)")
st.markdown(
    "Developed by: **Mahmoud Elnajjar** | Demonstrating AI-driven preprocessing and XAI for subsurface utility detection.")

st.sidebar.header("⚙️ Hyperbola Detection Settings")
st.sidebar.info("Adjust parameters to optimize hyperbola detection.")

uploaded_file = st.sidebar.file_uploader("Upload a GPR B-scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("1. Original GPR B-scan")
    st.image(img, channels="BGR", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Detection Parameters")

    # Sliders
    clahe_clip = st.sidebar.slider("CLAHE Contrast", 1.0, 10.0, 4.0, 0.5)
    blur_kernel = st.sidebar.slider("Gaussian Blur", 3, 15, 5, 2)
    threshold_val = st.sidebar.slider("Threshold Value", 50, 200, 120, 10)
    min_area = st.sidebar.slider("Min Detection Area", 500, 5000, 1500, 500)

    if st.sidebar.button("🚀 Run AI Analysis Pipeline"):
        with st.spinner("Processing signal and detecting hyperbolas..."):

            # 1. Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 2. Apply Gaussian blur
            blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)

            # 3. Threshold
            _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

            # 4. Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_img = img.copy()
            heatmap = np.zeros_like(img, dtype=np.float32)
            detection_count = 0
            detections_list = []

            if len(contours) > 0:
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

                for i, c in enumerate(contours_sorted[:5]):
                    area = cv2.contourArea(c)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(c)
                        aspect_ratio = float(w) / h

                        if 0.5 < aspect_ratio < 3.0:
                            # Calculate center
                            center_x = x + w // 2
                            center_y = y + h // 2

                            # 1. Draw bounding box (red)
                            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

                            # 2. Draw crosshair marker at center (bright green)
                            marker_size = 20
                            cv2.line(result_img, (center_x - marker_size, center_y),
                                     (center_x + marker_size, center_y), (0, 255, 0), 3)
                            cv2.line(result_img, (center_x, center_y - marker_size),
                                     (center_x, center_y + marker_size), (0, 255, 0), 3)

                            # 3. Draw circle around center
                            cv2.circle(result_img, (center_x, center_y), 15, (0, 255, 0), 2)

                            # 4. Add label with coordinates
                            label = f"#{detection_count + 1} ({center_x},{center_y})"
                            cv2.putText(result_img, label, (x, y - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # 5. Add to heatmap
                            cv2.circle(heatmap, (center_x, center_y), 30, (1.0,), -1)

                            detections_list.append({
                                'id': detection_count + 1,
                                'center': (center_x, center_y),
                                'bbox': (x, y, w, h),
                                'area': area
                            })

                            detection_count += 1

                if detection_count > 0:
                    # Create smooth heatmap
                    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
                    heatmap = np.uint8(255 * heatmap / np.max(heatmap))
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

                    # Add markers to overlay too
                    for det in detections_list:
                        cx, cy = det['center']
                        marker_size = 20
                        cv2.line(overlay, (cx - marker_size, cy),
                                 (cx + marker_size, cy), (0, 255, 0), 3)
                        cv2.line(overlay, (cx, cy - marker_size),
                                 (cx, cy + marker_size), (0, 255, 0), 3)
                        cv2.circle(overlay, (cx, cy), 15, (0, 255, 0), 2)

                    st.sidebar.success(f"✅ Detected {detection_count} hyperbola(s)!")
                else:
                    overlay = img
                    st.sidebar.warning(
                        "⚠️ No hyperbolas detected. Try adjusting Threshold Value (lower it) or Min Detection Area.")
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
                st.caption(
                    f"Green crosshairs: Hyperbola centers. Red boxes: Detection regions. Heatmap: XAI confidence.")

            # Display detection details
            if detection_count > 0:
                st.subheader("📊 Detection Details")
                for det in detections_list:
                    st.markdown(
                        f"**Hyperbola #{det['id']}**: Center at ({det['center'][0]}, {det['center'][1]}), Area: {det['area']:.0f} pixels")

            st.success("✅ Pipeline executed successfully!")
else:
    st.info(" Please upload a GPR B-scan image from the sidebar to begin.")