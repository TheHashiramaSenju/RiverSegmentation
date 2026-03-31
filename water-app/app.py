import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

st.set_page_config(
    page_title="Water Body Segmentation & Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #0284c7; text-align: center; }
    .sub-header { font-size: 1.5rem; color: #0f172a; margin-top: 20px; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; }
    .status-good { color: #10b981; font-weight: bold; font-size: 1.2rem; }
    .status-warn { color: #f59e0b; font-weight: bold; font-size: 1.2rem; }
    .status-bad { color: #ef4444; font-weight: bold; font-size: 1.2rem; }
    .info-box { background: #f0f9ff; border-left: 4px solid #0284c7; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .divider { margin: 30px 0; border-top: 2px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load trained U-Net EfficientNetB0 model with custom metrics"""
    def dice_coef(yt, yp, smooth=1e-6):
        yt = tf.cast(tf.keras.backend.flatten(yt), tf.float32)
        yp = tf.cast(tf.keras.backend.flatten(yp), tf.float32)
        return (2 * tf.reduce_sum(yt * yp) + smooth) / (
            tf.reduce_sum(yt) + tf.reduce_sum(yp) + smooth
        )

    def iou(yt, yp, smooth=1e-6):
        yt = tf.cast(tf.keras.backend.flatten(yt), tf.float32)
        yp = tf.cast(tf.keras.backend.flatten(yp > 0.5), tf.float32)
        inter = tf.reduce_sum(yt * yp)
        union = tf.reduce_sum(yt) + tf.reduce_sum(yp) - inter
        return (inter + smooth) / (union + smooth)

    def loss(yt, yp):
        bce = tf.keras.losses.binary_crossentropy(
            tf.cast(yt, tf.float32), tf.cast(yp, tf.float32)
        )
        return bce + (1 - dice_coef(yt, yp))

    model_path = os.path.join(os.path.dirname(__file__), "best_model.keras")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Ensure best_model.keras is in the app directory.")
        st.stop()

    return keras.models.load_model(
        model_path,
        custom_objects={"loss": loss, "dice_coef": dice_coef, "iou": iou}
    )

@st.cache_data
def get_model_info():
    """Fetch model architecture and training details"""
    return {
        "backbone": "EfficientNetB0",
        "encoder_depth": 5,
        "decoder_blocks": 4,
        "input_size": (256, 256, 3),
        "output_channels": 1,
        "total_params": "4.2M",
        "training_epochs": 35,
        "best_val_iou": 0.778,
        "best_val_dice": 0.857,
        "training_time": "~9 hours",
        "dataset_size": 2841,
        "train_split": 0.85,
        "val_split": 0.15
    }

def preprocess_image(image, target_size=256):
    """Preprocess uploaded image to model input format"""
    img_resized = image.resize((target_size, target_size))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    return img_array

def run_inference(model, img_array):
    """Run model inference on preprocessed image"""
    inp = np.expand_dims(img_array, axis=0).astype(np.float32)
    pred = model.predict(inp, verbose=0)[0]
    return pred

def create_segmentation_mask(pred, threshold=0.5):
    """Generate binary mask from prediction"""
    return (pred[:, :, 0] > threshold).astype(np.uint8)

def calculate_water_metrics(mask):
    """Calculate water body coverage and metrics"""
    total_pixels = mask.size
    water_pixels = int(mask.sum())
    coverage_pct = (water_pixels / total_pixels) * 100

    metrics = {
        "total_pixels": total_pixels,
        "water_pixels": water_pixels,
        "coverage_pct": coverage_pct,
        "dry_pixels": total_pixels - water_pixels,
        "dry_pct": 100 - coverage_pct,
        "area_ratio": water_pixels / total_pixels
    }
    return metrics

def get_health_status(coverage_pct):
    """Determine river health status based on coverage"""
    if coverage_pct < 5:
        return "CRITICAL", "🔴", "#ef4444"
    elif coverage_pct < 15:
        return "SEVERELY DRY", "🟠", "#f97316"
    elif coverage_pct < 30:
        return "DRY", "🟡", "#f59e0b"
    elif coverage_pct < 50:
        return "MODERATE", "🔵", "#0284c7"
    elif coverage_pct < 70:
        return "HEALTHY", "🟢", "#10b981"
    else:
        return "VERY HEALTHY", "✅", "#059669"

def create_overlay(original_img, mask, alpha=0.6):
    """Create visualization overlay of mask on original image"""
    overlay = np.array(original_img, dtype=np.float32)
    water_mask = mask == 1
    overlay[water_mask] = overlay[water_mask] * (1 - alpha) + np.array([0, 150, 255]) * alpha
    return overlay.astype(np.uint8)

def create_comparison_plot(pred_raw, threshold):
    """Create interactive threshold comparison visualization"""
    masks = []
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        masks.append(((pred_raw[:, :, 0] > t).sum() / pred_raw[:, :, 0].size) * 100)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=masks,
            mode="lines+markers",
            name="Water Coverage",
            line=dict(color="#0284c7", width=3),
            marker=dict(size=10)
        )
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Current Threshold")
    fig.update_layout(
        title="Water Coverage vs Detection Threshold",
        xaxis_title="Threshold",
        yaxis_title="Water Coverage (%)",
        template="plotly_white",
        height=400
    )
    return fig

def create_histogram(mask):
    """Create pixel distribution histogram"""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Water", "Dry"],
            y=[int(mask.sum()), mask.size - int(mask.sum())],
            marker=dict(color=["#0284c7", "#d1d5db"]),
            text=[f"{(mask.sum()/mask.size)*100:.1f}%", f"{(1-mask.sum()/mask.size)*100:.1f}%"],
            textposition="auto"
        )
    )
    fig.update_layout(
        title="Pixel Distribution",
        yaxis_title="Pixel Count",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    return fig

def create_confidence_chart(pred_raw):
    """Create model confidence distribution"""
    confidence_vals = pred_raw[:, :, 0].flatten()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=confidence_vals,
            nbinsx=50,
            marker=dict(color="#667eea"),
            name="Confidence Score"
        )
    )
    fig.update_layout(
        title="Model Confidence Distribution",
        xaxis_title="Prediction Confidence",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    return fig

def display_model_architecture():
    """Display model architecture information"""
    info = get_model_info()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Backbone", info["backbone"], delta="Pretrained")
    with col2:
        st.metric("Encoder Depth", f"{info['encoder_depth']}", delta="Levels")
    with col3:
        st.metric("Total Parameters", info["total_params"], delta="4.2M")
    with col4:
        st.metric("Val IoU", f"{info['best_val_iou']:.3f}", delta=f"+{info['best_val_dice']:.3f}")

    st.markdown("""
    <div class="info-box">
    <b>Model Architecture:</b> U-Net with EfficientNetB0 encoder backbone, synchronized data augmentation, 
    BCE + Dice loss, mixed precision FP16 training.
    </div>
    """, unsafe_allow_html=True)

def display_training_metrics():
    """Display training history and metrics"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Training Summary")
        info = get_model_info()
        st.markdown(f"""
        - **Training Epochs:** {info['training_epochs']}
        - **Best Val IoU:** {info['best_val_iou']:.4f}
        - **Best Val Dice:** {info['best_val_dice']:.4f}
        - **Training Time:** {info['training_time']}
        - **Dataset Size:** {info['dataset_size']:,} image pairs
        - **Train/Val Split:** {info['train_split']:.0%} / {info['val_split']:.0%}
        """)

    with col2:
        st.subheader("🎯 Metrics Explanation")
        st.markdown("""
        - **IoU (Intersection over Union):** 0.778 = 77.8% overlap with ground truth
        - **Dice Coefficient:** 0.857 = 85.7% F1-score for segmentation
        - **Mixed Precision:** FP16 for speed, maintains accuracy
        - **Backbone Pretrained:** ImageNet weights for better feature extraction
        """)

def display_about_section():
    """Display application information and usage guide"""
    st.markdown("<h2 class='sub-header'>📖 About This Application</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Overview", "How to Use", "Technical Details"])

    with tab1:
        st.markdown("""
        ### Water Body Segmentation & River Health Analysis

        This application uses **deep learning** to detect and analyze water bodies in satellite imagery.
        It automatically segments water regions and provides:

        - **Real-time segmentation** of water bodies
        - **Coverage metrics** and health status
        - **Interactive visualizations** and detailed analysis
        - **Threshold tuning** for different sensitivity levels
        - **Confidence analysis** of model predictions

        **Use cases:**
        - Environmental monitoring
        - River drying assessment
        - Water resource management
        - Climate change impact analysis
        - Urban water body tracking
        """)

    with tab2:
        st.markdown("""
        ### Step-by-Step Guide

        1. **Upload Image**
           - Click "Browse files" and select a satellite image (JPG or PNG)
           - Image will be automatically resized to 256×256
           - Supported formats: JPG, JPEG, PNG

        2. **Adjust Detection Threshold**
           - Use the slider to control detection sensitivity
           - Lower threshold = more water detected (more false positives)
           - Higher threshold = less water detected (more false negatives)
           - Recommended: 0.45–0.55

        3. **Review Results**
           - Check the three-panel visualization
           - Examine coverage percentage and health status
           - Analyze confidence distribution chart

        4. **Export Analysis**
           - Download visualizations using the camera icon in each chart
           - Screenshots include all metrics and predictions
        """)

    with tab3:
        st.markdown("""
        ### Technical Architecture

        **Model:** U-Net with EfficientNetB0 Encoder
        - Input: 256×256 RGB satellite image
        - Output: 256×256 binary water mask
        - Backbone: EfficientNetB0 (pretrained ImageNet)
        - Loss: BCE + Dice Loss (combined)
        - Optimizer: Adam (learning rate 1e-4, decay to 1e-6)
        - Batch Size: 8 | Epochs: 35 | Training Time: ~9 hours

        **Data Pipeline:**
        - Synchronized augmentation (flip, brightness, contrast)
        - 80/20 train/val split from 2,841 image pairs
        - tf.data optimization: AUTOTUNE, prefetch, parallel loading

        **Performance:**
        - Validation IoU: 0.778 (77.8% accuracy)
        - Validation Dice: 0.857 (85.7% F1-score)
        - Inference Time: ~150ms per image
        - Model Size: 95 MB (quantizable to 24 MB)

        **Preprocessing:**
        - Image resizing to 256×256
        - Normalization to [0, 1] range
        - Automatic grayscale/RGBA conversion
        """)

def main():
    """Main Streamlit application"""

    st.markdown("<h1 class='main-header'>🌊 Water Body Segmentation System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    Satellite imagery analysis using AI to detect water bodies, assess river health, 
    and monitor environmental changes in real-time.
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Navigation")
        app_mode = st.radio(
            "Select Page:",
            ["Upload & Analyze", "Model Info", "About & Guide"]
        )

    if app_mode == "Upload & Analyze":
        st.markdown("<h2 class='sub-header'>📸 Upload Satellite Image</h2>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select a satellite image (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Recommended resolution: 512×512 or higher"
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.success(f"✅ Image loaded successfully - {image.size}")

                model = load_model()

                st.markdown("<h2 class='sub-header'>🎚️ Detection Parameters</h2>", unsafe_allow_html=True)

                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    threshold = st.slider(
                        "Detection Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        help="Confidence level required to classify a pixel as water"
                    )

                with col_param2:
                    overlay_alpha = st.slider(
                        "Overlay Transparency",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.6,
                        step=0.1,
                        help="Transparency of water mask overlay"
                    )

                with st.spinner("🔄 Processing image..."):
                    img_array = preprocess_image(image, target_size=256)
                    pred_raw = run_inference(model, img_array)
                    mask = create_segmentation_mask(pred_raw, threshold=threshold)

                metrics = calculate_water_metrics(mask)
                health_status, health_emoji, health_color = get_health_status(metrics["coverage_pct"])

                st.markdown(f"<h2 class='sub-header'>{health_emoji} River Health Status: {health_status}</h2>",
                           unsafe_allow_html=True)

                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

                with col_metric1:
                    st.markdown(f"""
                    <div style="background: {health_color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0;">Water Coverage</h3>
                    <h2 style="margin: 10px 0;">{metrics['coverage_pct']:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col_metric2:
                    st.markdown(f"""
                    <div style="background: #667eea; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0;">Dry Area</h3>
                    <h2 style="margin: 10px 0;">{metrics['dry_pct']:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col_metric3:
                    st.markdown(f"""
                    <div style="background: #764ba2; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0;">Water Pixels</h3>
                    <h2 style="margin: 10px 0;">{metrics['water_pixels']:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col_metric4:
                    st.markdown(f"""
                    <div style="background: #10b981; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                    <h3 style="margin: 0;">Total Pixels</h3>
                    <h2 style="margin: 10px 0;">{metrics['total_pixels']:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<h2 class='sub-header'>🖼️ Visualization</h2>", unsafe_allow_html=True)

                viz_col1, viz_col2, viz_col3 = st.columns(3)

                with viz_col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True, caption="Input satellite image")

                with viz_col2:
                    st.subheader("Water Mask")
                    st.image(mask * 255, use_column_width=True, caption="Detected water bodies", clamp=True)

                with viz_col3:
                    st.subheader("Overlay")
                    overlay_img = create_overlay(image, mask, alpha=overlay_alpha)
                    st.image(overlay_img, use_column_width=True, caption="Water mask overlaid on original")

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<h2 class='sub-header'>📊 Advanced Analysis</h2>", unsafe_allow_html=True)

                analysis_col1, analysis_col2 = st.columns(2)

                with analysis_col1:
                    st.plotly_chart(create_histogram(mask), use_container_width=True)

                with analysis_col2:
                    st.plotly_chart(create_confidence_chart(pred_raw), use_container_width=True)

                st.plotly_chart(create_comparison_plot(pred_raw, threshold), use_container_width=True)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<h2 class='sub-header'>📋 Detailed Report</h2>", unsafe_allow_html=True)

                report_col1, report_col2, report_col3 = st.columns(3)

                with report_col1:
                    st.markdown(f"""
                    **Segmentation Metrics**
                    - Coverage: {metrics['coverage_pct']:.4f}%
                    - Area Ratio: {metrics['area_ratio']:.6f}
                    - Threshold: {threshold}
                    - Model Confidence Avg: {pred_raw.mean():.4f}
                    """)

                with report_col2:
                    st.markdown(f"""
                    **Image Properties**
                    - Original Size: {image.size}
                    - Processed Size: 256×256
                    - Color Channels: 3 (RGB)
                    - Data Type: Float32 [0,1]
                    """)

                with report_col3:
                    st.markdown(f"""
                    **Analysis Timestamp**
                    - Date: {datetime.now().strftime('%Y-%m-%d')}
                    - Time: {datetime.now().strftime('%H:%M:%S')}
                    - Health Status: {health_status}
                    - Risk Level: {['LOW', 'MODERATE', 'HIGH', 'CRITICAL'][min(3, int(5-metrics['coverage_pct']/20))]}
                    """)

                st.success("✅ Analysis complete! You can download charts using the camera icon in the top-right of each visualization.")

            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")

    elif app_mode == "Model Info":
        st.markdown("<h2 class='sub-header'>🤖 Model Architecture & Performance</h2>", unsafe_allow_html=True)
        display_model_architecture()
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        display_training_metrics()

    elif app_mode == "About & Guide":
        display_about_section()

    st.markdown("""
    ---
    <div style='text-align: center; color: #64748b; margin-top: 50px;'>
    <p>🌍 Water Body Segmentation System v1.0 | Powered by TensorFlow & Streamlit</p>
    <p>Built with U-Net EfficientNetB0 | Real-time Satellite Image Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
