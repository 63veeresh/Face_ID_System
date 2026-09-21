"""
Local Face Recognition & Identification System - Interactive Streamlit Application.
Features real-time face detection, embedding matching, enrollment, database gallery,
genuine evaluation benchmarking, and technical interview defense guide.
"""

import time
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="DeepFaceID | Local Face Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern premium dark-mode interface
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }
        
        .status-badge-known {
            background: linear-gradient(135deg, #065F46 0%, #047857 100%);
            color: #A7F3D0;
            padding: 0.35rem 0.8rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
            border: 1px solid #10B981;
        }
        
        .status-badge-unknown {
            background: linear-gradient(135deg, #7F1D1D 0%, #B91C1C 100%);
            color: #FECACA;
            padding: 0.35rem 0.8rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
            border: 1px solid #EF4444;
        }

        .candidate-row {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 0.6rem 1rem;
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Import system pipeline modules
from src.config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    UNKNOWN_LABEL,
    DEVICE,
    DATA_DIR,
)
from src.pipeline import FaceRecognitionPipeline


@st.cache_resource(show_spinner="Initializing MTCNN & InceptionResnetV1 weights...")
def get_pipeline() -> FaceRecognitionPipeline:
    """Cached pipeline instance ensures weights load only once into memory."""
    return FaceRecognitionPipeline(device=DEVICE)


pipeline = get_pipeline()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/splash.png", use_column_width=True)
    st.markdown("### ⚙️ Engine Parameters")
    
    threshold_val = st.slider(
        "Similarity Threshold (τ)",
        min_value=0.30,
        max_value=0.95,
        value=float(DEFAULT_SIMILARITY_THRESHOLD),
        step=0.01,
        help="Similarity score cutoff for accepting an identity. Matches below this are rejected as UNKNOWN.",
    )
    
    show_landmarks = st.checkbox("Show 5 Facial Landmarks", value=True)
    
    st.markdown("---")
    st.markdown("### 🖥️ Hardware & State")
    st.caption(f"**Compute Device:** `{pipeline.device.upper()}`")
    st.caption(f"**Detector:** `MTCNN (P-Net, R-Net, O-Net)`")
    st.caption(f"**Embedder:** `InceptionResnetV1 (VGGFace2)`")
    st.caption(f"**Embedding Vector:** `512-Dimensional (Unit Norm)`")
    
    enrolled_count = len(pipeline.database)
    st.metric(label="Enrolled People", value=enrolled_count)
    
    st.markdown("---")
    app_mode = st.radio(
        "Navigation",
        [
            "🔍 Face Identification",
            "👤 Enroll New Identity",
            "🗂️ Enrolled Database Gallery",
            "📊 Empirical Evaluation",
            "🧠 Architecture & Interview Guide",
        ],
        index=0,
    )

# ----------------- TAB 1: FACE IDENTIFICATION -----------------
if app_mode == "🔍 Face Identification":
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800;">Face Identification & Recognition</h1>
            <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:1.05rem;">
                Detect faces via MTCNN, compute 512-D VGGFace2 embeddings, and match against enrolled database with threshold rejection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Input Source")
        source_mode = st.radio("Choose Input Mode", ["Upload Image File", "Use Live Camera"], horizontal=True)
        
        input_image = None
        if source_mode == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Upload photo (JPEG, PNG)", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
        else:
            camera_file = st.camera_input("Take a photo")
            if camera_file is not None:
                input_image = Image.open(camera_file).convert("RGB")

        if input_image is not None:
            st.image(input_image, caption="Original Input Image", use_column_width=True)

    with col2:
        st.markdown("### 🎯 Detection & Recognition Results")
        if input_image is not None:
            with st.spinner("Processing face detection & embedding..."):
                start_time = time.time()
                results = pipeline.identify_image(input_image, threshold=threshold_val)
                elapsed = (time.time() - start_time) * 1000

            st.caption(f"⚡ Inferred in **{elapsed:.1f} ms** | Faces detected: **{len(results)}**")

            if not results:
                st.warning("⚠️ No faces detected in the image. Ensure the face is clearly visible and well-lit.")
            else:
                annotated = pipeline.annotate_image(input_image, results, show_landmarks=show_landmarks)
                st.image(annotated, caption="Annotated Bounding Boxes & Identifications", use_column_width=True)

                st.markdown("---")
                st.markdown("#### Detailed Match Breakdown")

                for i, (face, match) in enumerate(results, start=1):
                    with st.container():
                        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                        f_col1, f_col2 = st.columns([1, 3])
                        with f_col1:
                            st.image(face.face_image, caption=f"Face #{i} Crop", width=110)
                        with f_col2:
                            if match.is_known:
                                st.markdown(
                                    f'<div class="status-badge-known">✓ RECOGNIZED: {match.identity}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<div class="status-badge-unknown">✕ REJECTED: {UNKNOWN_LABEL}</div>',
                                    unsafe_allow_html=True,
                                )

                            st.markdown(f"**Similarity Score:** `{match.similarity:.4f}` (Threshold τ = `{match.threshold:.4f}`)")
                            st.progress(max(0.0, min(1.0, float(match.similarity))))

                            if match.is_known:
                                st.success(f"Match confirmed! Closest enrolled candidate is **{match.identity}** with score **{match.similarity:.4f}**.")
                            else:
                                st.error(f"Rejection Reason: {match.rejection_reason}")

                            if match.all_candidates:
                                with st.expander("Inspect Candidate Similarity Rankings"):
                                    for cand in match.all_candidates[:5]:
                                        st.write(f"**#{cand.rank}** `{cand.name}` — Cosine Similarity: `{cand.similarity:.4f}`")
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Upload an image or capture a photo using the left panel to begin recognition.")

# ----------------- TAB 2: ENROLL NEW IDENTITY -----------------
elif app_mode == "👤 Enroll New Identity":
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800;">Enroll New Identity</h1>
            <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:1.05rem;">
                Register a new person into the local biometric database with automatic MTCNN face cropping and 512-D embedding extraction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Identity Details")
        person_name = st.text_input("Full Name / Identifier *", placeholder="e.g. Marie Curie")
        person_notes = st.text_input("Notes / Role (Optional)", placeholder="e.g. Lead Researcher")
        
        source = st.radio("Photo Source", ["Upload Image", "Webcam Capture"], horizontal=True)
        enroll_img = None
        if source == "Upload Image":
            upload = st.file_uploader("Upload Enrollment Photo", type=["jpg", "jpeg", "png"], key="enroll_upload")
            if upload:
                enroll_img = Image.open(upload).convert("RGB")
        else:
            cam = st.camera_input("Take Enrollment Photo", key="enroll_cam")
            if cam:
                enroll_img = Image.open(cam).convert("RGB")

    with col2:
        st.markdown("### 🔍 Face Verification & Preview")
        if enroll_img is not None:
            st.image(enroll_img, caption="Enrollment Source Image", use_column_width=True)
            detected = pipeline.detector.detect_single_face(enroll_img)

            if detected is None:
                st.error("❌ No face detected. Please provide a clear, front-facing portrait.")
            else:
                st.success(f"✓ Face detected with MTCNN confidence {detected.score:.2%}")
                st.image(detected.face_image, caption="Aligned 160x160 Face Crop", width=160)

                if st.button("💾 Complete Enrollment", type="primary"):
                    if not person_name.strip():
                        st.error("Please enter a valid identity name before enrolling.")
                    else:
                        with st.spinner("Extracting 512-D VGGFace2 embedding and persisting..."):
                            success, msg, _ = pipeline.enroll_person(
                                name=person_name.strip(),
                                image_input=enroll_img,
                                notes=person_notes,
                            )
                        if success:
                            st.balloons()
                            st.success(msg)
                        else:
                            st.error(msg)
        else:
            st.info("👈 Provide an image and name on the left panel to register an identity.")

# ----------------- TAB 3: ENROLLED DATABASE GALLERY -----------------
elif app_mode == "🗂️ Enrolled Database Gallery":
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800;">Enrolled Identity Catalog</h1>
            <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:1.05rem;">
                Browse, search, and manage registered profiles stored in the local persistent database.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = pipeline.database.list_all()
    
    col_search, col_action = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Search Enrolled Person", placeholder="Type name to filter...")
    with col_action:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Entire Database", type="secondary"):
            pipeline.database.clear()
            st.warning("Database cleared!")
            st.rerun()

    filtered = [r for r in records if search_query.lower() in r["name"].lower()] if search_query else records

    st.markdown(f"**Total Records:** {len(records)} | **Matching Filter:** {len(filtered)}")
    st.markdown("---")

    if not filtered:
        st.info("No identities found in the database. Head to 'Enroll New Identity' to register people!")
    else:
        # Render grid of cards
        cols_per_row = 3
        rows = [filtered[i:i + cols_per_row] for i in range(0, len(filtered), cols_per_row)]
        for row in rows:
            grid_cols = st.columns(cols_per_row)
            for idx, item in enumerate(row):
                with grid_cols[idx]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if item.get("image_path") and Path(item["image_path"]).exists():
                        st.image(item["image_path"], use_column_width=True)
                    else:
                        st.markdown("📷 *No image preview*")
                    
                    st.markdown(f"### {item['name']}")
                    st.caption(f"**Samples:** {item.get('sample_count', 1)}")
                    st.caption(f"**Enrolled:** {item.get('enrolled_at', 'N/A')}")
                    if item.get("notes"):
                        st.caption(f"**Notes:** {item['notes']}")
                    
                    if st.button(f"Delete {item['name']}", key=f"del_{item['name']}"):
                        pipeline.database.delete(item["name"])
                        st.success(f"Deleted {item['name']}")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- TAB 4: EMPIRICAL EVALUATION -----------------
elif app_mode == "📊 Empirical Evaluation":
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800;">Empirical System Evaluation</h1>
            <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:1.05rem;">
                Genuine benchmarking across genuine test pairs and unknown impostor cases. No fabricated numbers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    from src.data_setup import load_or_create_evaluation_dataset
    from evaluate import evaluate_recognition_pipeline

    eval_json = DATA_DIR / "evaluation_results.json"

    if st.button("🚀 Run Live Evaluation Benchmark", type="primary"):
        with st.spinner("Running genuine evaluation across test sets..."):
            dataset = load_or_create_evaluation_dataset()
            report = evaluate_recognition_pipeline(pipeline, dataset)
            if "error" not in report:
                with open(eval_json, "w", encoding="utf-8") as f:
                    import json
                    json.dump(report, f, indent=2)
                st.success("Evaluation benchmark successfully finished!")
            else:
                st.error(report["error"])

    # Display results if file exists
    if eval_json.exists():
        import json
        with open(eval_json, "r", encoding="utf-8") as f:
            rep = json.load(f)

        cfg = rep["metrics_at_configured_threshold"]
        stats = rep["similarity_statistics"]

        st.markdown(f"### 📈 Performance Summary (τ = `{rep['configured_threshold']}`)")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Accuracy", f"{cfg['Accuracy']*100:.1f}%")
        m_col2.metric("True Accept Rate (Recall)", f"{cfg['TAR_Recall']*100:.1f}%")
        m_col3.metric("False Accept Rate (FAR)", f"{cfg['FAR']*100:.1f}%")
        m_col4.metric("F1-Score", f"{cfg['F1_Score']*100:.1f}%")

        st.markdown("---")
        st.markdown("### 📊 Similarity Score Distributions")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Genuine Matches (Same Person):**\n- Mean Similarity: `{stats['genuine_mean']:.4f}`\n- Std Dev: `{stats['genuine_std']:.4f}`\n- Range: `[{stats['genuine_min']:.4f}, {stats['genuine_max']:.4f}]`")
        with c2:
            st.warning(f"**Impostor Queries (Different/Unknown Person):**\n- Mean Similarity: `{stats['impostor_mean']:.4f}`\n- Std Dev: `{stats['impostor_std']:.4f}`\n- Range: `[{stats['impostor_min']:.4f}, {stats['impostor_max']:.4f}]`")

        st.markdown(f"**Separation Margin:** `{stats['separation_margin']:.4f}` (higher indicates clearer decision boundary).")

        st.markdown("---")
        st.markdown("### 🎚️ Threshold Sensitivity & Error Trade-off Analysis")
        st.caption("Inspect how changing threshold τ influences True Accept Rate (TAR) vs False Accept Rate (FAR):")

        # Display sweep table
        sweep_data = rep["threshold_sweep"]
        st.dataframe(sweep_data, use_container_width=True)

# ----------------- TAB 5: ARCHITECTURE & INTERVIEW GUIDE -----------------
elif app_mode == "🧠 Architecture & Interview Guide":
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin:0; font-size:2.2rem; font-weight:800;">Architecture & Interview Guide</h1>
            <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:1.05rem;">
                Comprehensive technical explanation of design decisions, mathematical principles, failure modes, and interview defense.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    ### 1. System Pipeline Overview
    The system performs closed-set or open-set facial identification using a three-stage modular architecture:
    
    1. **Detection & Alignment (MTCNN)**:
       - **P-Net (Proposal Network)**: Scans input image at multiple pyramid scales to generate candidate face bounding boxes.
       - **R-Net (Refinement Network)**: Filters candidates and discards background false positives via non-maximum suppression (NMS).
       - **O-Net (Output Network)**: Outputs refined bounding coordinates and 5 critical facial landmark coordinates (left eye, right eye, nose, left mouth corner, right mouth corner).
       - **Alignment**: Rotates and crops face to standard $160 \\times 160$ pixel resolution standardized to $[-1, 1]$.
    
    2. **Deep Feature Extraction (InceptionResnetV1)**:
       - Employs InceptionResnetV1 pretrained on VGGFace2 (comprising 3.31M images across 9,131 identities).
       - Compresses high-dimensional facial geometry into a compact **512-dimensional continuous embedding vector**.
       - **L2 Normalization**: Every embedding vector $v$ is projected onto the unit hypersphere: $\\hat{v} = \\frac{v}{\\|v\\|_2}$ such that $\\|\\hat{v}\\|_2 = 1.0$.
    
    3. **Matching & Rejection Logic**:
       - Compares query embedding $q$ against all enrolled embeddings $e_i$ using **Cosine Similarity**:
         $$\\text{Cosine Similarity}(q, e_i) = \\frac{q \\cdot e_i}{\\|q\\|_2 \\|e_i\\|_2} = q \\cdot e_i$$
       - Finds the highest candidate similarity $s^* = \\max_i(s_i)$ and identity $I^* = \\text{argmax}_i(s_i)$.
       - **Rejection Boundary**:
         - If $s^* \\ge \\tau$ (default $\\tau = 0.65$): returns $I^*$ as recognized person.
         - If $s^* < \\tau$: rejects and returns **`UNKNOWN`** with explanation.
    
    ---
    
    ### 2. Why Cosine Similarity over Euclidean Distance?
    - Deep face embeddings trained with angular / cosine margin losses (or triplet loss with normalized vectors) reside on a hypersphere.
    - Since vectors are L2-normalized:
      $$\\|u - v\\|_2^2 = \\|u\\|_2^2 + \\|v\\|_2^2 - 2 (u \\cdot v) = 1 + 1 - 2 \\cos(\\theta) = 2(1 - \\cos(\\theta))$$
    - Cosine similarity directly represents the angular similarity $\\cos(\\theta)$ bounded strictly in $[-1, 1]$, making threshold calibration intuitive and independent of vector magnitude.
    
    ---
    
    ### 3. Threshold Selection Methodology
    - **Low Threshold (e.g. 0.45)**: High TAR (Recall), but high FAR (impostors incorrectly accepted). Suitable only when false rejection is intolerable.
    - **High Threshold (e.g. 0.80)**: Low FAR, high security, but high FRR (genuine people rejected under varying lighting/expressions).
    - **Empirical Optimum (0.65)**: VGGFace2 InceptionResnetV1 achieves the Equal Error Rate (EER) / optimal F1 balance around **0.60 - 0.70**, ensuring minimal False Accepts while preserving high True Acceptance.
    
    ---
    
    ### 4. Known Failure Cases & Mitigations
    1. **Extreme Poses (> 45° Yaw/Pitch)**: MTCNN detection confidence drops, and profile views hide key facial landmarks. *Mitigation: Request user to face camera or enroll multiple poses.*
    2. **Low Resolution (< 40px face size)**: Insufficient high-frequency details for InceptionResnetV1. *Mitigation: MTCNN `min_face_size` filter rejects blur.*
    3. **Severe Lighting / Shadows**: Strong directional shadows skew embeddings. *Mitigation: Standardize input contrast, histogram equalization.*
    4. **Physical Occlusions (Masks, Large Sunglasses)**: Occludes eye/mouth landmarks. *Mitigation: Multi-modal verification or masked-face fine-tuned models.*
    
    ---
    
    ### 5. Production-Ready Improvements
    - **FAISS / Milvus Vector Index**: For scaling to millions of enrolled identities, replace linear $O(N)$ dot product with an Approximate Nearest Neighbor (ANN) HNSW index.
    - **ArcFace / AdaFace Loss**: Modern additive angular margin loss achieves even sharper inter-class separation than VGGFace2 triplet loss.
    - **Liveness Detection & Anti-Spoofing**: Add depth analysis, blink detection, or texture frequency analysis (FAS) to prevent 2D photo spoofing.
    """)
