"""Streamlit frontend  (Member 4 - OCR & Application Engineer).

This file is the *UI layer*: it draws widgets and shows what the backend
returns.  All processing happens in ``backend/`` (validation, detection,
geometry, enhancement, OCR, export), so nothing here is hard-coded or faked.

Run with:   streamlit run app.py

Streamlit re-runs this whole script on every interaction, so results are kept
in ``st.session_state``; changing the preview mode or typing in a text box does
NOT reprocess the image.  Only the "Scan document" and "Apply corners" buttons
call the pipeline.
"""
from __future__ import annotations

import cv2
import streamlit as st

from backend.contracts import ScannerError
from backend.enhancement import EnhancementOptions
from backend.export import images_to_pdf
from backend.input_handler import load_image
from backend.ocr import check_tesseract, configure_tesseract, draw_word_boxes
from backend.pipeline import PipelineOptions, make_outline_preview, rescan_with_corners, run_pipeline
from backend.ui_support import (CORNER_NAMES, VARIANT_LABELS, corners_to_percent, default_manual_corners,
                                describe_ocr, get_variant, ocr_text_bytes, percent_to_corners,
                                variant_pdf_bytes, variant_png_bytes)

st.set_page_config(page_title="Document Scanner", page_icon="📄", layout="wide",
                   initial_sidebar_state="collapsed")
configure_tesseract()

MAX_PAGES = 10   # pages one PDF may hold; every page keeps its enhanced images in memory

# --------------------------------------------------------------------------
# Dark theme (indigo ink).  Palette:
#   background #0f1220 | panel #171b2e | border #2a3050
#   text #e8eaf6       | muted #9aa1c4 | accent #8b93ff | accent hover #a3a9ff
# The same colours are in .streamlit/config.toml; the CSS below also works
# without that file and styles the parts the theme file cannot reach.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0f1220;
        --panel: #171b2e;
        --panel-2: #1d2238;
        --border: #2a3050;
        --text: #e8eaf6;
        --muted: #9aa1c4;
        --accent: #8b93ff;
        --accent-hover: #a3a9ff;
    }

    /* No sidebar at all */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] { display: none !important; }

    /* Page */
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
    h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }
    h1 { font-weight: 700; }
    [data-testid="stCaptionContainer"], .stCaption { color: var(--muted); }

    /* Expanders (settings, corner controls, previews) */
    [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    [data-testid="stExpander"] summary { color: var(--text); }
    [data-testid="stExpander"] summary:hover { color: var(--accent-hover); }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        background: var(--panel);
        border: 1.5px dashed var(--border);
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--accent); }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: var(--panel-2);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        transition: border-color .15s, background .15s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent-hover);
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #0f1220;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #0f1220;
    }
    .stButton > button:disabled, .stDownloadButton > button:disabled { opacity: .45; }

    /* Inputs */
    [data-baseweb="input"], [data-baseweb="select"] > div, textarea {
        background: var(--panel-2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border); gap: 4px; }
    .stTabs [data-baseweb="tab"] { color: var(--muted); }
    .stTabs [aria-selected="true"] { color: var(--accent-hover); }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

    /* Metrics as small panels */
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }

    /* Alerts: keep them readable on dark */
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--border); }

    /* Images */
    [data-testid="stImage"] img { border-radius: 10px; border: 1px solid var(--border); }

    /* Keyboard focus stays visible */
    button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 2px solid var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state defaults
#   result_page_id = id of the page (in ``pages``) that the current ``result`` was stored as.
#   A manual-corner re-scan replaces exactly that page, never some other one.
# --------------------------------------------------------------------------
for key, default in {"result": None, "file_key": None, "result_uid": 0, "corner_epoch": 0,
                     "default_pct": None, "pages": [], "page_counter": 0,
                     "current_page_id": None, "result_page_id": None}.items():
    st.session_state.setdefault(key, default)


@st.cache_data(ttl=30, show_spinner=False)
def _tesseract_status(lang: str):
    return check_tesseract(lang)


@st.cache_data(show_spinner=False, max_entries=24)
def _png_bytes(uid: int, variant: str) -> bytes:
    """Encoded once per result (uid) instead of on every Streamlit re-run."""
    return variant_png_bytes(st.session_state["result"], variant)


@st.cache_data(show_spinner=False, max_entries=24)
def _pdf_bytes(uid: int, variant: str) -> bytes:
    return variant_pdf_bytes(st.session_state["result"], variant)


def _store_result(result, reset_corner_widgets: bool) -> None:
    st.session_state["result"] = result
    st.session_state["result_uid"] += 1
    if reset_corner_widgets and result.original is not None:
        start = result.corners if result.corners is not None else default_manual_corners(result.original.shape)
        st.session_state["default_pct"] = corners_to_percent(start, result.original.shape)
        st.session_state["corner_epoch"] += 1   # new slider keys -> new default values


def _register_page(res, name, replace_page_id=None):
    """Add the scan as a page of the multi-page document, or swap an existing page.

    ``replace_page_id`` (manual-corner re-scan): the id of the page to swap.  The new page takes the
    SAME position, so the page order in the PDF does not change.  If that id no longer exists (the page
    was removed, or the first scan failed and never became a page) the scan is appended instead, so a
    different page is never overwritten by mistake.

    Only the enhanced variants are kept (not the whole PipelineResult) to save memory.
    Returns the id of the stored page, or None if nothing was stored.
    """
    if not res.ok or res.enhancement is None:
        return None
    pages = st.session_state["pages"]
    st.session_state["page_counter"] += 1
    color = res.enhancement.variants.get("color")
    scale = 260.0 / max(color.shape[:2]) if color is not None else 1.0
    page = {"id": st.session_state["page_counter"], "name": name or "page",
            "variants": dict(res.enhancement.variants),
            "thumb": cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if color is not None else None}
    index = None
    if replace_page_id is not None:
        index = next((i for i, p in enumerate(pages) if p["id"] == replace_page_id), None)
    if index is not None:
        page["name"] = pages[index]["name"]
        pages[index] = page            # same position -> page order does not change
    elif len(pages) < MAX_PAGES:
        pages.append(page)
    else:
        return None
    st.session_state["current_page_id"] = page["id"]
    return page["id"]


@st.cache_data(show_spinner=False, max_entries=8)
def _pdf_all_pages_bytes(page_ids: tuple, variant: str) -> bytes:
    """One PDF with every page, in order.  Cached on the page ids + chosen format.

    A replaced page gets a brand-new id, so the cache key changes and the PDF always shows the new image.
    """
    pages = {p["id"]: p for p in st.session_state["pages"]}
    images = [pages[i]["variants"].get(variant) for i in page_ids]
    return images_to_pdf([img for img in images if img is not None])


# --------------------------------------------------------------------------
# Header + settings (these used to live in the sidebar)
# --------------------------------------------------------------------------
st.title("📄 Document Scanner & Image Enhancement")
st.caption("Upload a photo of a paper document. The app finds the page, straightens it (perspective correction), "
           "cleans it up, and extracts the text with OCR. Classical computer vision (OpenCV) + Tesseract - no model training.")

with st.expander("⚙️ Settings", expanded=False):
    left, right = st.columns(2)
    with left:
        st.subheader("Text extraction")
        ocr_on = st.checkbox("Run OCR (text extraction)", value=True)
        ocr_lang = st.text_input("OCR language code(s)", value="eng", help="Tesseract language, e.g. eng or eng+fra")
        ocr_psm = st.selectbox("Page layout (PSM)", [3, 4, 6, 11], index=0,
                               format_func=lambda v: {3: "3 - Automatic (default)", 4: "4 - Single column",
                                                      6: "6 - Single uniform block", 11: "11 - Sparse text"}[v])
        ocr_source = st.selectbox("Image given to OCR", ["grayscale", "bw", "color"],
                                  format_func=lambda k: VARIANT_LABELS[k],
                                  help="Grayscale usually works best; compare with evaluation/ocr_metrics.py")
        if ocr_on:
            ok, message = _tesseract_status(ocr_lang)
            (st.success if ok else st.warning)(message)
    with right:
        st.subheader("Enhancement")
        shadow_removal = st.checkbox("Remove shadows / uneven light", value=True)
        denoise = st.checkbox("Denoise", value=True)
        clahe = st.selectbox("Contrast boost (CLAHE)", ["auto", "on", "off"], index=0,
                             help="auto = only when text/paper contrast is weak")
        threshold_c = st.slider("B&W threshold constant", 2, 25, 10,
                                help="Higher = cleaner white background but thinner/fainter text")
        auto_orient = st.checkbox("Auto-rotate sideways pages (needs Tesseract)", value=True)
        auto_deskew = st.checkbox("Straighten tilted text (deskew)", value=True,
                                  help="Rotates the page so text lines are horizontal. Big accuracy gain on tilted photos.")


def _options() -> PipelineOptions:
    return PipelineOptions(
        enhancement=EnhancementOptions(shadow_removal=shadow_removal, denoise=denoise, clahe=clahe,
                                       threshold_c=int(threshold_c)),
        run_ocr=ocr_on, ocr_lang=ocr_lang.strip() or "eng", ocr_psm=int(ocr_psm),
        ocr_source=ocr_source, auto_orient=auto_orient, auto_deskew=auto_deskew)


# --------------------------------------------------------------------------
# Upload + scan
# --------------------------------------------------------------------------
uploaded = st.file_uploader("Upload a document photo (JPEG, PNG or WebP)", type=["jpg", "jpeg", "png", "webp"])
if uploaded is not None:
    file_key = (uploaded.name, uploaded.size)
    if file_key != st.session_state["file_key"]:       # a different file: forget old results
        st.session_state.update(file_key=file_key, result=None, default_pct=None, result_page_id=None)
    with st.expander("Preview of the uploaded file", expanded=st.session_state["result"] is None):
        # Validate with the backend loader first: st.image() on a corrupted file would raise inside Streamlit.
        try:
            preview = load_image(uploaded.getvalue(), uploaded.name)
            st.image(preview.image, channels="BGR", caption=f"{uploaded.name} ({preview.image.shape[1]}x"
                     f"{preview.image.shape[0]} px)", use_container_width=True)
        except ScannerError as exc:
            st.error(f"This file cannot be used: {exc.message}")
            if exc.hint:
                st.caption(exc.hint)

pages_full = len(st.session_state["pages"]) >= MAX_PAGES
if pages_full:
    st.warning(f"The document already has {MAX_PAGES} pages (the maximum). Remove a page or start a new "
               "document before scanning another one.")
if st.button("🔍 Scan document", type="primary", disabled=uploaded is None or pages_full):
    with st.spinner("Detecting the page, correcting perspective, enhancing and reading text..."):
        outcome = run_pipeline(uploaded.getvalue(), uploaded.name, _options())
    _store_result(outcome, reset_corner_widgets=True)
    # Remember which page this result became (None if the scan failed), so a later manual-corner
    # re-scan replaces exactly this page.
    st.session_state["result_page_id"] = _register_page(outcome, uploaded.name)

result = st.session_state["result"]


def _pages_panel() -> None:
    pages = st.session_state["pages"]
    if not pages:
        return
    with st.expander(f"📚 Document pages ({len(pages)}/{MAX_PAGES})", expanded=len(pages) > 1):
        st.caption("The PDF download contains all these pages in this order. "
                   "The PNG download is only the page you are viewing.")
        cols = st.columns(min(len(pages), 5))
        for i, page in enumerate(pages):
            with cols[i % len(cols)]:
                if page["thumb"] is not None:
                    st.image(page["thumb"], channels="BGR", caption=f"Page {i + 1}: {page['name']}")
                if st.button("Remove", key=f"rm_{page['id']}"):
                    st.session_state["pages"] = [p for p in pages if p["id"] != page["id"]]
                    if st.session_state["current_page_id"] == page["id"]:
                        st.session_state["current_page_id"] = None
                    if st.session_state["result_page_id"] == page["id"]:
                        st.session_state["result_page_id"] = None
                    st.rerun()
        if st.button("Start new document"):
            st.session_state.update(pages=[], current_page_id=None, result_page_id=None)
            st.rerun()


_pages_panel()


# --------------------------------------------------------------------------
# Manual corner correction (a real control: it re-runs the pipeline)
# --------------------------------------------------------------------------
def _manual_corner_controls(res) -> None:
    with st.expander("✏️ Adjust corners manually",
                     expanded=(not res.ok) or res.corner_source == "virtual"):
        st.caption("Move the four points so that the outline follows the page edges, then press "
                   "**Apply corners**. The preview updates as you move the sliders; "
                   "the scan is recomputed only when you press the button.")
        epoch = st.session_state["corner_epoch"]
        start = st.session_state["default_pct"] or corners_to_percent(
            default_manual_corners(res.original.shape), res.original.shape)
        values = []
        cols = st.columns(4)
        for i, name in enumerate(CORNER_NAMES):
            with cols[i]:
                st.markdown(f"**{name}**")
                x = st.slider("X (%)", 0.0, 100.0, float(start[2 * i]), 0.5, key=f"c{epoch}_{i}_x")
                y = st.slider("Y (%)", 0.0, 100.0, float(start[2 * i + 1]), 0.5, key=f"c{epoch}_{i}_y")
                values.extend([x, y])
        chosen = percent_to_corners(values, res.original.shape)
        st.image(make_outline_preview(res.original, chosen), channels="BGR", use_container_width=True,
                 caption="Outline from the current slider positions")
        if st.button("✅ Apply corners and re-scan"):
            with st.spinner("Re-running perspective correction, enhancement and OCR..."):
                again = rescan_with_corners(res, chosen, _options())
            _store_result(again, reset_corner_widgets=False)
            # Replace the page that THIS result was stored as (same position in the PDF).
            # If the re-scan failed nothing is stored and the old page stays untouched.
            new_id = _register_page(again, None, replace_page_id=st.session_state["result_page_id"])
            if new_id is not None:
                st.session_state["result_page_id"] = new_id
            st.rerun()


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if result is not None and not result.ok:
    err = result.error
    st.error(f"**{err.stage.capitalize()} problem:** {err.message}")
    if err.hint:
        st.info(err.hint)
    for w in result.warnings:
        st.warning(w)
    if result.original is not None:
        st.image(result.outline_preview, channels="BGR", caption="Your image", use_container_width=True)
        if result.detection is not None and result.detection.debug:
            with st.expander("Detection diagnostics (what the detector saw)"):
                dbg = result.detection.debug
                cols = st.columns(2)
                cols[0].image(dbg["edges"], caption="Edges (Canny)", use_container_width=True)
                cols[1].image(dbg["candidates"], channels="BGR", caption="Candidate outlines", use_container_width=True)
        _manual_corner_controls(result)

elif result is not None:
    det = result.detection
    seconds = sum(result.timings.values())
    if result.corner_source == "manual":
        st.success(f"Scan complete in {seconds:.1f} s using the corners you set manually.")
    elif result.corner_source == "text":
        st.info(f"Scan complete in {seconds:.1f} s. The page edge was not found, so the crop was fitted around the text.")
    elif result.corner_source == "virtual":
        st.warning("The page was not detected, so the whole photo was used. Text accuracy will be lower. "
                   "Set the four corners manually below.")
    else:
        st.success(f"Scan complete in {seconds:.1f} s. Page found by '{det.method}' "
                   f"(candidate score {det.quality:.2f}; a heuristic ranking, not a probability).")
    for w in result.warnings:
        st.warning(w)

    mode = st.radio("Enhanced view / download format", list(VARIANT_LABELS), format_func=lambda k: VARIANT_LABELS[k],
                    horizontal=True, key="view_mode")
    c1, c2, c3 = st.columns(3)
    c1.image(result.outline_preview, channels="BGR", caption="1. Original with detected outline", use_container_width=True)
    c2.image(result.warped, channels="BGR", caption="2. Perspective-corrected page", use_container_width=True)
    shown = get_variant(result, mode)
    c3.image(shown, channels="BGR" if shown.ndim == 3 else "GRAY", clamp=True,
             caption=f"3. {VARIANT_LABELS[mode]}", use_container_width=True)

    uid = st.session_state["result_uid"]
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("⬇️ Enhanced scan (PNG)", _png_bytes(uid, mode), file_name="scan.png", mime="image/png")
    page_ids = tuple(p["id"] for p in st.session_state["pages"])
    n_pages = len(page_ids)
    d2.download_button(f"⬇️ Enhanced scan (PDF, {n_pages} page{'s' if n_pages != 1 else ''})",
                       _pdf_all_pages_bytes(page_ids, mode) if page_ids else b"",
                       file_name="scan.pdf", mime="application/pdf", disabled=not page_ids)
    d3.download_button("⬇️ Corrected page (PNG)", _png_bytes(uid, "color"), file_name="scan_color.png", mime="image/png")
    text_ready = result.ocr is not None and result.ocr.status == "ok"
    d4.download_button("⬇️ Extracted text (TXT)", ocr_text_bytes(result.ocr), file_name="text.txt",
                       mime="text/plain; charset=utf-8", disabled=not text_ready)

    tab_text, tab_conf, tab_diag = st.tabs(["📝 Extracted text", "🎯 OCR confidence", "🔧 Diagnostics"])
    with tab_text:
        if result.ocr is None:
            st.info("OCR is turned off in Settings.")
        elif result.ocr.status == "ok":
            st.text_area("Recognised text (you can select and copy it)", result.ocr.text, height=380)
        else:
            st.warning(describe_ocr(result.ocr))
            st.caption("The scanned image above is still available for download.")
    with tab_conf:
        ocr = result.ocr
        if ocr is None or ocr.status != "ok":
            st.info("No OCR confidence data available.")
        else:
            a, b, c = st.columns(3)
            a.metric("Words recognised", len(ocr.words))
            b.metric("Mean confidence", f"{ocr.mean_confidence:.0f} %")
            c.metric("Words below 60 %", len(ocr.low_confidence_words(60)))
            st.caption("Confidence is Tesseract's own estimate. It helps to spot doubtful words but does not "
                       "guarantee that a word is right or wrong.")
            if st.checkbox("Show word boxes (green = confident, red = doubtful)"):
                base = get_variant(result, result.ocr_source or "grayscale")
                st.image(draw_word_boxes(base, ocr.words), channels="BGR", use_container_width=True)
            low = ocr.low_confidence_words(60)
            if low:
                st.dataframe([{"word": w.text, "confidence": round(w.conf, 1)} for w in low], use_container_width=True)
    with tab_diag:
        q = result.enhancement.quality
        a, b, c = st.columns(3)
        a.metric("Sharpness (Laplacian var.)", f"{q.sharpness:.0f}")
        b.metric("Paper/ink contrast", f"{q.ink_contrast:.0f}")
        c.metric("Page size (px)", f"{q.resolution[0]} x {q.resolution[1]}")
        st.write("Filters applied:", ", ".join(result.enhancement.applied))
        st.write("Stage timings (s):", result.timings)
        if det is not None and det.debug and result.corner_source == "auto":
            cols = st.columns(2)
            cols[0].image(det.debug["edges"], caption="Edges (Canny)", use_container_width=True)
            cols[1].image(det.debug["candidates"], channels="BGR", caption="Candidate outlines (green = chosen)",
                          use_container_width=True)
    _manual_corner_controls(result)

elif uploaded is None:
    st.info("Upload an image to begin. Tip: photograph the page on a plain, contrasting surface with all four "
            "corners visible.")