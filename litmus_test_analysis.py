### Import supporting libraries
import datetime
import getpass
import io
import psycopg2
from psycopg2 import pool
import pytz
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import docx
import scipy.ndimage as ndi
from scipy.interpolate import interp1d as intp
import skimage
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks, rescale
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import find_contours

### Page configuration MUST be the first Streamlit command
st.set_option("client.toolbarMode", "minimal")
st.set_page_config(
    page_title="AO Litmus Test Analysis",
    page_icon="Active",
    layout="wide",
)

### Configure Neon Connection String & Constants
NEON_CONN = st.secrets["NEON_CONN"]
logged_in_user = getpass.getuser()
pst = pytz.timezone("America/Los_Angeles")


### Thread-safe connection pool shared across all concurrent user sessions
@st.cache_resource
def get_db_pool():
    return pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=NEON_CONN,
        sslmode="require",
    )


def save_test_to_db(data):
    """Callback to handle DB insertion using the pooled connection."""
    db_pool = get_db_pool()
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO litmus_results (
                    timestamp, spray_cell, chemical_type,
                    pass_fail, mean_width, mean_deflection, output_image
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                data,
            )
        conn.commit()
        st.toast("Record saved to database!", icon="✅")
    except Exception as e:
        if conn:
            conn.rollback()
        st.error(f"Database error: {e}")
    finally:
        if conn:
            db_pool.putconn(conn)


@st.cache_resource
def initialize_db():
    db_pool = get_db_pool()
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS litmus_results (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ,
                    spray_cell INTEGER,
                    chemical_type TEXT,
                    pass_fail TEXT,
                    mean_width REAL,
                    mean_deflection REAL,
                    output_image BYTEA
                )
                """
            )
        conn.commit()
    finally:
        if conn:
            db_pool.putconn(conn)


initialize_db()


### Cached, memory-bounded image processing helpers
@st.cache_data(max_entries=15, ttl=1800, show_spinner=False)
def load_image_and_dpi(image_bytes: bytes):
    with Image.open(io.BytesIO(image_bytes)) as meta_img:
        input_dpi = meta_img.info.get("dpi", (200, 200))
        image = np.array(meta_img.convert("RGB"))
    return image, input_dpi


@st.cache_data(max_entries=15, ttl=1800, show_spinner="Detecting paper boundaries...")
def compute_auto_crop(image_bytes: bytes):
    image, _ = load_image_and_dpi(image_bytes)
    h, w = image.shape[:2]

    # Downscale 4x before SLIC segmentation (~16x less RAM and CPU)
    scale = 0.25
    small_img = rescale(
        image, scale, channel_axis=-1, anti_aliasing=True, preserve_range=True
    ).astype(np.uint8)

    segments = skimage.segmentation.slic(small_img, n_segments=3, convert2lab=True)
    unborder = skimage.segmentation.clear_border(labels=segments, buffer_size=0)
    mask = np.repeat((unborder > 0)[:, :, np.newaxis], 3, axis=2)
    m_image = np.ma.masked_where(~mask, small_img)

    tc, bc, lc, rc = 0, 0, 0, 0
    sh, sw = m_image.shape[:2]

    for row in range(0, sh):
        if not m_image.mask[row, :, :].all():
            tc = row
            break
    for row in range(sh - 1, 0, -1):
        if not m_image.mask[row, :, :].all():
            bc = sh - row
            break

    acceptance_rate = 0.2
    for col in range(0, sw):
        region = m_image.mask[tc : sh - bc, col, :]
        if np.ma.count(region) > 0 and (region.sum() / np.ma.count(region) <= acceptance_rate):
            lc = col
            break
    for col in range(sw - 1, 0, -1):
        region = m_image.mask[tc : sh - bc, col, :]
        if np.ma.count(region) > 0 and (region.sum() / np.ma.count(region) <= acceptance_rate):
            rc = sw - col
            break
    for row in range(sh - bc, 0, -1):
        region = m_image.mask[row, lc : sw - rc, :]
        if np.ma.count(region) > 0 and (region.sum() / np.ma.count(region) <= acceptance_rate):
            bc = sh - row
            break

    inv_scale = 1.0 / scale
    tc = min(int(round(tc * inv_scale)), h - 1)
    bc = min(int(round(bc * inv_scale)), h - 1)
    lc = min(int(round(lc * inv_scale)), w - 1)
    rc = min(int(round(rc * inv_scale)), w - 1)

    return tc, bc, lc, rc, unborder


@st.cache_data(max_entries=15, ttl=1800, show_spinner="Analyzing spray pattern...")
def analyze_litmus(
    image_bytes: bytes,
    tc: int,
    bc: int,
    lc: int,
    rc: int,
    edit_center: bool,
    m_thresh: int,
    image_dpi: float,
):
    image, _ = load_image_and_dpi(image_bytes)
    cropped_image = skimage.util.crop(image, ((tc, bc), (lc, rc), (0, 0)))

    lab_img = rgb2lab(cropped_image)
    gray_img = lab_img[:, :, 0]
    red_img = lab_img[:, :, 2]
    del lab_img

    g_thresh = m_thresh if edit_center else threshold_otsu(gray_img)

    mask = gray_img < g_thresh
    mask_top = int(0.35 * mask.shape[0])
    mask_bot = int(0.65 * mask.shape[0])
    mask[:mask_top] = 0
    mask[mask_bot:] = 0

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(mask, theta=tested_angles)
    _, angle, dist = hough_line_peaks(h, theta, d, num_peaks=1)
    angle = angle[0]
    slope = np.tan(angle + np.pi / 2)
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    cntr = (float(x0[0]), float(y0[0]), float(slope))

    y_end = cntr[2] * mask.shape[1] + cntr[1]
    mask_top = max(0, int(min(cntr[1], y_end) - 0.02 * mask.shape[0]))
    mask_bot = min(mask.shape[0], int(max(cntr[1], y_end) + 0.02 * mask.shape[0]))
    mask[:mask_top, :] = 0
    mask[mask_bot:, :] = 0

    r_thresh = threshold_otsu(red_img)
    red_masked = red_img + r_thresh * mask
    blr_img = ndi.uniform_filter(red_masked, size=0.2 * image_dpi)
    spray_mask = remove_small_holes(
        remove_small_objects(blr_img > r_thresh, image_dpi * 100),
        image_dpi * 200,
    )

    spray_contours = find_contours(spray_mask)
    if len(spray_contours) == 2:
        bot_x = spray_contours[1][:, 1]
        bot_y = spray_contours[1][:, 0]
        top_x = spray_contours[0][:, 1]
        top_y = spray_contours[0][:, 0]

        pix_in = image_dpi
        x = np.arange(0, cropped_image.shape[1] + 1, 1)
        bot_interp = intp(bot_x, bot_y, assume_sorted=False, fill_value="extrapolate")(x)
        top_interp = intp(top_x, top_y, assume_sorted=False, fill_value="extrapolate")(x)
        center = cntr[2] * x + cntr[1]

        spray_width = np.abs(top_interp - bot_interp) / pix_in
        bot_height = (bot_interp - center) / pix_in
        top_height = (center - top_interp) / pix_in
        apparent_center = np.mean([top_interp, bot_interp], axis=0)
        deflection = np.arctan(((center - apparent_center) / pix_in) / 1.0) * (180 / np.pi)

        results = pd.DataFrame(
            {
                "mean": [spray_width.mean(), top_height.mean(), bot_height.mean(), deflection.mean()],
                "σ": [spray_width.std(), top_height.std(), bot_height.std(), deflection.std()],
                "min.": [spray_width.min(), top_height.min(), bot_height.min(), deflection.min()],
                "max.": [spray_width.max(), top_height.max(), bot_height.max(), deflection.max()],
            },
            index=[
                "Spray Pattern Width (in.)",
                "Top Side Width (in.)",
                "Bottom Side Width (in.)",
                "Spray Cone Deflection (deg.)",
            ],
        )

        metrics = {
            "characterize": True,
            "cropped_shape": cropped_image.shape,
            "cntr": cntr,
            "x": x,
            "bot_x": bot_x,
            "bot_y": bot_y,
            "top_x": top_x,
            "top_y": top_y,
            "bot_interp": bot_interp,
            "top_interp": top_interp,
            "center": center,
            "spray_width": spray_width,
            "deflection": deflection,
            "results": results,
        }
    else:
        metrics = {
            "characterize": False,
            "cropped_shape": cropped_image.shape,
            "cntr": cntr,
            "spray_mask": spray_mask,
        }

    return cropped_image, metrics


@st.cache_data(max_entries=10, ttl=1800, show_spinner=False)
def build_docx_report(
    disp: str,
    fig_jpeg_bytes: bytes,
    img_height_inches: float,
    edit_center: bool,
    m_thresh: int,
    image_dpi: float,
    results: pd.DataFrame,
    timestamp_str: str,
) -> bytes:
    doc = docx.Document()
    section = doc.sections[-1]
    section.left_margin = docx.shared.Inches(0.75)
    section.right_margin = docx.shared.Inches(0.75)
    section.top_margin = docx.shared.Inches(0.75)
    section.bottom_margin = docx.shared.Inches(0.75)

    title = doc.add_paragraph()
    title.paragraph_format.space_after = 0
    title_run = title.add_run("Litmus Test Characterization Report")
    title_run.font.size = docx.shared.Pt(18)
    title_run.font.name = "Arial"
    title_run.underline = True

    date = doc.add_paragraph()
    date_run = date.add_run(timestamp_str)
    date_run.font.size = docx.shared.Pt(10)
    date_run.font.name = "Arial"

    pf = doc.add_paragraph()
    pf_run = pf.add_run(disp)
    pf_run.font.color.rgb = (
        docx.shared.RGBColor(40, 166, 18) if disp == "PASS" else docx.shared.RGBColor(252, 3, 3)
    )
    pf_run.font.size = docx.shared.Pt(16)
    pf_run.font.name = "Arial"
    pf_run.bold = True

    doc.add_picture(io.BytesIO(fig_jpeg_bytes), height=docx.shared.Inches(img_height_inches))

    if edit_center:
        pic_p = doc.paragraphs[-1]
        pic_p.paragraph_format.space_after = 0
        center_p = doc.add_paragraph()
        center_p.paragraph_format.space_before = 0
        center_run = center_p.add_run(f"Centerline threshold adjusted to {m_thresh}")
        center_run.font.size = docx.shared.Pt(8)
        center_run.font.name = "Arial"

    if image_dpi != 200:
        dpi_p = doc.paragraphs[-1]
        dpi_p.paragraph_format.space_after = 0
        center_p = doc.add_paragraph()
        center_p.paragraph_format.space_before = 0
        center_run = center_p.add_run(
            f"Image resolution did not match expected 200 DPI. Input image resolution was {image_dpi}. "
            "This may affect dimensional measurements."
        )
        center_run.font.size = docx.shared.Pt(8)
        center_run.font.name = "Arial"

    table = doc.add_table(results.shape[0] + 1, results.shape[1] + 1)
    for i in range(results.shape[0]):
        table.cell(i + 1, 0).text = results.index[i]
        table.cell(i + 1, 0)._tc.tcPr.tcW.type = "auto"
    for j in range(results.shape[-1]):
        table.cell(0, j + 1).text = results.columns[j]
        table.cell(0, j + 1)._tc.tcPr.tcW.type = "auto"
    for i in range(results.shape[0]):
        for j in range(results.shape[-1]):
            table.cell(i + 1, j + 1).text = f"{results.values[i, j]:.3f}"
            table.cell(i + 1, j + 1)._tc.tcPr.tcW.type = "auto"
    table.style = "Table Grid"
    table.style.font.name = "Arial"
    table.allow_autofit = True

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


### Styling
st.markdown(
    """
    <style>
    footer,
    .stDeployButton,
    .stToolbar {
        display: none;
    }
    ::-webkit-scrollbar {
        width: 5px;
    }
    .block-container {
        padding-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

### Sidebar Controls
with st.sidebar:
    st.subheader("Test Inputs")

    spray_cell = st.selectbox(
        "Spray Cell Number",
        options=["-- Select --", 1, 2, 3, 4],
        index=0,
    )

    chemical_type = st.selectbox(
        "Chemical Type",
        options=["-- Select --", "M1b", "Glass", "MALP"],
        index=0,
    )

    uploaded_image = st.file_uploader(
        label="Upload Image",
        type=["jpg"],
        label_visibility="collapsed",
    )
    st.divider()
    st.subheader("Image Settings")
    auto_crop = st.checkbox("Auto-crop (beta)", value=True)
    if not auto_crop:
        tc = int(st.number_input("Top edge crop", value=25, step=5))
        bc = int(st.number_input("Bottom edge crop", value=1215, step=5))
        lc = int(st.number_input("Left edge crop", value=960, step=5))
        rc = int(st.number_input("Right edge crop", value=25, step=5))
        crop_done = st.checkbox("Done Cropping", value=False)
    else:
        tc, bc, lc, rc = 0, 0, 0, 0
        crop_done = False

    edit_center = st.checkbox("Adjust centerline identification", value=False)
    m_thresh = (
        st.slider(label="Centerline threshold", min_value=30, max_value=60, value=45)
        if edit_center
        else 45
    )

    if auto_crop:
        st.markdown("\n\n")
        st.markdown(
            ":grey[NOTE: The new auto-crop feature is still being tested and will likely not be completely reliable. "
            "This feature is very sensitive to the condition of the edges of the litmus paper. Straight, clean cut edges will produce the best result.]"
        )

image_dpi = 200

### Page Header & Tabs
st.header("Auto-Oxi Litmus Test Characterization")
tab1, tab2 = st.tabs(["Litmus Test", "Database Dashboard"])

with tab1:
    if spray_cell == "-- Select --" or chemical_type == "-- Select --":
        st.warning("Please select both Spray Cell Number and Chemical Type before uploading an image.")
    else:
        inst_col, m1, crop_col = st.columns([0.28, 0.02, 0.7])
        with crop_col:
            img_col, m2, val_col = st.columns([0.5, 0.01, 0.49])

        with inst_col:
            st.markdown(
                """Instructions:
                \n1. After the litmus paper has dried, scan the paper using a .jpg file type and an image resolution of 200 dpi.
                \n2. Upload the image to the "File Input" area on the left sidebar.
                \n3. Crop the image so that the edges of the litmus paper are just out of frame. If cropping manually, check the "Done Cropping" box to proceed with analysis.
                \n4. Confirm that the white dashed line correctly aligns with the centerline drawn on the litmus paper. 
                If it does not align, check
