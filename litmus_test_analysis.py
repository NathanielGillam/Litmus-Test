### Import supporting libraries
import psycopg2
import ssl
import getpass
import pytz
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import docx
import io
import datetime
import scipy.ndimage as ndi
from scipy.interpolate import interp1d as intp
import skimage
from PIL import Image
from PIL.ExifTags import TAGS
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import find_contours

### Configure streamlit webpage title, icon, and page width
st.set_option("client.toolbarMode", "minimal")
st.set_page_config(
    page_title='AO Litmus Test Analysis',
    page_icon='Active',
    layout='wide'
)

### Configure Neon Connection String
NEON_CONN = st.secrets["NEON_CONN"]

### User information
logged_in_user = getpass.getuser()

### Timestamp
pst = pytz.timezone("America/Los_Angeles")


### Auto Connect to NEON
def get_connection():
    return psycopg2.connect(NEON_CONN, sslmode="require")


### Saving Helper Function
def save_test_to_db(data):
    """Callback to handle the DB insertion reliably."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO litmus_results (
                timestamp, spray_cell, chemical_type, 
                pass_fail, mean_width, mean_deflection, output_image
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, data)
        conn.commit()
        cur.close()
        conn.close()
        st.toast("Record saved to database!", icon="✅")
    except Exception as e:
        st.error(f"Database error: {e}")


### DB initialization (cached so it runs once instead of opening a connection on every rerun)
@st.cache_resource
def initialize_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
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
    """)
    conn.commit()
    cur.close()
    conn.close()


initialize_db()


### Bounded Cached Image Processing Helpers
@st.cache_data(max_entries=15, ttl=1800, show_spinner=False)
def load_image_and_dpi(image_bytes: bytes):
    image = skimage.io.imread(io.BytesIO(image_bytes))
    with Image.open(io.BytesIO(image_bytes)) as meta_img:
        meta_dict = meta_img.info
        input_dpi = meta_dict['dpi']
    return image, input_dpi


@st.cache_data(max_entries=15, ttl=1800, show_spinner="Detecting litmus paper boundaries...")
def run_auto_crop(image_bytes: bytes):
    image, _ = load_image_and_dpi(image_bytes)
    segments = skimage.segmentation.slic(image, n_segments=3, convert2lab=True)
    unborder = skimage.segmentation.clear_border(labels=segments, buffer_size=0)
    mask = unborder > 0
    mask = np.stack([mask, mask, mask], axis=2)
    m_image = np.ma.masked_where(~mask, image)

    ### First round of cropping
    tc, bc, lc, rc = 0, 0, 0, 0

    for row in range(0, m_image.shape[0]):
        if not m_image.mask[row, :, :].all():
            tc = row
            break
    for row in range(m_image.shape[0] - 1, 0, -1):
        if not m_image.mask[row, :, :].all():
            bc = m_image.shape[0] - row
            break

    acceptance_rate = 0.2

    for col in range(0, m_image.shape[1]):
        if m_image.mask[tc:m_image.shape[0] - bc, col, :].sum() / np.ma.count(m_image.mask[tc:m_image.shape[0] - bc, col, :]) <= acceptance_rate:
            lc = col
            break
    for col in range(m_image.shape[1] - 1, 0, -1):
        if m_image.mask[tc:m_image.shape[0] - bc, col, :].sum() / np.ma.count(m_image.mask[tc:m_image.shape[0] - bc, col, :]) <= acceptance_rate:
            rc = m_image.shape[1] - col
            break
    for row in range(m_image.shape[0] - bc, 0, -1):
        if m_image.mask[row, lc:m_image.shape[1] - rc, :].sum() / np.ma.count(m_image.mask[row, lc:m_image.shape[1] - rc, :]) <= acceptance_rate:
            bc = m_image.shape[0] - row
            break

    return tc, bc, lc, rc, unborder


@st.cache_data(max_entries=15, ttl=1800, show_spinner="Analyzing spray pattern...")
def analyze_spray_pattern(
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

    ### Convert the input image to the LAB colorspace and use the lightness channel to find dark pixels
    gray_img = rgb2lab(cropped_image)[:, :, 0]
    ### If the user chooses to edit the centerline threshold, use their input threshold
    if edit_center:
        g_thresh = m_thresh
    ### Otherwise, use the threshold_otsu function to automatically select a threshold based on the range of lightness values in the image.
    else:
        g_thresh = threshold_otsu(gray_img)

    ### Use masking to limit the area in which the centerline can be identified to a horizontal band in the middle of the image.
    mask = gray_img < g_thresh
    mask_top = int(0.35 * mask.shape[0])
    mask_bot = int(0.65 * mask.shape[0])
    mask[:mask_top] = 0
    mask[mask_bot:] = 0

    ### Find candidate straight lines using a straight line Hough transform.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(mask, theta=tested_angles)
    _, angle, dist = hough_line_peaks(h, theta, d, num_peaks=1)
    angle = angle[0]
    slope = np.tan(angle + np.pi / 2)
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    cntr = (float(x0[0]), float(y0[0]), float(slope))

    ### Use the red/green channel of the LAB colorspace image to identify the edges of the sprayed area.
    red_img = rgb2lab(cropped_image)[:, :, 2]
    ### Mask off the black pixels of the centerline to prevent it from being identified as a spray pattern edge
    y_end = cntr[2] * mask.shape[1] + cntr[1]
    mask_top = int(min(cntr[1], y_end) - 0.02 * mask.shape[0])
    mask_bot = int(max(cntr[1], y_end) + 0.02 * mask.shape[0])
    mask[:mask_top, :] = 0
    mask[mask_bot:, :] = 0
    r_thresh = threshold_otsu(red_img)
    red_masked = red_img + r_thresh * mask
    ### Blur the red filtered image by a constant value to deal with small droplets at the edge of the spray band.
    blr_img = ndi.uniform_filter(red_masked, size=0.2 * image_dpi)
    ### Use remove small holes and remove small objects to clean up small gaps and droplets.
    spray_mask = remove_small_holes(
        remove_small_objects(blr_img > r_thresh, image_dpi * 100),
        image_dpi * 200,
    )

    ### Use find contours to identify the pixels at the top and bottom edge of the spray band.
    spray_contours = find_contours(spray_mask)
    if len(spray_contours) == 2:
        bot_x = spray_contours[1][:, 1]
        bot_y = spray_contours[1][:, 0]
        top_x = spray_contours[0][:, 1]
        top_y = spray_contours[0][:, 0]
        characterize = True
    else:
        characterize = False

    pix_in = image_dpi

    if characterize:
        x = np.arange(0, cropped_image.shape[1] + 1, 1)
        bot_interp = intp(bot_x, bot_y, assume_sorted=False, fill_value='extrapolate')(x)
        top_interp = intp(top_x, top_y, assume_sorted=False, fill_value='extrapolate')(x)
        center = cntr[2] * x + cntr[1]
        spray_width = abs(top_interp - bot_interp) / pix_in
        bot_height = (bot_interp - center) / pix_in
        top_height = (center - top_interp) / pix_in
        apparent_center = np.mean([top_interp, bot_interp], axis=0)
        deflection = np.arctan(((center - apparent_center) / pix_in) / 1) * (180 / np.pi)

        results = pd.DataFrame()
        results.loc['Spray Pattern Width (in.)', 'mean'] = spray_width.mean()
        results.loc['Spray Pattern Width (in.)', 'σ'] = spray_width.std()
        results.loc['Spray Pattern Width (in.)', 'min.'] = spray_width.min()
        results.loc['Spray Pattern Width (in.)', 'max.'] = spray_width.max()
        results.loc['Top Side Width (in.)', 'mean'] = top_height.mean()
        results.loc['Top Side Width (in.)', 'σ'] = top_height.std()
        results.loc['Top Side Width (in.)', 'min.'] = top_height.min()
        results.loc['Top Side Width (in.)', 'max.'] = top_height.max()
        results.loc['Bottom Side Width (in.)', 'mean'] = bot_height.mean()
        results.loc['Bottom Side Width (in.)', 'σ'] = bot_height.std()
        results.loc['Bottom Side Width (in.)', 'min.'] = bot_height.min()
        results.loc['Bottom Side Width (in.)', 'max.'] = bot_height.max()
        results.loc['Spray Cone Deflection (deg.)', 'mean'] = deflection.mean()
        results.loc['Spray Cone Deflection (deg.)', 'σ'] = deflection.std()
        results.loc['Spray Cone Deflection (deg.)', 'min.'] = deflection.min()
        results.loc['Spray Cone Deflection (deg.)', 'max.'] = deflection.max()

        return {
            "cropped_image": cropped_image,
            "characterize": True,
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

    return {
        "cropped_image": cropped_image,
        "characterize": False,
        "cntr": cntr,
        "spray_mask": spray_mask,
    }


@st.cache_data(max_entries=10, ttl=1800, show_spinner=False)
def build_docx_report(
    disp: str,
    fig_png_bytes: bytes,
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
    title_run = title.add_run('Litmus Test Characterization Report')
    title_run.font.size = docx.shared.Pt(18)
    title_run.font.name = 'Arial'
    title_run.underline = True

    date = doc.add_paragraph()
    date_run = date.add_run(timestamp_str)
    date_run.font.size = docx.shared.Pt(10)
    date_run.font.name = 'Arial'

    pf = doc.add_paragraph()
    pf_run = pf.add_run(disp)
    if disp == 'PASS':
        pf_run.font.color.rgb = docx.shared.RGBColor(40, 166, 18)
    else:
        pf_run.font.color.rgb = docx.shared.RGBColor(252, 3, 3)
    pf_run.font.size = docx.shared.Pt(16)
    pf_run.font.name = 'Arial'
    pf_run.bold = True

    doc.add_picture(io.BytesIO(fig_png_bytes), height=docx.shared.Inches(img_height_inches))

    if edit_center:
        pic_p = doc.paragraphs[-1]
        pic_p.paragraph_format.space_after = 0
        center_p = doc.add_paragraph()
        center_p.paragraph_format.space_before = 0
        center_run = center_p.add_run('Centerline threshold adjusted to {}'.format(m_thresh))
        center_run.font.size = docx.shared.Pt(8)
        center_run.font.name = 'Arial'

    if image_dpi != 200:
        dpi_p = doc.paragraphs[-1]
        dpi_p.paragraph_format.space_after = 0
        center_p = doc.add_paragraph()
        center_p.paragraph_format.space_before = 0
        center_run = center_p.add_run(
            'Image resolution did not match expected 200 DPI. Input image resolution was {}. This may affect dimensional measurements.'.format(
                image_dpi
            )
        )
        center_run.font.size = docx.shared.Pt(8)
        center_run.font.name = 'Arial'

    table = doc.add_table(results.shape[0] + 1, results.shape[1] + 1)
    for i in range(results.shape[0]):
        table.cell(i + 1, 0).text = results.index[i]
        table.cell(i + 1, 0)._tc.tcPr.tcW.type = 'auto'
    for j in range(results.shape[-1]):
        table.cell(0, j + 1).text = results.columns[j]
        table.cell(0, j + 1)._tc.tcPr.tcW.type = 'auto'
    for i in range(results.shape[0]):
        for j in range(results.shape[-1]):
            table.cell(i + 1, j + 1).text = '{:.3f}'.format(results.values[i, j])
            table.cell(i + 1, j + 1)._tc.tcPr.tcW.type = 'auto'
    table.style = 'Table Grid'
    table.style.font.name = 'Arial'
    table.allow_autofit = True

    download_object = io.BytesIO()
    doc.save(download_object)
    return download_object.getvalue()


st.markdown(
    """
    <style>
    footer,
    .stDeployButton,
    .stToolbar {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


### Force the scrollbar to be a constant width.
st.markdown("""
                <html>
                    <head>
                    <style>
                        ::-webkit-scrollbar {
                            width: 5px;
                            }
                    </style>
                    </head>
                    <body>
                    </body>
                </html>
            """, unsafe_allow_html=True)

### Revise the size of the margin at the top of the page.
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0.5rem;
               }
        </style>
        """, unsafe_allow_html=True)

### Create the page sidebar with most of the user's controls
with st.sidebar:
    st.subheader("Test Inputs")

    spray_cell = st.selectbox(
        "Spray Cell Number",
        options=["-- Select --", 1, 2, 3, 4],
        index=0
    )

    chemical_type = st.selectbox(
        "Chemical Type",
        options=["-- Select --", "M1b", "Glass", "MALP"],
        index=0
    )

    uploaded_image = st.file_uploader(
        label="Upload Image",
        type=["jpg"],
        label_visibility="collapsed"
    )
    st.divider()
    st.subheader('Image Settings')
    auto_crop = st.checkbox('Auto-crop (beta)', value=True)
    if not auto_crop:
        tc = int(st.number_input('Top edge crop', value=25, step=5))
        bc = int(st.number_input('Bottom edge crop', value=1215, step=5))
        lc = int(st.number_input('Left edge crop', value=960, step=5))
        rc = int(st.number_input('Right edge crop', value=25, step=5))
        crop_done = st.checkbox('Done Cropping', value=False)
    else:
        tc, bc, lc, rc = 0, 0, 0, 0
        crop_done = False

    edit_center = st.checkbox('Adjust centerline identification', value=False)
    if edit_center:
        m_thresh = st.slider(label='Centerline threshold', min_value=30, max_value=60, value=45)
    else:
        m_thresh = 45

    if auto_crop:
        st.markdown(body="\n\n")
        st.markdown(body=""":grey[NOTE: The new auto-crop feature is still being tested and will likely not be completely reliable.
                    This feature is very sensitive to the condition of the edges of the litmus paper. Straight, clean cut edges will produce the best result.]""")

### Set the standard image resolution used to 200 dpi
image_dpi = 200

### Create a page header and separate the page into columns for instructions, image results, and measurement results.
st.header('Auto-Oxi Litmus Test Characterization')
inst_col, m1, crop_col = st.columns([0.28, 0.02, 0.7])
with crop_col:
    img_col, m2, val_col = st.columns([0.5, 0.01, 0.49])

tab1, tab2 = st.tabs(["Litmus Test", "Database Dashboard"])

with tab1:
    ### Always write out instructions on the use of the tool in the instruction column of the page.
    with inst_col:
        st.markdown(
            body="""Instructions:
            \n1. After the litmus paper has dried, scan the paper using a .jpg file type and an image resolution of 200 dpi.
            \n2. Upload the image to the "File Input" area on the left sidebar.
            \n3. Crop the image so that the edges of the litmus paper are just out of frame. If cropping manually, check the "Done Cropping" box to proceed with analysis.
            \n4. Confirm that the white dashed line correctly aligns with the centerline drawn on the litmus paper. 
            If it does not align, check the "Adjust centerline identification" box in the "Image Settings" area on the left sidebar and adjust the threshold until the centerlines align correctly.
            \n5. Dimensional characterization results will be shown to the right of the litmus paper image. Use the "Download Report" button to generate a file documenting the analysis results."""
        )
        st.divider()

    ### Create placeholders for the image during cropping, the resultant image, and the summarized measurements
    with crop_col:
        crop_placeholder = st.empty()
    with img_col:
        image_placeholder = st.empty()
    with val_col:
        values_placeholder = st.empty()

    # Require user selections before proceeding with image analysis
    if spray_cell == "-- Select --" or chemical_type == "-- Select --":
        with crop_placeholder.container():
            st.warning("Please select both Spray Cell Number and Chemical Type before uploading an image.")
    else:
        ############################
        # Image processing section
        ############################
        if uploaded_image is not None:
            image_bytes = uploaded_image.getvalue()
            image, input_dpi = load_image_and_dpi(image_bytes)

            ### Actions taken if the image DPI does not match the expected value of 200
            if not ((input_dpi[0] == 200) & (input_dpi[1] == 200)):
                disp = 'WARNING: Input image resolution did not match the expected resolution of 200 DPI. The input image resolution is {}.'.format(input_dpi)
                st.error(disp, icon="⚠️")
                if input_dpi[0] == input_dpi[1]:
                    image_dpi = input_dpi[0]

            auto_crop_failed = False
            unborder = None

            ### Auto-crop function
            if auto_crop:
                try:
                    tc, bc, lc, rc, unborder = run_auto_crop(image_bytes)
                except Exception:
                    auto_crop_failed = True

                if auto_crop_failed or not (tc > 0) & (bc > 0) & (lc > 0) & (rc > 0):
                    with crop_placeholder.container():
                        st.error('Auto-crop was unable to identify the boundaries of the litmus paper in the submitted image. Please use manual cropping.', icon="⚠️")
                    crop_done = False
                else:
                    crop_done = True

            ### If the user has not checked the "Done Cropping" box, show the crop_check image
            if not crop_done and not auto_crop_failed:
                extend = 20
                crop_check, ax = plt.subplots()
                try:
                    if auto_crop and unborder is not None:
                        ax.imshow(skimage.segmentation.mark_boundaries(image, unborder, mode='thick', color=(1, 0.1, 0.6)))
                    else:
                        ax.imshow(image)
                    ax.set_xlim(0, image.shape[1])
                    ax.set_ylim(image.shape[0], 0)
                    ax.tick_params(labelsize=6)
                    ax.plot([lc - extend, image.shape[1] - rc + extend], [image.shape[0] - bc, image.shape[0] - bc], color='red', lw=0.8, alpha=0.6)
                    ax.plot([lc - extend, image.shape[1] - rc + extend], [tc, tc], color='red', lw=0.8, alpha=0.6)
                    ax.plot([lc, lc], [image.shape[0] - bc + extend, tc - extend], color='red', lw=0.8, alpha=0.6)
                    ax.plot([image.shape[1] - rc, image.shape[1] - rc], [image.shape[0] - bc + extend, tc - extend], color='red', lw=0.8, alpha=0.6)

                    with crop_placeholder.container():
                        st.pyplot(crop_check)
                finally:
                    plt.close(crop_check)

            ### If the user has checked the "Done Cropping" box, proceed with image analysis.
            if crop_done:
                analysis = analyze_spray_pattern(
                    image_bytes=image_bytes,
                    tc=tc,
                    bc=bc,
                    lc=lc,
                    rc=rc,
                    edit_center=edit_center,
                    m_thresh=m_thresh,
                    image_dpi=image_dpi,
                )

                cropped_image = analysis["cropped_image"]
                characterize = analysis["characterize"]
                cntr = analysis["cntr"]
                pix_in = image_dpi

                ### Generate the resultant image showing the identified centerline and spray edges overlaid on the cropped spray pattern image.
                fig, ax = plt.subplot_mosaic("A", dpi=200)
                try:
                    ax['A'].imshow(cropped_image)
                    ax['A'].set_xticks([])
                    ax['A'].set_yticks([])

                    ### If two spray pattern edges were successfully identified...
                    if characterize:
                        x = analysis["x"]
                        bot_x = analysis["bot_x"]
                        bot_y = analysis["bot_y"]
                        top_x = analysis["top_x"]
                        top_y = analysis["top_y"]
                        bot_interp = analysis["bot_interp"]
                        top_interp = analysis["top_interp"]
                        center = analysis["center"]
                        spray_width = analysis["spray_width"]
                        deflection = analysis["deflection"]
                        results = analysis["results"]

                        ax['A'].plot(bot_x, bot_y, color='black', lw=1)
                        ax['A'].plot(top_x, top_y, color='black', lw=1)

                        ### Use a white outward pointing arrow to identify the maximum spray pattern width.
                        max_loc = np.argmax(spray_width)
                        ax['A'].scatter([x[max_loc]], [bot_interp[max_loc]], marker=7, color='white', zorder=4, alpha=0.5, clip_on=False)
                        ax['A'].scatter([x[max_loc]], [top_interp[max_loc]], marker=6, color='white', zorder=4, alpha=0.5, clip_on=False)
                        ax['A'].plot([x[max_loc], x[max_loc]], [bot_interp[max_loc], top_interp[max_loc]], color='white', lw=1, ls=(0, (2, 2)), alpha=0.5)

                        ### Use a white inward pointing arrow to identify the minimum spray pattern width.
                        min_loc = np.argmin(spray_width)
                        ax['A'].scatter([x[min_loc]], [bot_interp[min_loc]], marker=6, color='white', zorder=4, alpha=0.5, clip_on=False)
                        ax['A'].scatter([x[min_loc]], [top_interp[min_loc]], marker=7, color='white', zorder=4, alpha=0.5, clip_on=False)
                        ax['A'].plot([x[min_loc], x[min_loc]], [bot_interp[min_loc], top_interp[min_loc]], color='white', lw=1, ls=(0, (2, 2)), alpha=0.5)

                        ### Use a red outward pointing arrow to identify the maximum deflection and its direction.
                        max_loc = np.argmax(deflection)
                        if deflection.max() > 0:
                            ax['A'].scatter([x[max_loc]], [top_interp[max_loc]], marker=6, color='red', zorder=4, alpha=0.5, clip_on=False)
                            ax['A'].plot([x[max_loc], x[max_loc]], [center[max_loc], top_interp[max_loc]], color='red', lw=1, ls=(2, (2, 2)), alpha=0.5)
                        else:
                            ax['A'].scatter([x[max_loc]], [bot_interp[max_loc]], marker=6, color='red', zorder=4, alpha=0.5, clip_on=False)
                            ax['A'].plot([x[max_loc], x[max_loc]], [center[max_loc], bot_interp[max_loc]], color='red', lw=1, ls=(2, (2, 2)), alpha=0.5)

                        ### Use a red inward pointing arrow to identify the minimum deflection and its direction.
                        min_loc = np.argmin(deflection)
                        if deflection.min() > 0:
                            ax['A'].scatter([x[min_loc]], [top_interp[min_loc]], marker=7, color='red', zorder=4, alpha=0.5, clip_on=False)
                            ax['A'].plot([x[min_loc], x[min_loc]], [center[min_loc], top_interp[min_loc]], color='red', lw=1, ls=(2, (2, 2)), alpha=0.5)
                        else:
                            ax['A'].scatter([x[min_loc]], [bot_interp[min_loc]], marker=7, color='red', zorder=4, alpha=0.5, clip_on=False)
                            ax['A'].plot([x[min_loc], x[min_loc]], [center[min_loc], bot_interp[min_loc]], color='red', lw=1, ls=(2, (2, 2)), alpha=0.5)
                    else:
                        ax['A'].contour(analysis["spray_mask"], colors='black')

                    ### Use a white dashed line to mark the identified centerline.
                    ax['A'].axline((cntr[0], cntr[1]), slope=cntr[2], ls=(0, (4, 4)), color='white', lw=1.5)
                    ax['A'].set_xlim(0, cropped_image.shape[1])
                    ax['A'].set_ylim(cropped_image.shape[0], 0)

                    ### Save the resultant image to memory for later access when generating a report.
                    fig_mem = io.BytesIO()
                    fig.savefig(fig_mem, format='png', bbox_inches='tight', pad_inches=0)
                    fig_png_bytes = fig_mem.getvalue()

                    ### Show the final image in the image placeholder container location.
                    with image_placeholder.container():
                        st.pyplot(fig)
                finally:
                    plt.close(fig)

                ### Display the results table in the values placeholder container location.
                with values_placeholder.container():
                    if characterize:
                        st.subheader('Analysis Results')
                        st.dataframe(results)

                        ##################################
                        # Spray Pattern Pass/Fail Criteria
                        ##################################
                        if np.isnan(spray_width.mean()):
                            disp = 'FAIL: spray pattern edge detection unsuccessful'
                            st.write("#### :red[{}]".format(disp))
                        elif spray_width.mean() < 1:
                            disp = 'FAIL: mean spray width must be >1 in.'
                            st.write("#### :red[{}]".format(disp))
                        elif spray_width.mean() > 2:
                            disp = 'FAIL: mean spray width must be <2 in.'
                            st.write('#### :red[{}]'.format(disp))
                        elif abs(deflection.mean()) > 10:
                            disp = 'FAIL: mean deflection angle must be <10°'
                            st.write("#### :red[{}]".format(disp))
                        elif (deflection.max() - deflection.min()) > 15:
                            disp = 'FAIL: deflection range too large'
                            st.write("#### :red[{}]".format(disp))
                        elif abs(deflection).max() > 18:
                            disp = 'FAIL: suspected drip inclusion'
                            st.write("#### :red[{}]".format(disp))
                        else:
                            disp = 'PASS'
                            st.write("### :green[{}]".format(disp))

                        ### Create a placeholder for the report download button under the measurement summary table.
                        button_place = st.empty()

                        ### Display the pass/fail criteria to the user
                        st.divider()
                        st.markdown(
                            body="""\nPass Criteria:
                            \n- Mean Width: >1 in. and <2 in.
                            \n- Mean Deflection: <10°
                            \n- Deflection Range (max - min): <15°
                            \n- Max. spray pattern deflection >18° indicates drip inclusion"""
                        )
                        st.divider()

                        #######################
                        # Output Result Section
                        #######################
                        current_time = datetime.datetime.now(pst)
                        docx_bytes = build_docx_report(
                            disp=disp,
                            fig_png_bytes=fig_png_bytes,
                            img_height_inches=cropped_image.shape[1] / pix_in,
                            edit_center=edit_center,
                            m_thresh=m_thresh,
                            image_dpi=image_dpi,
                            results=results,
                            timestamp_str=current_time.strftime('%B %d, %Y - %I:%M %p'),
                        )

                        ### Use the download button to kick off DB entry
                        with button_place.container():
                            db_data = (
                                datetime.datetime.now(pst),
                                spray_cell,
                                chemical_type,
                                disp,
                                float(spray_width.mean()),
                                float(deflection.mean()),
                                psycopg2.Binary(fig_png_bytes)
                            )

                            st.download_button(
                                label='Download Report',
                                data=docx_bytes,
                                file_name=f'litmus_test_report_{datetime.datetime.now().strftime("%m_%d_%y_%H_%M")}.docx',
                                mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                on_click=save_test_to_db,
                                args=(db_data,)
                            )
                    else:
                        disp = 'FAIL: spray pattern edge detection unsuccessful'
                        st.write("#### :red[{}]".format(disp))

with tab2:
    st.header("Database History")

    # Connect to DB and load minimal columns first
    st.subheader("Filters")

    st.subheader("Filters")
    
    # Spray Cell Filter
    filter_cell = st.selectbox(
        "Filter by Spray Cell",
        options=["All", 1, 2, 3, 4],
        index=0
    )
    
    # Chemical Type Filter
    filter_chem = st.selectbox(
        "Filter by Chemical Type",
        options=["All", "M1b", "Glass", "MALP"],
        index=0
    )
    
    # Date Filter
    filter_date = st.date_input(
        "Filter by Date (PST)",
        value=None
    )
    
    # Convert "All" to None so it doesn't break SQL
    cell_param = None if filter_cell == "All" else filter_cell
    chem_param = None if filter_chem == "All" else filter_chem
    date_param = filter_date if filter_date else None
    
    # SQL Query - Convert the UTC storage back to PST for display
    query = """
        SELECT 
            id, 
            timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles' AS timestamp, 
            spray_cell, 
            chemical_type,
            mean_width,
            pass_fail
        FROM litmus_results
        WHERE (%s IS NULL OR spray_cell = %s)
          AND (%s IS NULL OR chemical_type = %s)
          AND (%s IS NULL OR DATE(timestamp AT TIME ZONE 'America/Los_Angeles') = %s)
        ORDER BY timestamp DESC
        LIMIT 1000
    """
    
    params = (
        cell_param, cell_param,
        chem_param, chem_param,
        date_param, date_param
    )
    
    # Load filtered data
    conn = get_connection()
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if not df.empty:
        # Format the timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %I:%M %p')
        
        # Round the Mean Width to 3 decimal places for a cleaner table
        df['mean_width'] = df['mean_width'].map('{:,.3f}'.format)
        
        # Optional: Rename the column for the UI
        df = df.rename(columns={'mean_width': 'Mean Width (in)'})
    
    # Display table
    st.dataframe(df, use_container_width=True)

    st.subheader("View Record Details")

    record_id = st.number_input("Enter Record ID to view full details:", min_value=1, step=1)

    
    if st.button("Load Record"):
    
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                timestamp,
                spray_cell,
                chemical_type,
                pass_fail,
                mean_width,
                mean_deflection,
                output_image
            FROM litmus_results
            WHERE id = %s
        """, (record_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
    
        if row is None:
            st.error("Record not found.")
        else:
            timestamp, cell, chem, pf, mw, md, image_bytes = row

            # Convert the UTC timestamp from the database back to PST for display
            local_time = timestamp.astimezone(pst)
    
            st.write(f"**Timestamp:** {timestamp}")
            st.write(f"**Spray Cell:** {cell}")
            st.write(f"**Chemical Type:** {chem}")
            st.write(f"**Result:** {pf}")
            st.write(f"**Mean Width:** {mw:.3f} in")
            st.write(f"**Mean Deflection:** {md:.3f}°")
    
            if image_bytes:
                st.subheader("Stored Spray Image")
                # Convert memoryview → bytes
                img_bytes = bytes(image_bytes)
                # Now Streamlit can render it
                st.image(img_bytes, caption="Annotated Spray Pattern", width="stretch")
