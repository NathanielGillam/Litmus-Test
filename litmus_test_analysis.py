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
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import find_contours

### Configure streamlit webpage title, icon, and page width
st.set_option("client.toolbarMode", "minimal")
st.set_page_config(
    page_title = 'AO Litmus Test Analysis',
    page_icon = 'Active',
    layout = 'wide'
)

# Custom CSS to clean up UI
st.markdown(
    """
    <style>
    footer, .stDeployButton, .stToolbar { display: none; }
    .block-container { padding-top: 0.5rem; }
    ::-webkit-scrollbar { width: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

### Create the page sidebar
with st.sidebar:
    uploaded_image = st.file_uploader(label = '', type = ['jpg'], label_visibility = 'collapsed')
    st.divider()
    st.subheader('Image Settings')
    auto_crop = st.checkbox('Auto-crop (beta)', value = True)
    
    # Placeholder for dynamic crop inputs
    if not auto_crop and uploaded_image is not None:
        # We define these later after the image is loaded to get dimensions
        pass
    
    edit_center = st.checkbox('Adjust centerline identification', value = False)
    if edit_center:
        m_thresh = st.slider(label = 'Centerline threshold', min_value = 30, max_value = 60, value = 45)

    if auto_crop:
        st.markdown(body = "\n\n")
        st.markdown(body = """:grey[NOTE: Auto-crop is optimized for 96-200 DPI.]""")

### Page Header and Columns
st.header('Auto-Oxi Litmus Test Characterization')
inst_col, m1, crop_col = st.columns([0.28, 0.02, 0.7])

with inst_col:
    st.markdown("""
        **Instructions:**
        1. Scan dry litmus paper at **96 DPI** or **200 DPI**.
        2. Upload the .jpg file.
        3. Confirm cropping and centerline alignment (white dashed line).
        4. Download the generated report.
    """)
    st.divider()

with crop_col:
    crop_placeholder = st.empty()
    img_col, m2, val_col = st.columns([0.5, 0.01, 0.49])

with img_col:
    image_placeholder = st.empty()
with val_col:
    values_placeholder = st.empty()

############################
# Image processing section
############################

if uploaded_image is not None:
    image = skimage.io.imread(uploaded_image)
    h, w, _ = image.shape

    ### DPI IDENTIFICATION
    meta_img = Image.open(uploaded_image)
    # Default to 96 if metadata is missing
    raw_dpi = meta_img.info.get('dpi', (96, 96))
    pix_in = raw_dpi[0]

    # Check for non-standard DPI (anything not 96 or 200)
    if int(pix_in) not in [96, 200]:
        st.warning(f'Detection: {pix_in} DPI. Results may vary outside 96/200 DPI range.', icon="⚠️")
    else:
        st.info(f'Resolution: {int(pix_in)} DPI')

    ### DYNAMIC CROP INPUTS (If manual)
    if not auto_crop:
        with st.sidebar:
            tc = st.number_input('Top edge crop (px)', value = int(h*0.02), step = 5)
            bc = st.number_input('Bottom edge crop (px)', value = int(h*0.02), step = 5)
            lc = st.number_input('Left edge crop (px)', value = int(w*0.25), step = 5)
            rc = st.number_input('Right edge crop (px)', value = int(w*0.02), step = 5)
            crop_done = st.checkbox('Done Cropping', value = False)
    else:
        crop_done = False

    ### Auto-crop function
    if auto_crop:
        try:
            segments = skimage.segmentation.slic(image, n_segments = 3, convert2lab = True)
            unborder = skimage.segmentation.clear_border(labels = segments, buffer_size = 0)
            mask = unborder > 0
            mask_3d = np.stack([mask, mask, mask], axis = 2)
            m_image = np.ma.masked_where(~mask_3d, image)

            tc, bc, lc, rc = 0,0,0,0
            # Detect top/bottom
            for row in range(0, m_image.shape[0]):
                if not m_image.mask[row,:,:].all():
                    tc = row
                    break
            for row in range(m_image.shape[0]-1, 0, -1):
                if not m_image.mask[row,:,:].all():
                    bc = m_image.shape[0] - row
                    break
            
            # Detect sides with tolerance
            acc_rate = 0.2
            for col in range(0, m_image.shape[1]):
                if m_image.mask[tc:h-bc, col, :].sum() / (3 * (h-bc-tc)) <= acc_rate:
                    lc = col
                    break
            for col in range(m_image.shape[1]-1, 0, -1):
                if m_image.mask[tc:h-bc, col, :].sum() / (3 * (h-bc-tc)) <= acc_rate:
                    rc = m_image.shape[1] - col
                    break
            
            if all([tc, bc, lc, rc]):
                crop_done = True
            else:
                st.error("Auto-crop failed to find edges. Switch to manual.")
                st.stop()
        except:
            st.error('Auto-crop error. Please use manual cropping.')
            st.stop()

    ### Perform Crop
    cropped_image = skimage.util.crop(image, ((tc, bc), (lc, rc), (0,0)))

    # Visualization for cropping confirmation
    crop_check, ax_crop = plt.subplots()
    ax_crop.imshow(image)
    # Draw red crop boundary lines
    ax_crop.axhline(tc, color='red', lw=1)
    ax_crop.axhline(h-bc, color='red', lw=1)
    ax_crop.axvline(lc, color='red', lw=1)
    ax_crop.axvline(w-rc, color='red', lw=1)
    
    if not crop_done:
        with crop_placeholder.container():
            st.pyplot(crop_check)

############################
# Analysis section
############################

if crop_done:
    # 1. Centerline Identification (LAB Lightness)
    gray_img = rgb2lab(cropped_image)[:,:,0]
    g_thresh = m_thresh if edit_center else threshold_otsu(gray_img)

    mask_center = gray_img < g_thresh
    m_h, m_w = mask_center.shape
    mask_center[:int(0.35*m_h)] = 0
    mask_center[int(0.65*m_h):] = 0
    
    tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint = False)
    h_trans, theta, d = hough_line(mask_center, theta = tested_angles)
    _, angle, dist = hough_line_peaks(h_trans, theta, d, num_peaks = 1)
    
    angle = angle[0]
    slope = np.tan(angle + np.pi/2)
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    cntr = (x0, y0, slope)

    # 2. Spray Detection (LAB 'a' channel for Reds)
    # Using 'a' channel (index 1) is often better for red/magenta spray than 'b'
    red_img = rgb2lab(cropped_image)[:,:,1] 
    r_thresh = threshold_otsu(red_img)
    
    # Blur scaled by DPI
    blur_radius = int(0.2 * pix_in)
    blr_img = ndi.uniform_filter(red_img, size = blur_radius)
    
    # Morphological cleaning scaled by DPI
    # A 100px hole at 200DPI is 0.5 inches; at 96DPI it's ~1 inch. 
    # We use area = (physical_inch * pix_in)^2
    obj_min = int((0.15 * pix_in)**2) 
    hole_min = int((0.25 * pix_in)**2)

    spray_mask = remove_small_holes(remove_small_objects(blr_img > r_thresh, obj_min), hole_min)
    spray_contours = find_contours(spray_mask)

    if len(spray_contours) == 2:
        # Sort contours by Y position to distinguish top vs bottom
        spray_contours = sorted(spray_contours, key=lambda c: np.mean(c[:,0]))
        top_y, top_x = spray_contours[0][:,0], spray_contours[0][:,1]
        bot_y, bot_x = spray_contours[1][:,0], spray_contours[1][:,1]
        characterize = True
    else:
        characterize = False

    #####################
    # Measurement Section
    #####################
    if characterize:
        x_range = np.arange(0, cropped_image.shape[1]+1, 1)
        top_interp = intp(top_x, top_y, assume_sorted=False, fill_value='extrapolate')(x_range)
        bot_interp = intp(bot_x, bot_y, assume_sorted=False, fill_value='extrapolate')(x_range)
        center_line = cntr[2] * x_range + cntr[1]

        spray_width = abs(top_interp - bot_interp) / pix_in
        top_height = (center_line - top_interp) / pix_in
        bot_height = (bot_interp - center_line) / pix_in
        
        apparent_center = np.mean([top_interp, bot_interp], axis = 0)
        # Deflection calculated over 1 inch horizontal run
        deflection = np.arctan(((center_line - apparent_center)/pix_in) / 1) * (180/np.pi)

        # Plotting results
        fig_res, ax_res = plt.subplots(dpi=150)
        ax_res.imshow(cropped_image)
        ax_res.plot(x_range, top_interp, color='cyan', lw=1, label='Top Edge')
        ax_res.plot(x_range, bot_interp, color='magenta', lw=1, label='Bottom Edge')
        ax_res.axline((cntr[0], cntr[1]), slope=cntr[2], color='white', ls='--', lw=1.5)
        ax_res.set_axis_off()

        with image_placeholder.container():
            st.pyplot(fig_res)

        # Tables
        results = pd.DataFrame({
            'mean': [spray_width.mean(), top_height.mean(), bot_height.mean(), deflection.mean()],
            'σ': [spray_width.std(), top_height.std(), bot_height.std(), deflection.std()],
            'min.': [spray_width.min(), top_height.min(), bot_height.min(), deflection.min()],
            'max.': [spray_width.max(), top_height.max(), bot_height.max(), deflection.max()]
        }, index=['Spray Width (in)', 'Top Side (in)', 'Bottom Side (in)', 'Deflection (deg)'])

        with values_placeholder.container():
            st.subheader('Analysis Results')
            st.dataframe(results.style.format("{:.3f}"))

            # Pass/Fail
            if spray_width.mean() < 1 or spray_width.mean() > 2:
                st.error(f"FAIL: Mean width {spray_width.mean():.2f}in is outside 1-2in range.")
            elif abs(deflection.mean()) > 10:
                st.error(f"FAIL: Mean deflection {deflection.mean():.2f}° exceeds 10°.")
            else:
                st.success("PASS")

            # Report Generation (simplified for brevity)
            doc = docx.Document()
            doc.add_heading('Litmus Test Report', 0)
            doc.add_paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            doc.add_paragraph(f"Detected Resolution: {pix_in} DPI")
            
            # Save and provide download
            report_io = io.BytesIO()
            doc.save(report_io)
            st.download_button("Download Report", data=report_io.getvalue(), file_name="report.docx")
    else:
        st.error("Could not identify exactly two spray edges. Check image quality.")
