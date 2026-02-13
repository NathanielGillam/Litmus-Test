### Import supporting libraries
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
from PIL import Image
from PIL.ExifTags import TAGS

### Configure streamlit webpage title, icon, and page width
st.set_option("client.toolbarMode", "minimal")
st.set_page_config(
    page_title = 'AO Litmus Test Analysis',
    page_icon = 'Active',
    layout = 'wide'
)

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
        </style>
        """, unsafe_allow_html=True)
  
### Create the page sidebar with most of the user's controls
with st.sidebar:
    uploaded_image = st.file_uploader(label = '', type = ['jpg'], label_visibility = 'collapsed')
    st.divider()
    st.subheader('Image Settings')
    auto_crop = st.checkbox('Auto-crop (beta)', value = True)
    if not auto_crop:
        tc = st.number_input('Top edge crop', value = 25, step = 5)
        bc = st.number_input('Bottom edge crop', value = 1215, step = 5)
        lc = st.number_input('Left edge crop', value = 960, step = 5)
        rc = st.number_input('Right edge crop', value = 25, step = 5)
        crop_done = st.checkbox('Done Cropping', value = False)
    else:
        # Initialize crop_done here to prevent NameError if auto_crop is checked
        crop_done = False 
    
    edit_center = st.checkbox('Adjust centerline identification', value = False)
    if edit_center:
        m_thresh = st.slider(label = 'Centerline threshold', min_value = 30, max_value = 60, value = 45)

    if auto_crop:
        st.markdown(body = "\n\n")
        st.markdown(body = """:grey[NOTE: The new auto-crop feature is still being tested and will likely not be completely reliable.]""")

### Default baseline (will be updated by metadata)
image_dpi = 96

### Create a page header 
st.header('Auto-Oxi Litmus Test Characterization')
inst_col, m1, crop_col = st.columns([0.28, 0.02, 0.7])
with crop_col:
    img_col, m2, val_col = st.columns([0.5, 0.01, 0.49])

with inst_col:
    st.markdown(
        body = """Instructions:
        \n1. Scan at **96** or **200 dpi**.
        \n2. Upload image.
        \n3. Crop and confirm centerline alignment.
        \n4. Results and Download Report will appear on the right.""")
    st.divider()

with crop_col:
    crop_placeholder = st.empty()
with img_col:
    image_placeholder = st.empty()
with val_col:
    values_placeholder = st.empty()

############################
# Image processing section
############################

# 1. Initialize logic flags
crop_done = False
characterize = False

if uploaded_image is not None:
    image = skimage.io.imread(uploaded_image)
    h, w, _ = image.shape

    # 2. Initialize crop coordinates with safe defaults (no crop)
    tc, bc, lc, rc = 0, 0, 0, 0

    # 3. Handle DPI detection
    meta_img = Image.open(uploaded_image)
    input_dpi = meta_img.info.get('dpi', (96, 96))
    image_dpi = input_dpi[0]
    pix_in = image_dpi

    if not (int(image_dpi) in [96, 200]):
        st.warning(f'Detection: {image_dpi} DPI. Expected 96 or 200.', icon="⚠️")

    # 4. Auto-Crop Logic
    if auto_crop:
        try:
            # Using Lightness to isolate the paper
            l_chan = rgb2lab(image)[:, :, 0]
            paper_mask = l_chan > threshold_otsu(l_chan)
            
            # Remove noise based on DPI
            clean_size = int(500 * (image_dpi/200))
            paper_mask = remove_small_objects(paper_mask, min_size=clean_size)
            paper_mask = remove_small_holes(paper_mask, area_threshold=clean_size)

            coords = np.column_stack(np.where(paper_mask))
            
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Assign crop values with a small 5px buffer
                tc = int(max(0, y_min + 5))
                bc = int(max(0, h - y_max + 5))
                lc = int(max(0, x_min + 5))
                rc = int(max(0, w - x_max + 5))
                
                crop_done = True
            else:
                st.error("Auto-crop: Could not find paper. Try manual mode.")
        except Exception as e:
            st.error(f"Auto-crop failed: {e}")
    else:
        # Manual mode: get values from sidebar number_inputs
        # Note: tc, bc, lc, rc must match the variable names in your sidebar
        crop_done = st.session_state.get('done_cropping', False) # Use checkbox from sidebar

    # 5. Execute Crop
    try:
        cropped_image = skimage.util.crop(image, ((tc, bc), (lc, rc), (0,0)))
    except Exception as e:
        st.error(f"Crop Error: {e}. Resetting to full image.")
        cropped_image = image
        tc, bc, lc, rc = 0, 0, 0, 0

    # 6. Display Crop Preview (Red Lines)
    extend = 20
    fig_crop, ax_c = plt.subplots()
    ax_c.imshow(image)
    ax_c.set_xlim(0, w)
    ax_c.set_ylim(h, 0)
    # Draw red boundaries
    ax_c.axhline(tc, color='red', lw=1, alpha=0.6)
    ax_c.axhline(h-bc, color='red', lw=1, alpha=0.6)
    ax_c.axvline(lc, color='red', lw=1, alpha=0.6)
    ax_c.axvline(w-rc, color='red', lw=1, alpha=0.6)
    
    # If we are in manual mode and haven't clicked "Done", show the red lines
    if not auto_crop and not crop_done:
        with crop_placeholder.container():
            st.pyplot(fig_crop)
    elif auto_crop and not crop_done:
        # If auto-crop failed, show the image so user can see why
        with crop_placeholder.container():
            st.pyplot(fig_crop)

############################
# Analysis section
############################

if crop_done:
    # Use 'a' channel for red spray detection (LAB index 1)
    # This is much more robust than 'b' or Grayscale for pink/red spray
    lab_img = rgb2lab(cropped_image)
    red_img = lab_img[:,:,1] 
    
    # Auto-threshold for the spray
    r_thresh = threshold_otsu(red_img)
    
    # Scale cleaning filters to DPI
    blur_size = int(0.2 * image_dpi)
    blr_img = ndi.uniform_filter(red_img, size=blur_size)
    
    # Define "Droplet" vs "Spray" size by physical inches
    obj_min = int((0.1 * image_dpi)**2) 
    hole_min = int((0.2 * image_dpi)**2)

    spray_mask = remove_small_holes(remove_small_objects(blr_img > r_thresh, obj_min), hole_min)
    
    

    spray_contours = find_contours(spray_mask)
    
    if len(spray_contours) == 2:
        # Sort by Y-coordinate to ensure index 0 is Top and index 1 is Bottom
        spray_contours = sorted(spray_contours, key=lambda c: np.mean(c[:,0]))
        top_y, top_x = spray_contours[0][:,0], spray_contours[0][:,1]
        bot_y, bot_x = spray_contours[1][:,0], spray_contours[1][:,1]
        characterize = True
    else:
        characterize = False
        with image_placeholder.container():
            st.error(f"FAIL: Detected {len(spray_contours)} edges. Need exactly 2.")
            # Show the mask to the user so they can see what went wrong
            st.image(spray_mask.astype(float), caption="Debug: Spray Mask")
