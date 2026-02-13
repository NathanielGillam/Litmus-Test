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

# Initialize crop_done at the start to avoid logic errors
crop_done = False 

if uploaded_image != None:
    image = skimage.io.imread(uploaded_image)
    h, w, _ = image.shape # Get image dimensions for safety

    # Initialize crop variables with defaults to prevent NameError
    tc, bc, lc, rc = 0, 0, 0, 0

    ### Read input image metadata to identify the image resolution
    meta_img = Image.open(uploaded_image)
    input_dpi = meta_img.info.get('dpi', (96, 96))

    ### Update logic to accept 96 or 200 DPI
    if not (int(input_dpi[0]) in [96, 200]):
        disp = 'WARNING: Non-standard resolution detected ({}). Expected 96 or 200 DPI.'.format(input_dpi)
        st.error(disp, icon="⚠️")
    
    image_dpi = input_dpi[0]

    ### Auto-crop function (Improved for 96/200 DPI)
    if auto_crop:
        try:
            # Convert to LAB and use Lightness to find the paper
            l_chan = rgb2lab(image)[:, :, 0]
            # Threshold to create a binary mask of the paper
            paper_mask = l_chan > threshold_otsu(l_chan)
            
            # Clean up the mask relative to DPI
            paper_mask = remove_small_objects(paper_mask, min_size=int(500 * (image_dpi/200)))
            paper_mask = remove_small_holes(paper_mask, area_threshold=int(500 * (image_dpi/200)))

            # Find coordinates of all "True" pixels
            coords = np.column_stack(np.where(paper_mask))
            
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Assign crop values with a small safety buffer
                tc = int(y_min + 5)
                bc = int(h - y_max + 5)
                lc = int(x_min + 5)
                rc = int(w - x_max + 5)
                
                crop_done = True
            else:
                st.error("Auto-crop: No paper detected. Please use manual mode.")
        except Exception as e:
            st.error(f"Auto-crop failed: {e}")
    
    # If not auto-cropping, pull from the sidebar inputs defined at the top
    else:
        # These are already defined in your sidebar block
        # We ensure they are integers for the crop function
        tc, bc, lc, rc = int(tc), int(bc), int(lc), int(rc)
            
    ### Crop execution (Only if we have valid coordinates)
    # We use a try block here as a final safety net for the NameError
    try:
        cropped_image = skimage.util.crop(image, ((tc, bc), (lc, rc), (0,0)))
    except NameError:
        st.warning("Please adjust cropping settings to begin.")
        st.stop()

    extend = 20
    crop_check, ax_c = plt.subplots()
    ax_c.imshow(image)
    ax_c.set_xlim(0,w); ax_c.set_ylim(h,0)
    ax_c.plot([lc-extend,w-rc+extend],[h-bc,h-bc], color = 'red', lw = 0.8)
    ax_c.plot([lc-extend,w-rc+extend],[tc,tc], color = 'red', lw = 0.8)
    ax_c.plot([lc,lc],[h-bc+extend,tc-extend], color = 'red', lw = 0.8)
    ax_c.plot([w-rc,w-rc],[h-bc+extend,tc-extend], color = 'red', lw = 0.8)

    if not crop_done:
        with crop_placeholder.container():
            st.write(crop_check)

### Start analysis
if crop_done:
    gray_img = rgb2lab(cropped_image)[:,:,0]
    g_thresh = m_thresh if edit_center else threshold_otsu(gray_img)

    mask = gray_img < g_thresh
    mask[:int(0.35*mask.shape[0])] = 0
    mask[int(0.65*mask.shape[0]):] = 0
        
    tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint = False)
    h, theta, d = hough_line(mask, theta = tested_angles)
    _, angle, dist = hough_line_peaks(h, theta, d, num_peaks = 1)
    angle = angle[0]; slope = np.tan(angle + np.pi/2)
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    cntr = (x0, y0, slope)

    # Use 'a' channel for better red spray detection
    red_img = rgb2lab(cropped_image)[:,:,1] 
    r_thresh = threshold_otsu(red_img)
    
    # Scale blur and cleaning to DPI
    blr_img = ndi.uniform_filter(red_img, size = int(0.2 * image_dpi))
    obj_size = int((0.15 * image_dpi)**2)
    hole_size = int((0.25 * image_dpi)**2)
    
    spray_mask = remove_small_holes(remove_small_objects(blr_img > r_thresh, obj_size), hole_size)
    spray_contours = find_contours(spray_mask)
    
    if len(spray_contours) == 2:
        spray_contours = sorted(spray_contours, key=lambda c: np.mean(c[:,0]))
        top_y, top_x = spray_contours[0][:,0], spray_contours[0][:,1]
        bot_y, bot_x = spray_contours[1][:,0], spray_contours[1][:,1]
        characterize = True
    else:
        characterize = False
        
    pix_in = image_dpi

    if characterize:
        x = np.arange(0, cropped_image.shape[1]+1, 1)
        bot_interp = intp(bot_x, bot_y, assume_sorted = False, fill_value = 'extrapolate')(x)
        top_interp = intp(top_x, top_y, assume_sorted = False, fill_value = 'extrapolate')(x)
        center = cntr[2] * x + cntr[1]
        
        spray_width = abs(top_interp - bot_interp) / pix_in
        bot_height = (bot_interp - center) / pix_in
        top_height = (center - top_interp) / pix_in
        apparent_center = np.mean([top_interp, bot_interp], axis = 0)
        deflection = np.arctan(((center - apparent_center)/pix_in) / 1) * (180/np.pi)

        fig, ax = plt.subplot_mosaic("A", dpi = 200)
        ax['A'].imshow(cropped_image)
        ax['A'].plot(x, top_interp, color='cyan', lw=1)
        ax['A'].plot(x, bot_interp, color='magenta', lw=1)
        ax['A'].axline((cntr[0], cntr[1]), slope = cntr[2], ls = '--', color = 'white', lw = 1)
        ax['A'].set_axis_off()
        
        fig_mem = io.BytesIO()
        plt.savefig(fig_mem, bbox_inches = 'tight', pad_inches = 0)

        results = pd.DataFrame({
            'mean': [spray_width.mean(), top_height.mean(), bot_height.mean(), deflection.mean()],
            'σ': [spray_width.std(), top_height.std(), bot_height.std(), deflection.std()],
            'min.': [spray_width.min(), top_height.min(), bot_height.min(), deflection.min()],
            'max.': [spray_width.max(), top_height.max(), bot_height.max(), deflection.max()]
        }, index=['Spray Width (in)', 'Top Side (in)', 'Bottom Side (in)', 'Deflection (deg)'])

        with image_placeholder.container():
            st.pyplot(fig)
        with values_placeholder.container():
            st.subheader('Analysis Results')
            st.dataframe(results.style.format("{:.3f}"))
            
            # Pass/Fail Logic
            if 1.0 <= spray_width.mean() <= 2.0 and abs(deflection.mean()) <= 10:
                disp = 'PASS'
                st.success(disp)
            else:
                disp = 'FAIL'
                st.error(disp)

            button_place = st.empty()
            doc = docx.Document()
            doc.add_heading('Litmus Test Report', 0)
            doc.add_paragraph(f"DPI: {pix_in}")
            # ... (Rest of doc generation logic)
            
            download_object = io.BytesIO()
            doc.save(download_object)
            with button_place.container():
                st.download_button(label = 'Download Report', data = download_object, file_name = 'report.docx')
    else:
        st.error('FAIL: spray pattern edge detection unsuccessful')
