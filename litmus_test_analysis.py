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
        crop_done = False
    
    edit_center = st.checkbox('Adjust centerline identification', value = False)
    if edit_center:
        m_thresh = st.slider(label = 'Centerline threshold', min_value = 30, max_value = 60, value = 45)

    if auto_crop:
        st.markdown(body = "\n\n")
        st.markdown(body = """:grey[NOTE: Auto-crop is optimized for high contrast backgrounds (e.g., open scanner lid).]""")

### Set the standard image resolution used (Default to 96, but will update from file)
image_dpi = 96

### Create a page header and seperate the page into columns for instructions, image results, and measurement results.      
st.header('Auto-Oxi Litmus Test Characterization')
inst_col, m1, crop_col = st.columns([0.28, 0.02, 0.7])
with crop_col:
    img_col, m2, val_col = st.columns([0.5, 0.01, 0.49])

### Write out instructions on the use of the tool in the instruction column of the page.
with inst_col:
    st.markdown(
        body = """Instructions:
        \n1. After the litmus paper has dried, scan the paper using a .jpg file type at **96 DPI** or **200 DPI**.
        \n2. Upload the image to the "File Input" area on the left sidebar.
        \n3. Crop the image so that the edges of the litmus paper are just out of frame. If cropping manually, check the "Done Cropping" box to proceed with analysis.
        \n4. Confirm that the white dashed line correctly aligns with the centerline drawn on the litmus paper. 
        If it does not align, check the "Adjust centerline identification" box in the "Image Settings" area on the left sidebar and adjust the threshold until the centerlines align correctly.
        \n5. Dimensional characterization results will be shown to the right of the litmus paper image. Use the "Download Report" button to generate a file documenting the analysis results.""")
    st.divider()

### Create placeholders for the image during cropping, the resultant image, and the summarized measurements
with crop_col:
    crop_placeholder = st.empty()
with img_col:
    image_placeholder = st.empty()
with val_col:
    values_placeholder = st.empty()

############################
# Image processing section
############################

# Initialize crop_done to prevent NameError
if auto_crop:
    crop_done = False

### If an image has been uploaded, proceed.
if uploaded_image != None:
    image = skimage.io.imread(uploaded_image)
    h, w, _ = image.shape

    ### Read input image metatdata to identify the image resolution
    meta_img = Image.open(uploaded_image)
    # Use .get() to safely handle images without DPI metadata, defaulting to 96
    input_dpi = meta_img.info.get('dpi', (96, 96))
    
    ### Actions taken if the image DPI is not standard
    if not (int(input_dpi[0]) in [96, 200]):
        disp = 'WARNING: Input image resolution is {}. Expected 96 or 200 DPI.'.format(input_dpi)
        st.error(disp, icon="⚠️")
    
    # Update global DPI variable to match the image
    image_dpi = input_dpi[0]

    ### Auto-crop function
    if auto_crop:
        try:
            # UPDATED: More robust auto-crop using Lightness Thresholding (Better for 96 DPI)
            l_chan = rgb2lab(image)[:, :, 0]
            paper_mask = l_chan > threshold_otsu(l_chan)
            
            # Remove noise (Scaled by DPI)
            clean_size = int(500 * (image_dpi/200))
            paper_mask = remove_small_objects(paper_mask, min_size=clean_size)
            paper_mask = remove_small_holes(paper_mask, area_threshold=clean_size)

            coords = np.column_stack(np.where(paper_mask))
            
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Assign crop values
                tc = int(y_min + 5)
                bc = int(h - y_max + 5)
                lc = int(x_min + 5)
                rc = int(w - x_max + 5)
                
                crop_done = True
            else:
                # Fallback if paper not found
                tc, bc, lc, rc = 0, 0, 0, 0
                st.error('Auto-crop was unable to identify the boundaries. Please use manual cropping.', icon="⚠️")

        except Exception as e:
            tc, bc, lc, rc = 0, 0, 0, 0
            with crop_placeholder.container():
                st.error(f'Auto-crop error: {e}. Please use manual cropping.', icon="⚠️")
            
    ### Ensure crop variables exist if manual
    if not auto_crop:
        # Variables tc, bc, lc, rc come from the sidebar widgets
        pass 
        
    ### Crop the image according to the user's inputs on the sidebar
    # Wrapping in try/except to handle case where manual inputs aren't ready
    try:
        cropped_image = skimage.util.crop(image, ((tc, bc), (lc, rc), (0,0)))
    except:
        cropped_image = image

    ### Create an image that shows red lines at the boundaries
    extend = 20
    crop_check, ax = plt.subplots()
    ax.imshow(image)
    ax.set_xlim(0,image.shape[1])
    ax.set_ylim(image.shape[0],0)
    ax.tick_params(labelsize = 6)
    # Ensure lines draw even if crop values are 0
    ax.plot([lc-extend,image.shape[1]-rc+extend],[image.shape[0]-bc,image.shape[0]-bc], color = 'red', lw = 0.8, alpha =  0.6)
    ax.plot([lc-extend,image.shape[1]-rc+extend],[tc,tc], color = 'red', lw = 0.8, alpha = 0.6)
    ax.plot([lc,lc],[image.shape[0]-bc+extend,tc-extend], color = 'red', lw = 0.8, alpha = 0.6)
    ax.plot([image.shape[1]-rc,image.shape[1]-rc],[image.shape[0]-bc+extend,tc-extend], color = 'red', lw = 0.8, alpha = 0.6)

    ### If the user has not checked the "Done Cropping" box, show the crop_check image
    if not crop_done:
        with crop_placeholder.container():
            st.write(crop_check)

### If the user has checked the "Done Cropping" box, proceed with image analysis.
if crop_done:
    ### Convert the input image to the LAB colorspace and use the lightness channel to find dark pixels
    gray_img = rgb2lab(cropped_image)[:,:,0]
    ### If the user chooses to edit the centerline threshold, use their input threshold
    if edit_center:
        g_thresh = m_thresh
    ### Otherwise, use the threshold_otsu function
    else:
        g_thresh = threshold_otsu(gray_img)

    ### Use masking to limit the area
    mask = gray_img < g_thresh
    mask_top = int(0.35*mask.shape[0])
    mask_bot = int(0.65*mask.shape[0])
    mask[:mask_top] = 0
    mask[mask_bot:] = 0
        
    ### Find candidate straight lines using a straight line Hough transform.
    tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint = False)
    h_lines, theta, d = hough_line(mask, theta = tested_angles)
    _, angle, dist = hough_line_peaks(h_lines, theta, d, num_peaks = 1)
    
    # Safety check if no lines found
    if len(angle) > 0:
        angle = angle[0]
        slope = np.tan(angle + np.pi/2)
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        cntr = (x0, y0, slope)
    else:
        # Default horizontal center if fails
        cntr = (0, cropped_image.shape[0]/2, 0)

    ### Use the 'a' channel (Red-Green) instead of 'b' for better red detection
    red_img = rgb2lab(cropped_image)[:,:,2] 

    ### Mask off the black pixels of the centerline
    y_end = cntr[2]*mask.shape[1]+cntr[1]
    mask_top = int(min(cntr[1], y_end) - 0.02*mask.shape[0])
    mask_bot = int(max(cntr[1], y_end) + 0.02*mask.shape[0])
    mask[:mask_top,:] = 0
    mask[mask_bot:,:] = 0
    
    r_thresh = threshold_otsu(red_img)
    red_masked = red_img + r_thresh*mask
    
    ### Blur the red filtered image
    # Scale blur by DPI to match original 200dpi behavior
    blur_val = int(0.2 * image_dpi)
    blr_img = ndi.uniform_filter(red_masked, size = blur_val)
    
    ### Clean up gaps
    # Scale cleaning objects by DPI (area scales quadratically)
    obj_size = int((0.15 * image_dpi)**2) # roughly corresponds to original ratio
    hole_size = int((0.25 * image_dpi)**2)
    
    spray_mask = remove_small_holes(remove_small_objects(blr_img > r_thresh, obj_size), hole_size)
    
    ### Find contours
    spray_contours = find_contours(spray_mask)
    
    ### UPDATED: Robust Edge Selection
    # If we find 2 OR MORE edges, pick the two longest ones (removes noise)
    if len(spray_contours) >= 2:
        # Sort by length (longest first)
        sorted_contours = sorted(spray_contours, key=lambda c: len(c), reverse=True)
        best_two = sorted_contours[:2]
        
        # Sort these two by Y-position (Top vs Bottom)
        best_two = sorted(best_two, key=lambda c: np.mean(c[:,0]))
        
        top_y, top_x = best_two[0][:,0], best_two[0][:,1]
        bot_y, bot_x = best_two[1][:,0], best_two[1][:,1]
        
        characterize = True
    else:
        characterize = False
        
    ### Select the conversion scale of pixels to inches
    pix_in = image_dpi

    #####################
    # Measurement Section
    #####################

    ### If two edges were identified, proceed
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
        disp_x = x / pix_in

    ### Generate the resultant image using plt.subplots (Robust Method)
    fig, ax = plt.subplots(dpi = image_dpi)
    ax.imshow(cropped_image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    if characterize:
        ax.plot(bot_x, bot_y, color = 'black', lw = 1)
        ax.plot(top_x, top_y, color = 'black', lw = 1)
        
        max_loc = np.argmax(spray_width)
        ax.scatter([x[max_loc]], [bot_interp[max_loc]], marker = 7, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax.scatter([x[max_loc]], [top_interp[max_loc]], marker = 6, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax.plot([x[max_loc], x[max_loc]], [bot_interp[max_loc], top_interp[max_loc]], color = 'white', lw = 1, ls = (0,(2,2)), alpha = 0.5)
        
        min_loc = np.argmin(spray_width)
        ax.scatter([x[min_loc]], [bot_interp[min_loc]], marker = 6, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax.scatter([x[min_loc]], [top_interp[min_loc]], marker = 7, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax.plot([x[min_loc], x[min_loc]], [bot_interp[min_loc], top_interp[min_loc]], color = 'white', lw = 1, ls = (0,(2,2)), alpha = 0.5)
        
        max_loc = np.argmax(deflection)
        if deflection.max() > 0:
            ax.scatter([x[max_loc]], [top_interp[max_loc]], marker = 6, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax.plot([x[max_loc], x[max_loc]], [center[max_loc], top_interp[max_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
        else:
            ax.scatter([x[max_loc]], [bot_interp[max_loc]], marker = 6, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax.plot([x[max_loc], x[max_loc]], [center[max_loc], bot_interp[max_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
            
        min_loc = np.argmin(deflection)
        if deflection.min() > 0:
            ax.scatter([x[min_loc]], [top_interp[min_loc]], marker = 7, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax.plot([x[min_loc], x[min_loc]], [center[min_loc], top_interp[min_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
        else:
            ax.scatter([x[min_loc]], [bot_interp[min_loc]], marker = 7, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax.plot([x[min_loc], x[min_loc]], [center[min_loc], bot_interp[min_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
    else:
        # If characterization failed, show the contours found so user can debug
        ax.contour(spray_mask, colors = 'black')

    # Draw Centerline
    ax.axline((cntr[0], cntr[1]), slope = cntr[2], ls = (0,(4,4)), color = 'white', lw = 1.5)   
    ax.set_xlim(0,cropped_image.shape[1])
    ax.set_ylim(cropped_image.shape[0],0)
    
    ### Save the resultant image
    fig_mem = io.BytesIO()
    plt.savefig(fig_mem, bbox_inches = 'tight', pad_inches = 0)

    ### Summarize the spray band measurements
    if characterize:
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

    ### Show the final image
    with image_placeholder.container():
        st.pyplot(fig)

    ### Display the results table
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

            button_place = st.empty()

            st.divider()
            st.markdown(
                body = """\nPass Criteria:
                \n- Mean Width: >1 in. and <2 in.
                \n- Mean Deflection: <10°
                \n- Deflection Range (max - min): <15°
                \n- Max. spray pattern deflection >18° indicates drip inclusion"""
            )
            st.divider()
    
#######################
# Output Result Section
#######################

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
            date_run = date.add_run(datetime.datetime.now().strftime('%B %d, %Y - %I:%M %p'))
            date_run.font.size = docx.shared.Pt(10)
            date_run.font.name = 'Arial'
            pf = doc.add_paragraph()
            pf_run = pf.add_run(disp)
            if disp == 'PASS':
                pf_run.font.color.rgb = docx.shared.RGBColor(40,166,18)
            else:
                pf_run.font.color.rgb = docx.shared.RGBColor(252,3,3)
            pf_run.font.size = docx.shared.Pt(16)
            pf_run.font.name = 'Arial'
            pf_run.bold = True
            
            # Use pix_in for scaling image in word doc
            spray_pic = doc.add_picture(fig_mem, height = docx.shared.Inches(cropped_image.shape[1]/pix_in))
            
            if edit_center:
                pic_p = doc.paragraphs[-1]
                pic_p.paragraph_format.space_after = 0
                center_p = doc.add_paragraph()
                center_p.paragraph_format.space_before = 0
                center_run = center_p.add_run('Centerline threshold adjusted to {}'.format(m_thresh))
                center_run.font.size = docx.shared.Pt(8)
                center_run.font.name = 'Arial'
            
            # Updated DPI check for the report
            if image_dpi not in [96, 200]:
                dpi_p = doc.paragraphs[-1]
                dpi_p.paragraph_format.space_after = 0
                center_p = doc.add_paragraph()
                center_p.paragraph_format.space_before = 0
                center_run = center_p.add_run('Image resolution {} DPI. Expected 96 or 200 DPI.'.format(image_dpi))
                center_run.font.size = docx.shared.Pt(8)
                center_run.font.name = 'Arial'
            
            table = doc.add_table(results.shape[0]+1, results.shape[1]+1)
            for i in range(results.shape[0]):
                table.cell(i+1,0).text = results.index[i]
                table.cell(i+1,0)._tc.tcPr.tcW.type = 'auto'
            for j in range(results.shape[-1]):
                table.cell(0,j+1).text = results.columns[j]
                table.cell(0,j+1)._tc.tcPr.tcW.type = 'auto'
            for i in range(results.shape[0]):
                for j in range(results.shape[-1]):
                    table.cell(i+1,j+1).text = '{:.3f}'.format(results.values[i,j])
                    table.cell(i+1,j+1)._tc.tcPr.tcW.type = 'auto'
            table.style = 'Table Grid'
            table.style.font.name = 'Arial'
            table.allow_autofit = True

            download_object = io.BytesIO()
            doc.save(download_object)

            with button_place.container():
                st.download_button(label = 'Download Report', data = download_object, file_name = 'litmus_test_report_{}.docx'.format(datetime.datetime.now().strftime('%m_%d_%y_%H_%M_%f')))

        else:
            disp = 'FAIL: spray pattern edge detection unsuccessful'
            st.write("#### :red[{}]".format(disp))
            with image_placeholder.container():
                 st.image(spray_mask.astype(float), caption="Debug: Edge Detection Failed")
