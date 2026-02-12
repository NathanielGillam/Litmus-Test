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
    # On some browsers, the scrollbar constantly switches back and forth between its thicker mouseover state
    # and its thinner non-mouseover state causing the page to resize cyclically.
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
        st.markdown(body = """:grey[NOTE: The new auto-crop feature is still being tested and will likely not be completely reliable.
                    This feature is very sensitive to the condition of the edges of the litmus paper. Straight, clean cut edges will produce the best result.]""")

### Set the standard image resolution used to 200 dpi
#image_dpi = 200

### Create a page header and seperate the page into columns for instructions, image results, and measurement results.      
st.header('Auto-Oxi Litmus Test Characterization')
inst_col, m1, crop_col = st.columns([0.28, 0.02, 0.7])
with crop_col:
    img_col, m2, val_col = st.columns([0.5, 0.01, 0.49])

### Write out instructions on the use of the tool in the instruction column of the page.
with inst_col:
    st.markdown(
        body = """Instructions:
        \n1. After the litmus paper has dried, scan the paper using a .jpg file type and an image resolution of 200 dpi.
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

### If an image has been uploaded, proceed.
if uploaded_image != None:
    image = skimage.io.imread(uploaded_image)

    ### Read input image metadata
    meta_img = Image.open(uploaded_image)
    meta_dict = meta_img.info
    
    # --- NEW LOGIC START ---
    # Default to 200 if no DPI is found
    default_dpi = 200 
    
    if 'dpi' in meta_dict:
        input_dpi = meta_dict['dpi']
        # Check if the metadata is a valid tuple/list
        if isinstance(input_dpi, (tuple, list)) and len(input_dpi) >= 2:
            image_dpi = input_dpi[0]
        else:
            image_dpi = default_dpi
    else:
        # If no DPI info exists, use the default
        image_dpi = default_dpi
        st.warning(f"No DPI metadata found. Defaulting to {image_dpi} DPI for calculations.")
    
    # Update pix_in immediately to ensure it's available for processing
    pix_in = image_dpi
    # --- NEW LOGIC END ---

    ### Actions taken if the image DPI does not match 200 (Optional Warning)
    if image_dpi != 200:
        disp = f'WARNING: Using {image_dpi} DPI. Expected 200 DPI. Measurements may vary.'
        st.error(disp, icon="⚠️")

    ### Auto-crop function
    if auto_crop:
    try:
        # Convert to grayscale and threshold to find the paper
        gray = skimage.color.rgb2gray(image)
        thresh = skimage.filters.threshold_otsu(gray)
        binary = gray > thresh
        
        # Clean up noise and find the paper's coordinates
        binary = skimage.morphology.remove_small_objects(binary, min_size=500)
        coords = np.column_stack(np.where(binary))
        
        if coords.size == 0:
            raise ValueError("No paper detected")

        # Get crop boundaries (top, bottom, left, right)
        t_row, l_col = coords.min(axis=0)
        b_row, r_col = coords.max(axis=0)
        
        # Set crop values with a small 5-pixel buffer
        tc, lc = t_row, l_col
        bc, rc = (image.shape[0] - b_row), (image.shape[1] - r_col)

    except Exception as e:
        with crop_placeholder.container():
            st.error(f'Auto-crop error: {e}. Try manual cropping.', icon="⚠️")

                
    ### Crop the image according to the user's inputs on the sidebar
    cropped_image = skimage.util.crop(image, ((tc, bc), (lc, rc), (0,0)))

    ### Create an image that shows red lines at the boundaries of the user's cropping inputs so that they can confirm their borders are correct.
    extend = 20
    crop_check, ax = plt.subplots()
    if auto_crop:
        ax.imshow(skimage.segmentation.mark_boundaries(image, unborder, mode = 'thick', color = (1,0.1,0.6)))
    else:
        ax.imshow(image)
    ax.set_xlim(0,image.shape[1])
    ax.set_ylim(image.shape[0],0)
    ax.tick_params(labelsize = 6)
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
    ### Otherwise, use the threshold_otsu function to automatically select a threshold based on the range of lightness values in the image.
    else:
        g_thresh = threshold_otsu(gray_img)

    ### Use masking to limit the area in which the centerline can be identified to a horizontal band in the middle of the image.
        ### This prevents the handwritten text or lines indicating areas to write on the litmus sheet from being accidentally identified as the centerline.
    mask = gray_img < g_thresh
    mask_top = int(0.35*mask.shape[0])
    mask_bot = int(0.65*mask.shape[0])
    mask[:mask_top] = 0
    mask[mask_bot:] = 0
        
    ### Find candidate straight lines using a straight line Hough transform. The longest distance straight line identified will be used.
    tested_angles = np.linspace(-np.pi/2, np.pi/2, 360, endpoint = False)
    h, theta, d = hough_line(mask, theta = tested_angles)
    _, angle, dist = hough_line_peaks(h, theta, d, num_peaks = 1)
    angle = angle[0]
    slope = np.tan(angle + np.pi/2)
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    cntr = (x0, y0, slope)

    ### Use the red/green channel of the LAB colorspace image to identify the edges of the sprayed area.
    red_img = rgb2lab(cropped_image)[:,:,2]
    ### Mask off the black pixels of the centerline to prevent it from being identified as a spray pattern edge
    y_end = cntr[2]*mask.shape[1]+cntr[1]
    mask_top = int(min(cntr[1], y_end) - 0.02*mask.shape[0])
    mask_bot = int(max(cntr[1], y_end) + 0.02*mask.shape[0])
    mask[:mask_top,:] = 0
    mask[mask_bot:,:] = 0
    r_thresh = threshold_otsu(red_img)
    red_masked = red_img + r_thresh*mask
    ### Blur the red filtered image by a constant value to deal with small droplets at the edge of the spray band.
    blr_img = ndi.uniform_filter(red_masked, size = 0.2*image_dpi)
    ### Use the remove small holes and remove small objects to clean up small gaps inside the sprayed area and small droplets outside the sprayed area.
        ### This ensures that only two edges (top and bottom) are identified for the spray band rather than an edge around every small gap or droplet.
    spray_mask = remove_small_holes(remove_small_objects(blr_img > r_thresh, image_dpi*100), image_dpi*200)
    
    
    ### Use the find contours function to identify the pixels at the top and bottom edge of the spray band.
    spray_contours = find_contours(spray_mask)
    ### If exactly two edges are found, proceed.
    if len(spray_contours) == 2:
        bot_x = spray_contours[1][:,1]
        bot_y = spray_contours[1][:,0]
        top_x = spray_contours[0][:,1]
        top_y = spray_contours[0][:,0]
        characterize = True
    ### Otherwise, return a failure message because the expected number of edges were not found.
    else:
        characterize = False
        
    ### Select the conversion scale of pixels to inches
        ### This can be found in the properties of the photo (often called DPI - dots per inch). In this case, the standard DPI was identified earlier in the code.
    pix_in = image_dpi

    #####################
    # Measurement Section
    #####################

    ### If two edges were identified, proceed
    if characterize:
        ### Use interpolation to turn the top and bottom edges into an array of one y value for each pixel on the x-axis (width).
            ### This allows the edges to be treated like functions and have mathematical operations performed on them.
        x = np.arange(0, cropped_image.shape[1]+1, 1)
        bot_interp = intp(bot_x, bot_y, assume_sorted = False, fill_value = 'extrapolate')(x)
        top_interp = intp(top_x, top_y, assume_sorted = False, fill_value = 'extrapolate')(x)
        ### Use the slope and y-intercept of the centerline previously identifed to create a centerline array at each x point.
        center = cntr[2] * x + cntr[1]
        ### Spray width array is the number of pixels between the top and bottom edges at each x point divided by the number of pixels per inch.
        spray_width = abs(top_interp - bot_interp) / pix_in
        ### Determine the widht of the spray band above and below the centerline.
        bot_height = (bot_interp - center) / pix_in
        top_height = (center - top_interp) / pix_in
        ### Take the average of the top and bottom edge arrays to find the array of the apparent center of the spray band
        apparent_center = np.mean([top_interp, bot_interp], axis = 0)
        ### Use the apparent center of the spray band to calculate the angle of deflection of the spray band at each x point.
        deflection = np.arctan(((center - apparent_center)/pix_in) / 1) * (180/np.pi)
        ### Create a display x array in inches to match the units of the other measurement arrays.
        disp_x = x / pix_in

    ### Generate the resultant image showing the identified centerline and spray edges overlaid on the cropped spray pattern image.
    fig, ax = plt.subplot_mosaic("A", dpi = 200)
    ax['A'].imshow(cropped_image)
    ax['A'].set_xticks([])
    ax['A'].set_yticks([])

    ### If two spray pattern edges were successfully identified...
    if characterize:
        ax['A'].plot(bot_x, bot_y, color = 'black', lw = 1)
        ax['A'].plot(top_x, top_y, color = 'black', lw = 1)
        
        ### Use a white outward pointing arrow to identify the maximum spray pattern width.
        max_loc = np.argmax(spray_width)
        ax['A'].scatter([x[max_loc]], [bot_interp[max_loc]], marker = 7, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax['A'].scatter([x[max_loc]], [top_interp[max_loc]], marker = 6, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax['A'].plot([x[max_loc], x[max_loc]], [bot_interp[max_loc], top_interp[max_loc]], color = 'white', lw = 1, ls = (0,(2,2)), alpha = 0.5)
        
        ### Use a white inward pointing arrow to identify the minimum spray pattern width.
        min_loc = np.argmin(spray_width)
        ax['A'].scatter([x[min_loc]], [bot_interp[min_loc]], marker = 6, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax['A'].scatter([x[min_loc]], [top_interp[min_loc]], marker = 7, color = 'white', zorder = 4, alpha = 0.5, clip_on = False)
        ax['A'].plot([x[min_loc], x[min_loc]], [bot_interp[min_loc], top_interp[min_loc]], color = 'white', lw = 1, ls = (0,(2,2)), alpha = 0.5)
        
        ### Use a red outward pointing arrow to identify the maximum deflection and its direction.
        max_loc = np.argmax(deflection)
        if deflection.max() > 0:
            ax['A'].scatter([x[max_loc]], [top_interp[max_loc]], marker = 6, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax['A'].plot([x[max_loc], x[max_loc]], [center[max_loc], top_interp[max_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
        else:
            ax['A'].scatter([x[max_loc]], [bot_interp[max_loc]], marker = 6, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax['A'].plot([x[max_loc], x[max_loc]], [center[max_loc], bot_interp[max_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
            
        ### Use a red inward pointing arrow to identify the minimum deflection and its direction.
        min_loc = np.argmin(deflection)
        if deflection.min() > 0:
            ax['A'].scatter([x[min_loc]], [top_interp[min_loc]], marker = 7, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax['A'].plot([x[min_loc], x[min_loc]], [center[min_loc], top_interp[min_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
        else:
            ax['A'].scatter([x[min_loc]], [bot_interp[min_loc]], marker = 7, color = 'red', zorder = 4, alpha = 0.5, clip_on = False)
            ax['A'].plot([x[min_loc], x[min_loc]], [center[min_loc], bot_interp[min_loc]], color = 'red', lw = 1, ls = (2,(2,2)), alpha = 0.5)
    else:
        ax['A'].contour(spray_mask, colors = 'black')

    ### Use a white dashed line to mark the identified centerline. If it lines up with the actual centerline, it should result in a white-black alternating dashed line.
    ax['A'].axline((cntr[0], cntr[1]), slope = cntr[2], ls = (0,(4,4)), color = 'white', lw = 1.5)   
    ax['A'].set_xlim(0,cropped_image.shape[1])
    ax['A'].set_ylim(cropped_image.shape[0],0)
    ### Save the resultant image to memory for later access when generating a report.
    fig_mem = io.BytesIO()
    plt.savefig(fig_mem, bbox_inches = 'tight', pad_inches = 0)

    ### Summarize the spray band measurements in a table which will be shown to the right of the image output.
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

    ### Show the final image in the image placeholder container location.
    with image_placeholder.container():
        st.write(fig)

    ### Display the results table inthe values placeholder container location.
    with values_placeholder.container():
        if characterize:
            st.subheader('Analysis Results')
            st.dataframe(results)

    ##################################
    # Spray Pattern Pass/Fail Criteria
    ##################################

            ### Fail if the mean spray width is not a number.
            if np.isnan(spray_width.mean()):
                disp = 'FAIL: spray pattern edge detection unsuccessful'
                st.write("#### :red[{}]".format(disp))
            ### Fail if the mean spray width is less than one inch.
            elif spray_width.mean() < 1:
                disp = 'FAIL: mean spray width must be >1 in.'
                st.write("#### :red[{}]".format(disp))
            ### Fail if the mean spray width is greater than two inches
            elif spray_width.mean() > 2:
                disp = 'FAIL: mean spray width must be <2 in.'
                st.write('#### :red[{}]'.format(disp))
            ### Fail if the mean deflection is greater than 10 degrees in either direction.
            elif abs(deflection.mean()) > 10:
                disp = 'FAIL: mean deflection angle must be <10°'
                st.write("#### :red[{}]".format(disp))
            ### Fail if the max deflection and min deflection are too far apart, indicating a spray band that is misaligned with the centerline.
            elif (deflection.max() - deflection.min()) > 15:
                disp = 'FAIL: deflection range too large'
                st.write("#### :red[{}]".format(disp))
            ### Fail if a very large maximum deflection angle is present, indicating a drip may have been included in the identified spray band.
            elif abs(deflection).max() > 18:
                disp = 'FAIL: suspected drip inclusion'
                st.write("#### :red[{}]".format(disp))
            ### If none of the failure conditions were present, the test passes.
            else:
                disp = 'PASS'
                st.write("### :green[{}]".format(disp))

            ### Create a placeholder for the report download button under the measurement summary table.
            button_place = st.empty()

            ### Display the pass/fail criteria to the user
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

            ### Initialize a word document.
            doc = docx.Document()
            ### Set up document margins
            section = doc.sections[-1]
            section.left_margin = docx.shared.Inches(0.75)
            section.right_margin = docx.shared.Inches(0.75)
            section.top_margin = docx.shared.Inches(0.75)
            section.bottom_margin = docx.shared.Inches(0.75)
            ### Add a title to the document
            title = doc.add_paragraph() 
            title.paragraph_format.space_after = 0
            title_run = title.add_run('Litmus Test Characterization Report')
            title_run.font.size = docx.shared.Pt(18)
            title_run.font.name = 'Arial'
            title_run.underline = True
            ### Add the date that the report was generated
            date = doc.add_paragraph()
            date_run = date.add_run(datetime.datetime.now().strftime('%B %d, %Y - %I:%M %p'))
            date_run.font.size = docx.shared.Pt(10)
            date_run.font.name = 'Arial'
            ### Add text specifying the outcome of the litmus test (pass or fail).
            pf = doc.add_paragraph()
            pf_run = pf.add_run(disp)
            if disp == 'PASS':
                pf_run.font.color.rgb = docx.shared.RGBColor(40,166,18)
            else:
                pf_run.font.color.rgb = docx.shared.RGBColor(252,3,3)
            pf_run.font.size = docx.shared.Pt(16)
            pf_run.font.name = 'Arial'
            pf_run.bold = True
            ### Add the resultant image of the spray pattern with the centerline and edge lines overlaid.
            spray_pic = doc.add_picture(fig_mem, height = docx.shared.Inches(cropped_image.shape[1]/pix_in))
            ### If the user selected their own centerline threshold, indicate that in the report.
            if edit_center:
                pic_p = doc.paragraphs[-1]
                pic_p.paragraph_format.space_after = 0
                center_p = doc.add_paragraph()
                center_p.paragraph_format.space_before = 0
                center_run = center_p.add_run('Centerline threshold adjusted to {}'.format(m_thresh))
                center_run.font.size = docx.shared.Pt(8)
                center_run.font.name = 'Arial'
            ### If the DPI is not 200, document it.
            if image_dpi != 200:
                dpi_p = doc.paragraphs[-1]
                dpi_p.paragraph_format.space_after = 0
                center_p = doc.add_paragraph()
                center_p.paragraph_format.space_before = 0
                center_run = center_p.add_run('Image resolution did not match expected 200 DPI. Input image resolution was {}. This may affect dimensional measurements.'.format(image_dpi))
                center_run.font.size = docx.shared.Pt(8)
                center_run.font.name = 'Arial'
            ### Add a table of the summarized spray band measurements.
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

            ### Save the report in memory so that it is ready for the user to download.
            download_object = io.BytesIO()
            doc.save(download_object)

            ### Use the previously created button container to provide a download button where the user can download the final report.
            with button_place.container():
                st.download_button(label = 'Download Report', data = download_object, file_name = 'litmus_test_report_{}.docx'.format(datetime.datetime.now().strftime('%m_%d_%y_%H_%M_%f')))

        ### Edge detection failure message if the number of edges found is anything other than two.
        else:
            disp = 'FAIL: spray pattern edge detection unsuccessful'
            st.write("#### :red[{}]".format(disp))   
