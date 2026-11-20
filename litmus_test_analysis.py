### Page Header & Tabs
st.header("Auto-Oxi Litmus Test Characterization")
tab1, tab2 = st.tabs(["Litmus Test", "Database Dashboard"])

with tab1:
    inst_col, m1, crop_col = st.columns([0.28, 0.02, 0.7])
    with crop_col:
        img_col, m2, val_col = st.columns([0.5, 0.01, 0.49])

    ### Always show instructions in the left column
    with inst_col:
        st.markdown(
            """Instructions:
            \n1. After the litmus paper has dried, scan the paper using a .jpg file type and an image resolution of 200 dpi.
            \n2. Upload the image to the "File Input" area on the left sidebar.
            \n3. Crop the image so that the edges of the litmus paper are just out of frame. If cropping manually, check the "Done Cropping" box to proceed with analysis.
            \n4. Confirm that the white dashed line correctly aligns with the centerline drawn on the litmus paper. 
            If it does not align, check the "Adjust centerline identification" box in the "Image Settings" area on the left sidebar and adjust the threshold until the centerlines align correctly.
            \n5. Dimensional characterization results will be shown to the right of the litmus paper image. Use the "Download Report" button to generate a file documenting the analysis results."""
        )
        st.divider()

    with crop_col:
        crop_placeholder = st.empty()
    with img_col:
        image_placeholder = st.empty()
    with val_col:
        values_placeholder = st.empty()

    # Require user selections before running analysis
    if spray_cell == "-- Select --" or chemical_type == "-- Select --":
        with crop_placeholder.container():
            st.warning("Please select both Spray Cell Number and Chemical Type before uploading an image.")
    else:
        if uploaded_image is not None:
            # ... rest of your image loading, cropping, analysis, and report code ...
