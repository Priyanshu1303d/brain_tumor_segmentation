import streamlit as st
import torch
import tempfile
import os
import nibabel as nib
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity
from monai.networks.nets import SwinUNETR
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- CONFIG --------------------
IMG_SIZE = (96, 96, 96)
IN_CHANNELS = 4
OUT_CHANNELS = 4
MODEL_PATH = "trials/swin_unetr_model.pth"
MODALITY_NAMES = ["T1n", "T1c", "T2w", "T2f"]
CLASS_NAMES = ["Background", "Necrotic core", "Edema", "Enhancing tumor"]
CLASS_COLORS = [(0, 0, 0, 0), (1, 0, 0, 0.7), (0, 1, 0, 0.7), (0, 0, 1, 0.7)]  # RGBA format for overlays

# -------------------- STREAMLIT UI --------------------
st.set_page_config(layout="wide", page_title="Brain Tumor Segmentation")
st.title("🧠 Brain Tumor Segmentation")
st.write("")
st.write("Upload MRI scans to detect and visualize brain tumors")
st.markdown("---")

# -------------------- TRANSFORMS --------------------
@st.cache_resource
def get_transform():
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(IMG_SIZE),
        ScaleIntensity()
    ])

transform = get_transform()

# Load model function with caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model file not found at {MODEL_PATH}. Please ensure the model exists before proceeding.")
        return None
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SwinUNETR(
            img_size=IMG_SIZE,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            feature_size=24,
            use_checkpoint=False
        )
        
        # Load with device mapping
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        st.success(f"Model loaded successfully (using {device})")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to validate NIfTI files
def is_valid_nifti(file):
    try:
        # Check file extension
        filename = file.name.lower()
        if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
            return False
        
        # Save to temp file and try loading with nibabel to verify it's a valid NIfTI
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp:
            temp.write(file.getvalue())
            temp_path = temp.name
        
        try:
            nib.load(temp_path)
            return True
        except:
            return False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except:
        return False

# Function to find slices containing tumor
def find_tumor_slices(pred_volume):
    """Find slices that contain tumor tissue in each dimension"""
    tumor_slices = {
        "Axial": [],
        "Coronal": [],
        "Sagittal": []
    }
    
    # Find tumor slices in each dimension
    for z in range(pred_volume.shape[2]):
        if np.any(pred_volume[:, :, z] > 0):
            tumor_slices["Axial"].append(z)
    
    for y in range(pred_volume.shape[1]):
        if np.any(pred_volume[:, y, :] > 0):
            tumor_slices["Coronal"].append(y)
            
    for x in range(pred_volume.shape[0]):
        if np.any(pred_volume[x, :, :] > 0):
            tumor_slices["Sagittal"].append(x)
    
    return tumor_slices

# Initialize sidebar
st.sidebar.title("Configuration")

# Model initialization in sidebar
with st.sidebar:
    with st.spinner("Initializing model..."):
        model = load_model()
    
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload all 4 modality MRI scans
    2. Ensure files are in NIfTI format (.nii or .nii.gz)
    3. Required modalities:
       - T1n: T1-weighted without contrast
       - T1c: T1-weighted with contrast
       - T2w: T2-weighted 
       - T2f: T2-FLAIR
    """)

# File upload section
st.write("### Upload MRI Scans")
st.write("Please upload all 4 MRI modality files (T1n, T1c, T2w, T2f)")

# Map from modality name to file
modality_files = {}

# Create file uploaders for each modality
col1, col2 = st.columns(2)
with col1:
    # Fix: Use None instead of specific extensions to avoid the .gz issue
    t1n_file = st.file_uploader("T1n (T1-weighted without contrast)", type=None, 
                               accept_multiple_files=False, help="Accept .nii or .nii.gz files")
    if t1n_file is not None and is_valid_nifti(t1n_file):
        modality_files["T1n"] = t1n_file
    
    t2w_file = st.file_uploader("T2w (T2-weighted)", type=None,
                               accept_multiple_files=False, help="Accept .nii or .nii.gz files")
    if t2w_file is not None and is_valid_nifti(t2w_file):
        modality_files["T2w"] = t2w_file

with col2:
    t1c_file = st.file_uploader("T1c (T1-weighted with contrast)", type=None,
                               accept_multiple_files=False, help="Accept .nii or .nii.gz files")
    if t1c_file is not None and is_valid_nifti(t1c_file):
        modality_files["T1c"] = t1c_file
    
    t2f_file = st.file_uploader("T2f (T2-FLAIR)", type=None,
                               accept_multiple_files=False, help="Accept .nii or .nii.gz files")
    if t2f_file is not None and is_valid_nifti(t2f_file):
        modality_files["T2f"] = t2f_file

# Show validation messages for uploaded files
for modality, file in list(modality_files.items()):
    if not is_valid_nifti(file):
        st.warning(f"File for {modality} is not a valid NIfTI file. Please upload a .nii or .nii.gz file.")
        del modality_files[modality]

# Check if we have all required modalities
if len(modality_files) == 4 and model is not None:
    st.success("All required files uploaded!")
    
    # Process files
    with st.spinner("Processing files..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            modality_paths = {}
            
            # Save uploaded files to temp directory
            for modality, file in modality_files.items():
                path = os.path.join(tmpdir, file.name)
                with open(path, "wb") as out:
                    out.write(file.read())
                paths.append(path)
                modality_paths[modality] = path
            
            # Process based on modality order expected by the model
            ordered_paths = [modality_paths.get(mod) for mod in MODALITY_NAMES]
            
            # Run inference
            with st.spinner("Running inference..."):
                try:
                    # Process images
                    images = []
                    original_images = []
                    original_affines = []
                    
                    for i, p in enumerate(ordered_paths):
                        try:
                            # Load original image for reference
                            nib_img = nib.load(p)
                            original_images.append(nib_img.get_fdata())
                            original_affines.append(nib_img.affine)
                            
                            # Process with MONAI transforms
                            img = transform(p)
                            images.append(img)
                        except Exception as e:
                            st.error(f"Error processing {os.path.basename(p)}: {str(e)}")
                            st.stop()
                    
                    # Create input tensor
                    image = torch.cat(images, dim=0).unsqueeze(0)  # Shape: (1, 4, H, W, D)
                    
                    # Get device
                    device = next(model.parameters()).device
                    image = image.to(device)
                    
                    # Run inference
                    with torch.no_grad():
                        output = model(image)
                        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                    
                    # Check if tumor is present
                    tumor_present = np.any(pred > 0)
                    
                    # Find tumor slices in each dimension
                    tumor_slices = find_tumor_slices(pred)
                    
                    # Get reference image
                    ref_img = image.squeeze(0)[0].cpu().numpy()  # T1n as reference
                    
                    # Create segmentation visualization
                    if ref_img.shape[2] > 0:  # Make sure there's a valid Z dimension
                        # Create tabs for different views and modalities
                        view_tab, stats_tab = st.tabs(["Image Visualization", "Tumor Statistics"])
                        
                        with view_tab:
                            # Add selector for modality
                            modality_idx = st.selectbox(
                                "Select MRI Modality", 
                                range(len(MODALITY_NAMES)), 
                                format_func=lambda i: MODALITY_NAMES[i]
                            )
                            
                            # Get selected modality image
                            selected_img = image.squeeze(0)[modality_idx].cpu().numpy()
                            
                            # Create columns for the view selection and controls
                            view_col, control_col = st.columns([3, 1])
                            
                            with control_col:
                                st.markdown("### View Controls")
                                view_type = st.radio("Select View", ["Axial", "Coronal", "Sagittal"])
                                show_overlay = st.checkbox("Show Tumor Overlay", value=True)
                                overlay_opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5)
                                
                                # Get class visibility toggles
                                st.markdown("### Classes")
                                class_visibility = {}
                                for i, class_name in enumerate(CLASS_NAMES):
                                    if i > 0:  # Skip background
                                        class_visibility[i] = st.checkbox(f"{class_name}", value=True)
                                
                                # Display tumor slice information
                                if tumor_present:
                                    st.markdown("### Tumor Slices")
                                    st.write(f"Tumor detected in:")
                                    if tumor_slices[view_type]:
                                        st.write(f"- {view_type}: {len(tumor_slices[view_type])} slices")
                                        if len(tumor_slices[view_type]) <= 10:
                                            st.write(f"  Slices: {', '.join(map(str, tumor_slices[view_type]))}")
                                        else:
                                            st.write(f"  Range: {min(tumor_slices[view_type])} - {max(tumor_slices[view_type])}")
                                        
                                        # Add a selector for tumor slices
                                        if len(tumor_slices[view_type]) > 0:
                                            st.markdown("**Jump to tumor slice:**")
                                            tumor_slice_idx = st.selectbox(
                                                "Select tumor slice", 
                                                tumor_slices[view_type],
                                                format_func=lambda i: f"Slice {i}"
                                            )
                                else:
                                    st.info("No tumor detected in this scan.")
                            
                            with view_col:
                                # If tumor is present and there are tumor slices in this view,
                                # default to showing a tumor slice instead of middle slice
                                default_slice = None
                                if tumor_present and tumor_slices[view_type]:
                                    default_slice = tumor_slices[view_type][len(tumor_slices[view_type])//2]
                                
                                if view_type == "Axial":
                                    if default_slice is None:
                                        default_slice = selected_img.shape[2]//2
                                    slice_idx = st.slider("Axial Slice", 0, selected_img.shape[2]-1, default_slice)
                                    img_slice = selected_img[:, :, slice_idx]
                                    pred_slice = pred[:, :, slice_idx]
                                    # Add indicator if tumor is present in this slice
                                    if slice_idx in tumor_slices["Axial"]:
                                        st.success("⚠️ Tumor present in this slice")
                                elif view_type == "Coronal":
                                    if default_slice is None:
                                        default_slice = selected_img.shape[1]//2
                                    slice_idx = st.slider("Coronal Slice", 0, selected_img.shape[1]-1, default_slice)
                                    img_slice = selected_img[:, slice_idx, :]
                                    pred_slice = pred[:, slice_idx, :]
                                    # Add indicator if tumor is present in this slice
                                    if slice_idx in tumor_slices["Coronal"]:
                                        st.success("⚠️ Tumor present in this slice")
                                else:  # Sagittal
                                    if default_slice is None:
                                        default_slice = selected_img.shape[0]//2
                                    slice_idx = st.slider("Sagittal Slice", 0, selected_img.shape[0]-1, default_slice)
                                    img_slice = selected_img[slice_idx, :, :]
                                    pred_slice = pred[slice_idx, :, :]
                                    # Add indicator if tumor is present in this slice
                                    if slice_idx in tumor_slices["Sagittal"]:
                                        st.success("⚠️ Tumor present in this slice")
                                
                                # Create visualization figure
                                fig, ax = plt.subplots(figsize=(10, 10))
                                ax.imshow(img_slice, cmap="gray")
                                
                                # Apply overlay mask if enabled
                                if show_overlay:
                                    # Create masked overlay for each class
                                    for class_idx in range(1, OUT_CHANNELS):  # Skip background
                                        if class_idx in class_visibility and class_visibility[class_idx]:
                                            mask = pred_slice == class_idx
                                            color = CLASS_COLORS[class_idx]
                                            colored_mask = np.zeros((*mask.shape, 4))
                                            colored_mask[mask] = color
                                            colored_mask[..., 3] = colored_mask[..., 3] * overlay_opacity
                                            ax.imshow(colored_mask)
                                
                                ax.set_title(f"{MODALITY_NAMES[modality_idx]} - {view_type} View (Slice {slice_idx})")
                                ax.axis("off")
                                st.pyplot(fig)
                        
                        with stats_tab:
                            st.subheader("Tumor Statistics")
                            
                            # Calculate tumor statistics
                            tumor_voxels = np.sum(pred > 0)
                            total_voxels = pred.size
                            tumor_percentage = (tumor_voxels / total_voxels) * 100
                            
                            # Only show detailed results if tumor is present
                            if tumor_present:
                                # Create tumor class distribution
                                classes = np.unique(pred)
                                class_counts = {int(c): np.sum(pred == c) for c in classes}
                                
                                # Display statistics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Total Tumor Voxels", f"{tumor_voxels:,}")
                                    st.metric("Tumor Volume Percentage", f"{tumor_percentage:.2f}%")
                                
                                with col2:
                                    # Calculate approximate volume in cm³ (assuming 1mm³ voxels)
                                    # This is a rough estimate - would need actual voxel dimensions for accuracy
                                    voxel_volume_mm3 = 1.0  # Placeholder
                                    tumor_volume_mm3 = tumor_voxels * voxel_volume_mm3
                                    tumor_volume_cm3 = tumor_volume_mm3 / 1000.0
                                    
                                    st.metric("Estimated Tumor Volume", f"{tumor_volume_cm3:.2f} cm³")
                                
                                # Display class distribution chart
                                st.subheader("Tumor Class Distribution")
                                
                                # Prepare data for pie chart
                                class_labels = [CLASS_NAMES[int(c)] for c in classes if c > 0]  # Exclude background
                                class_values = [class_counts[int(c)] for c in classes if c > 0]
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                colors = [CLASS_COLORS[i][:3] for i in classes if i > 0]  # RGB part of RGBA
                                ax.pie(class_values, labels=class_labels, autopct='%1.1f%%', 
                                       colors=colors, shadow=True, startangle=90)
                                ax.axis('equal')
                                st.pyplot(fig)
                                
                                # Detailed class breakdown table
                                st.subheader("Detailed Class Breakdown")
                                class_data = []
                                for c in classes:
                                    if int(c) < len(CLASS_NAMES):
                                        class_name = CLASS_NAMES[int(c)]
                                        count = class_counts[int(c)]
                                        percentage = (count / total_voxels) * 100
                                        class_data.append({
                                            "Class": class_name,
                                            "Voxel Count": f"{count:,}",
                                            "Percentage": f"{percentage:.2f}%"
                                        })
                                
                                st.table(class_data)
                                
                                # Tumor location summary
                                st.subheader("Tumor Location")
                                st.write("Tumor slices by view:")
                                
                                for view_name, slices in tumor_slices.items():
                                    if slices:
                                        if len(slices) <= 10:
                                            st.write(f"- **{view_name}**: Slices {', '.join(map(str, slices))}")
                                        else:
                                            st.write(f"- **{view_name}**: {len(slices)} slices, range {min(slices)}-{max(slices)}")
                            else:
                                st.info("No tumor detected in the scan.")
                    else:
                        st.error("Invalid image dimensions detected")
                        
                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
elif len(modality_files) > 0:
    missing = set(MODALITY_NAMES) - set(modality_files.keys())
    if missing:
        st.warning(f"Missing required modalities: {', '.join(missing)}")

# Alternative upload method for users having trouble with the file uploaders
with st.expander("Having trouble with file uploads?"):
    st.write("""
    If you're experiencing issues with the file uploaders above, you can try using a single uploader for all files:
    """)
    
    all_files = st.file_uploader("Upload all NIfTI files", type=None, accept_multiple_files=True,
                                help="Upload all 4 modality files (T1n, T1c, T2w, T2f)")
    
    if all_files:
        st.write("Please assign each file to its modality:")
        file_assignments = {}
        
        for file in all_files:
            if is_valid_nifti(file):
                modality = st.selectbox(f"Modality for {file.name}", 
                                      ["Select modality"] + MODALITY_NAMES,
                                      key=f"modality_{file.name}")
                if modality != "Select modality":
                    file_assignments[modality] = file
            else:
                st.warning(f"{file.name} is not a valid NIfTI file and will be ignored.")
        
        if len(file_assignments) == 4 and all(mod in file_assignments for mod in MODALITY_NAMES):
            st.success("All modalities assigned successfully!")
            if st.button("Process these files"):
                # Replace modality_files with the new assignments
                modality_files = file_assignments
                # (Processing code would be duplicated here or refactored into a function)

# Add explanations at the bottom
with st.expander("About this app"):
    st.write("""
    ## Brain Tumor Segmentation App
    
    This application performs automatic brain tumor segmentation using deep learning. It analyzes MRI scans and identifies different types of tumor tissue.
    
    ### How it works
    
    The app uses a Swin UNETR deep learning model, which is a state-of-the-art architecture for medical image segmentation. The model has been trained on the BraTS (Brain Tumor Segmentation) challenge dataset.
    
    ### Required inputs
    
    Four MRI modalities are required for accurate segmentation:
    
    - **T1n**: T1-weighted MRI without contrast
    - **T1c**: T1-weighted MRI with contrast enhancement
    - **T2w**: T2-weighted MRI
    - **T2f**: T2-FLAIR (Fluid Attenuated Inversion Recovery)
    
    ### Output classes
    
    The model segments brain tumors into multiple tissue classes:
    
    - **Necrotic core**: Areas of dead tumor tissue
    - **Edema**: Swelling around the tumor
    - **Enhancing tumor**: Active tumor regions that enhance with contrast
    
    ### Technical details
    
    - Model: Swin UNETR
    - Input resolution: 96×96×96 voxels
    - Framework: PyTorch + MONAI
    
    ### Disclaimer
    
    This is a research tool and should not be used for clinical diagnosis. Always consult with a qualified medical professional.
    """)

# Add a footer
st.write("")
st.markdown("---")
st.markdown("**Brain Tumor Segmentation Project** • Powered by MONAI & Streamlit • Report any [issues](https://github.com/sanskarmodi8/brain_tumor_segmentation/issues)")