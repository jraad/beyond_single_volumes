import streamlit as st
import numpy as np
import os
import torch
import cv2
from PIL import Image
import sqlite3
import pandas as pd

from importlib import reload
import model_architectures
import utilities

reload(model_architectures)
from model_architectures import VAE, Data3D

reload(utilities)
from utilities import get_colormaps

##### TODO
# Add overlay visualization
# Add cluster size suppresion
# Add colorbar



# Define paths
research_dir = r"D:/school/research"
code_dir = os.path.join(research_dir, "code")
model_dir = os.path.join(code_dir, "explore_again", "models")
data_dir = os.path.join(research_dir, "data")
dhcp_rel2 = os.path.join(data_dir, "dhcp_rel2")
processed_dir = os.path.join(dhcp_rel2, "processed")
volume_dir = os.path.join(processed_dir, "volumes")
seg_dir = os.path.join(processed_dir, "segments")
seg_vol_dir = os.path.join(processed_dir, "volume_segments")
pred_dir = os.path.join(dhcp_rel2, "predictions")
seg_pred_dir = os.path.join(pred_dir, "vae_9seg")

l1_dir = os.path.join(volume_dir, "l1")
l5_dir = os.path.join(volume_dir, "l5")

l1_seg_dir = os.path.join(seg_dir, "l1")
l5_seg_dir = os.path.join(seg_dir, "l5")

l1_seg_vol_dir = os.path.join(seg_vol_dir, "l1")
l5_seg_vol_dir = os.path.join(seg_vol_dir, "l5")

l1_seg_pred_dir = os.path.join(seg_pred_dir, "l1")
l5_seg_pred_dir = os.path.join(seg_pred_dir, "l5")


# Indices for train/val/test
np.random.seed(42)
num_samples = int(len(os.listdir(l1_dir)) / 2)
samples = np.array([i for i in range(0, num_samples)])
np.random.shuffle(samples)

split_val = int(0.8 * num_samples)
train_indices = samples[0:split_val]
val_indices = samples[split_val:]

num_test = int(len(os.listdir(l5_dir)) / 2)
test_indices = np.array([i for i in range(0, num_test)])


# Define Datasets
train = Data3D(l1_dir, train_indices)
val = Data3D(l1_dir, val_indices)
test = Data3D(l5_dir, test_indices)


# Additional data
model_options = {
    "T1 Only": {
        "weights": "vae_rel2_t1_second_session.pt",
        "model": VAE(1),
        "shape": (1, 1, 256, 256, 256)
    },
    "T2 Only": {
        "weights": "vae_rel2_t2_second_session.pt",
        "model": VAE(1),
        "shape": (1, 1, 256, 256, 256)
    },
    "Multimodal": {
        "weights": "vae_rel2_t1_t2_second_session.pt",
        "model": VAE(2),
        "shape": (1, 2, 256, 256, 256)
    },
}
max_z = 10
reds, blues = get_colormaps(max_z)
colorbar = Image.open("colorbar.png")


# Helper functions
def get_dataset(batch):
    if batch=="Train (Normal)":
        return train
    elif batch=="Validation (Normal)":
        return val
    else:
        return test

# @st.cache(allow_output_mutation=True)
def load_image(batch, subject):
    """Get subject images from correct batch."""
    data = get_dataset(batch)[int(subject)-1]
    
    t1 = np.stack((data[0],)*3, axis=-1)
    t1 = np.rot90(t1)
    t1 = np.concatenate((t1, np.ones((256, 256, 256, 1))), axis=3)
    
    
    t2 = np.stack((data[1],)*3, axis=-1)
    t2 = np.rot90(t2)
    t2 = np.concatenate((t2, np.ones((256, 256, 256, 1))), axis=3)
    
    if model_version == "T1 Only":
        norm = get_prediction(data[0])
        norm = np.rot90(norm)
    elif model_version == "T2 Only":
        norm = get_prediction(data[1])
        norm = np.rot90(norm)
    else:
        norm = get_prediction(data)
        norm = np.rot90(norm, axes=(1,2))
    
    return t1, t2, norm

def load_model(config):
    model_path = os.path.join(model_dir, config["weights"])
    model = config["model"]
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    
    return model

def get_prediction(img):
    pred = model(torch.Tensor(np.reshape(img, config["shape"])).cuda())
    diff = img - np.reshape(pred.cpu().detach().numpy(), img.shape)
    
    if stdev_version == "Global":
        return diff / 0.1
    else:
        mean = np.mean(diff)
        stdev = np.std(diff)
        
        return (diff - mean) / stdev

def get_cluster_mask(array, min_cluster):
    cluster_map = array.copy()
    cluster_map[cluster_map != 0] = 1
    cluster_map = cluster_map.astype('uint8')
    
    _, labels = cv2.connectedComponents(cluster_map)
    values, counts = np.unique(labels, return_counts=True)
    labels[np.isin(labels, [x for x, y in zip(values, counts) if y < min_cluster])] = 0
    labels[labels != 0] = 1
    
    return labels

def get_overlay(diff):
    rounded = np.round(diff, 1)
    # Values that fall below the mean/threshold
    below_zero = rounded.copy()
    below_zero[below_zero > -threshold] = 0
    below_zero = np.clip(below_zero, -max_z, 0)
    
    # Values that fall above the mean/threshold
    above_zero = rounded.copy()
    above_zero[above_zero < threshold] = 0
    above_zero = np.clip(above_zero, 0, max_z)
    
    if filter_clusters:
        az_mask = get_cluster_mask(above_zero, cluster_size)
        above_zero *= az_mask
        
        bz_mask = get_cluster_mask(below_zero, cluster_size)
        below_zero *= bz_mask
    
    combined = np.zeros((*above_zero.shape, 4))
    for value, color in reds.items():
        combined[above_zero == value] = color
    for value, color in blues.items():
        combined[below_zero == value] = color
        
    return combined
  
def hide_images():
    if "t1" in st.session_state:
        del st.session_state["t1"]

# Define sidebar
st.set_page_config(layout="wide")
st.title("MRI Visualizer")
batch = st.sidebar.selectbox(
    "Batch",
    options=["Train (Normal)", "Validation (Normal)", "Test (Abnormal)"],
    key="batch",
    # on_change=hide_images
)

subject = st.sidebar.selectbox(
    "Subject",
    options=range(1, len(get_dataset(batch))+1),
    key="subject",
    # on_change=hide_images
)
model_version = st.sidebar.selectbox(
    "Model Version",
    options=model_options.keys(),
    key="model_version",
    # on_change=hide_images
)
stdev_version = st.sidebar.radio(
    "Standard Deviation Type:",
    ("Sample-Wise", "Global")
)
submit = st.sidebar.button("Run")

# Define config for visualization
# Slice number slider
slice_number = st.slider(
    "Slice Number",
    1,
    256,
    value=128,
    key="slice_number",
)

# # Overlay config
# config_col1, config_col2 = st.columns([1, 6])
# show_overlay = config_col1.checkbox("Show Overlay", value=True)
# threshold = config_col2.slider(
#     "Threshold",
#     1.,
#     10., 
#     value=2.,
#     key="threshold",
#     step=0.5
# )

# # Cluster config
# cluster_col1, cluster_col2 = st.columns([1, 6])
# filter_clusters = cluster_col1.checkbox("Filter Clusters", value=True)
# cluster_size = cluster_col2.slider(
#    "Cluster Size",
#     1,
#     30,
#     value=2,
#     key="cluster_size"
# )
config_col1, config_col2, config_col3, config_col4 = st.columns([1, 3, 1, 3])
show_overlay = config_col1.checkbox("Show Overlay", value=True)
threshold = config_col2.slider(
    "Threshold",
    1.,
    10., 
    value=2.,
    key="threshold",
    step=0.5
)
# Cluster config
filter_clusters = config_col3.checkbox("Filter Clusters", value=True)
cluster_size = config_col4.slider(
   "Cluster Size",
    1,
    30,
    value=2,
    key="cluster_size"
)

# Get image data
if submit:
    for key in st.session_state.keys():
        del st.session_state[key]
    config = model_options[model_version]
    model = load_model(config)
    with st.spinner("Loading..."):
        st.session_state["slice"] = 128
        st.session_state["threshold"] = 2.
        st.session_state["cluster_size"] = 2
        t1, t2, norm = load_image(batch, subject)
    if t1 is not None:
        st.session_state["t1"] = t1
        st.session_state["t2"] = t2
        if model_version == "T1 Only":
            st.session_state["t1_norm"] = norm
        elif model_version == "T2 Only":
            st.session_state["t2_norm"] = norm
        else:
            st.session_state["t1_norm"] = norm[0]
            st.session_state["t2_norm"] = norm[1]


def get_slice(mode, slice_number):
    img = st.session_state[mode]
    image_pil = Image.fromarray(np.uint8(img[:,:,slice_number-1] * 255), mode="RGBA")
    mode_norm = f"{mode}_norm"
    if show_overlay and (mode_norm in st.session_state):
        norm = st.session_state[mode_norm]
        overlay = get_overlay(norm[:,:,slice_number-1])
        overlay_pil = Image.fromarray(np.uint8(overlay * 255), mode="RGBA")

        return Image.alpha_composite(image_pil, overlay_pil)
    else:
        return image_pil

def send_feedback():
    feedback_db = sqlite3.connect('feedback.db')
    feedback = "".join(ch for ch in st.session_state['feedback'] if ch.isalnum() or ch==" ")
    feedback_db.execute(
        f"""
        INSERT INTO MRI_FEEDBACK
        (BATCH, SUBJECT, MODEL, SLICE, FEEDBACK) \
        VALUES ('{batch}', {subject}, '{model_version}', {slice_number}, '{feedback}')
      """
    )
    feedback_db.commit()
    st.session_state["feedback"] = ""
    feedback_db.close()
    
def get_feedback():
    feedback_db = sqlite3.connect('feedback.db')
    feedback_df = pd.read_sql_query(f"""
        SELECT * FROM MRI_FEEDBACK
        WHERE
        BATCH='{batch}'
        AND
        SUBJECT={subject}
        AND
        MODEL='{model_version}'
        """,
        feedback_db
    )
    feedback_db.close()
    
    return feedback_df


# Show images
if "t1" in st.session_state:
    img_col1, img_col2 = st.columns([1, 1])
    img_col1.image(
        # st.session_state["t1"][:,:,slice_number-1],
        get_slice(
            "t1",
            slice_number
        ),
        use_column_width=True,
        caption="T1",
        output_format="PNG"
    )
    img_col2.image(
        get_slice(
            "t2",
            slice_number
        ),
        use_column_width=True,
        caption="T2"
    )
    cb = st.image(colorbar, output_format="PNG", use_column_width=True)

feedback_input = st.sidebar.text_input(
    "Enter feedback for this subject/slice:",
    key = "feedback",
    on_change=send_feedback
)



feedback_table = st.table(
    get_feedback()
)
