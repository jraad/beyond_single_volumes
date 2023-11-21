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
from model_architectures import VAESegment, Data3DSegToSeg

reload(utilities)
from utilities import get_colormaps

# FOR SHARING
# - use localtunnel to expose port to internet

# Define paths
research_dir = r"D:/school/research"
code_dir = os.path.join(research_dir, "code")
model_dir = os.path.join(code_dir, "paper_two_code", "models")
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
train = Data3DSegToSeg(l1_dir, l1_seg_vol_dir, train_indices)
val = Data3DSegToSeg(l1_dir, l1_seg_vol_dir, val_indices)
test = Data3DSegToSeg(l5_dir, l5_seg_vol_dir, test_indices)


# Additional data
model_options = {
    "9 Segments to 9 Segments": {
        "weights_t1": "vae_rel2t1_seg9_to_seg9.pt",
        "weights_t2": "vae_rel2t2_seg9_to_seg9.pt",
        "model": VAESegment(9, 9),
        "shape": (1, 9, 256, 256, 256)
    },
    "1 Segment to 1 Segment": {
        "weights_t1": "vae_rel2t1_seg_to_seg{}.pt",
        "weights_t2": "vae_rel2t2_seg_to_seg{}.pt",
        "model": VAESegment(1, 1),
        "shape": (1, 1, 256, 256, 256)
    }
}
max_z = 10
reds, blues = get_colormaps(max_z)
colorbar = Image.open("colorbar.png")
segments = [
    "Cerebrospinal Fluid",
    "Cortical Grey Matter",
    "White Matter",
    "Background",
    "Ventricle",
    "Cerebelum",
    "Deep Grey Matter",
    "Brainstem",
    "Hippocampus"
]

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
    
    # T1
    og_t1 = np.stack((data[0],)*3, axis=-1)
    og_t1 = np.rot90(og_t1, axes=(1,2))
    og_t1 = np.concatenate((og_t1, np.ones((9, 256, 256, 256, 1))), axis=4)

    norm_t1 = get_prediction(data[0], model_t1)
    norm_t1 = np.rot90(norm_t1, axes=(1,2))
    
    # T2
    og_t2 = np.stack((data[1],)*3, axis=-1)
    og_t2 = np.rot90(og_t2, axes=(1,2))
    og_t2 = np.concatenate((og_t2, np.ones((9, 256, 256, 256, 1))), axis=4)

    norm_t2 = get_prediction(data[1], model_t2)
    norm_t2 = np.rot90(norm_t2, axes=(1,2))

    return og_t1, norm_t1, og_t2, norm_t2

def load_model(config, version):
    model_path = os.path.join(model_dir, config[f"weights_{version}"])
    model = config["model"]
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    
    return model

def load_models_data(config, batch, subject, modality):
    mode = 1 if modality == "t2" else 0
    data = get_dataset(batch)[int(subject)-1][mode]
    norm = np.empty_like(data)
    for idx in range(0, len(segments)):
        img = data[idx]
        model_path = os.path.join(model_dir, config[f"weights_{modality}"].format(idx))
        model = config["model"]
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        
        pred = model(torch.Tensor(np.reshape(img, config["shape"])).cuda())
        norm_slc = img - np.reshape(pred.cpu().detach().numpy(), img.shape)
        
        if stdev_version == "Global":
            norm[idx] = norm_slc / 0.1
        else:
            norm[idx] = (norm_slc - np.mean(norm_slc)) / np.std(norm_slc)

    og = np.stack((data,)*3, axis=-1)
    og = np.rot90(og, axes=(1,2))
    og = np.concatenate((og, np.ones((9, 256, 256, 256, 1))), axis=4)

    # norm /= 0.1
    norm = np.rot90(norm, axes=(1,2))
    
    
    return og, norm


def get_prediction(img, model):
    pred = model(torch.Tensor(np.reshape(img, config["shape"])).cuda())
    diff = img - np.reshape(pred.cpu().detach().numpy(), img.shape)
    
    if stdev_version == "Global":
        diff /= 0.1
    else:
        means = [np.mean(x) for x in diff]
        stdevs = [np.std(x) for x in diff]
        for i in range(0, 9):
            diff[i] = (diff[i] - means[i]) / stdevs[i]

    return diff

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
    
# Define page
st.set_page_config(layout="wide")
st.title("MRI Visualizer")
batch = st.sidebar.selectbox(
    "Batch",
    options=["Train (Normal)", "Validation (Normal)", "Test (Abnormal)"],
    key="batch"
)
subject = st.sidebar.selectbox(
    "Subject",
    options=range(1, len(get_dataset(batch))+1),
    key="subject"
)
model_version = st.sidebar.selectbox(
    "Model Version",
    options=model_options.keys(),
    key="model_version"
)
stdev_version = st.sidebar.radio(
    "Standard Deviation Type:",
    ("Sample-Wise", "Global")
)
submit = st.sidebar.button("Run")

# Slice slider
slice_number = st.slider(
    "Slice Number",
    1,
    256,
    value=128,
    key="slice_number",
)

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
segment_value = st.selectbox(
    "Segment",
    options=segments,
    key="segment_value"
)

if submit:
    og_t2 = None
    for key in st.session_state.keys():
        del st.session_state[key]
    config = model_options[model_version]
    if model_version == list(model_options.keys())[0]:
        model_t1 = load_model(config, "t1")
        model_t2 = load_model(config, "t2")
        with st.spinner("Loading..."):
            og_t1, norm_t1, og_t2, norm_t2 = load_image(batch, subject)
    else:
        og_t1, norm_t1 = load_models_data(config, batch, subject, "t1")
        og_t2, norm_t2 = load_models_data(config, batch, subject, "t2")
    if og_t2 is not None:
        st.session_state["og_t1"] = og_t1
        st.session_state["norm_t1"] = norm_t1
        st.session_state["og_t2"] = og_t2
        st.session_state["norm_t2"] = norm_t2


def get_slice_segment(segment, slice_number, modality):
    img = st.session_state[f"og_{modality}"][segment]
    image_pil = Image.fromarray(np.uint8(img[:,:,slice_number-1] * 255), mode="RGBA")
    
    if show_overlay and (f"norm_{modality}" in st.session_state):
        norm = st.session_state[f"norm_{modality}"]
        overlay = get_overlay(norm[segment][:,:,slice_number-1])
        overlay_pil = Image.fromarray(np.uint8(overlay * 255), mode="RGBA")

        return Image.alpha_composite(image_pil, overlay_pil)
    else:
        return image_pil


def get_slice(slice_number, modality):
    img = np.sum(st.session_state[f"og_{modality}"], axis=0)
    image_pil = Image.fromarray(np.uint8(img[:,:,slice_number-1] * 255), mode="RGBA")
    
    if show_overlay and (f"norm_{modality}" in st.session_state):
        norm = st.session_state[f"norm_{modality}"]
        overlay = get_overlay(np.sum(norm, axis=0)[:,:,slice_number-1])
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


# Visualize image
if "og_t2" in st.session_state:    
    cols = st.columns([2, 2])
    cols[0].image(
        get_slice(slice_number, "t1"),
        use_column_width=True,
        caption="T1",
        output_format="PNG"
    )
    cols[1].image(
        get_slice(slice_number, "t2"),
        use_column_width=True,
        caption="T2",
        output_format="PNG"
    )

# Seg visuals
if "og_t2" in st.session_state:
    seg_cols = st.columns([2, 2])
    seg_cols[0].image(
        get_slice_segment(segments.index(segment_value), slice_number, "t1"),
        use_column_width=True,
        caption=f"T1 {segment_value}",
        output_format="PNG"
    )
    seg_cols[1].image(
        get_slice_segment(segments.index(segment_value), slice_number, "t2"),
        use_column_width=True,
        caption=f"T2 {segment_value}",
        output_format="PNG"
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

