import streamlit as st
import segyio
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")

st.title("📈 2D Seismic Line Viewer")

# --- File uploader ---
input_file = st.file_uploader("Upload a SEGY file", type=["sgy", "segy"])

# --- Sidebar for user inputs ---
st.sidebar.header("⚙️ Settings")

x_byte = st.sidebar.number_input("X Coordinate Byte", value=73, step=1)
y_byte = st.sidebar.number_input("Y Coordinate Byte", value=77, step=1)
cdp_byte = st.sidebar.number_input("CDP Byte", value=21, step=1)
shot_byte = st.sidebar.number_input("Shot Point Byte", value=17, step=1)

plot_width = st.sidebar.slider("Plot Width (inches)", min_value=8, max_value=30, value=15, step=1)
plot_height = st.sidebar.slider("Plot Height (inches)", min_value=4, max_value=20, value=8, step=1)

# --- Amplitude scaling ---
st.sidebar.subheader("Amplitude Scale")
amp_min = st.sidebar.number_input("Min Amplitude", value=-5000, step=100)
amp_max = st.sidebar.number_input("Max Amplitude", value=5000, step=100)
auto_scale = st.sidebar.checkbox("Auto Scale (use data min/max)", value=True)

# --- Only run if file uploaded ---
if input_file is not None:
    file_name = os.path.basename(input_file.name)

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(input_file.read())
        tmp_path = tmp.name

    with segyio.open(tmp_path, "r", ignore_geometry=True) as f:
        st.success("File opened successfully ✅")
        n_traces = f.tracecount
        n_samples = f.samples.size
        st.write(f"**Traces:** {n_traces}, **Samples per trace:** {n_samples}")

        # --- Show Textual Header in Sidebar ---
        textual_header = segyio.tools.wrap(f.text[0])  # 3200-byte textual header
        with st.sidebar.expander("📜 Textual Header", expanded=False):
            st.text(textual_header)

        # --- Read header values ---
        cdps = np.array([f.header[i][cdp_byte] for i in range(n_traces)])

        # --- Load traces ---
        data = np.stack([f.trace[i] for i in range(n_traces)])
        data = data.T  # shape = (samples, traces)
        times = f.samples

    # --- Auto scaling ---
    if auto_scale:
        amp_min, amp_max = float(np.min(data)), float(np.max(data))

     # --- Layout: 2 cols for Seismic + Histogram ---
    col1, col2 = st.columns([3, 1])
    plt.style.use('ggplot')
    # --- Seismic Section ---
    with col1:
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        cax = ax.imshow(
            data,
            cmap="seismic",
            aspect="auto",
            extent=[cdps.min(), cdps.max(), times.max(), times.min()],
            vmin=amp_min,
            vmax=amp_max
        )
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlabel("CDP")
        ax.set_ylabel("Time (ms)")
        ax.set_title(file_name)
        plt.grid(False)
        plt.colorbar(cax, ax=ax, label="Amplitude")
        st.pyplot(fig)

    # --- Histogram ---
    with col2:
    # --- Amplitude Spectrum ---
        dt = (times[1] - times[0]) / 1000.0  # sample interval in seconds
        fs = 1.0 / dt  # sampling frequency

        fft_data = np.fft.rfft(data, axis=0)
        amp_spectrum = np.abs(fft_data)
        mean_amp_spectrum = np.mean(amp_spectrum, axis=1)
        freqs = np.fft.rfftfreq(data.shape[0], d=dt)

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.plot(freqs, mean_amp_spectrum, color="black")
        ax.set_title("Average Amplitude Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)
