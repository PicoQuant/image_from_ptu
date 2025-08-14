import matplotlib.pyplot as plt
from ptufile import PtuFile
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)

# quick visualisation of which channels have been used
filename = "FLIM_20250715-170852/RawImage.ptu"
ptu = PtuFile(filename)
logging.info(f"ptu file type: {ptu.type}")
logging.info(f"Shape of the data: {ptu.shape}")
frames, _, _, channels, bins = ptu.shape
logging.info(f"Number of frames: {frames}")
logging.info(f"Number of channels: {channels}")
logging.info(f"Number of bins: {bins}")

# Plot single PTU histogram summed over channels to choose dtime window
hist = ptu.decode_histogram(dtime=0, asxarray=False)
logging.info(f"Histogram shape: {hist.shape}")
hist_sum = hist.sum(axis=0)
plt.figure(figsize=(8, 3))
plt.semilogy(hist_sum + 1, label="hist")
plt.title("Histogram (sum over channels) — log y-scale")
plt.xlabel("bin")
plt.ylabel("counts (log)")
plt.tight_layout()
plt.savefig("ptu_histogram_log.png", dpi=300)   
plt.show()

# Window selection: set the number of dtime windows here
N_WINDOWS = 3  # adjust as needed

def compute_windows_from_hist(hist_sum, n_windows: int):
    n = len(hist_sum)
    bounds = np.linspace(0, n, n_windows + 1, dtype=int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_windows)]

windows = compute_windows_from_hist(hist_sum, N_WINDOWS)
logging.info(f"Detected {len(windows)} peak window(s): {windows}")

# Helper: sum image over a delay-time window in chunks to limit memory
def sum_image_over_window(ptu: PtuFile, ch: int, h_start: int, h_end: int, batch: int = 1024) -> np.ndarray:
    """Return 2D image (Y,X) by summing over T and H in [h_start,h_end) for a channel.
    Processes H in batches to avoid loading huge 5D arrays at once.
    """
    t_total_start = time.perf_counter()
    img_sum = None
    n_batches = 0
    for hs in range(h_start, h_end, batch):
        he = min(h_end, hs + batch)
        t_dec_start = time.perf_counter()
        # Integrate frames during decode to avoid materializing the full T axis
        # Keep a wider dtype to avoid overflow when summing many frames/bins
        flim = ptu.decode_image(
            (..., slice(hs, he)),
            channel=ch,
            frame=-1,           # integrate over all frames (T)
            keepdims=False,     # drop the T axis
            dtype="uint32",
            asxarray=False,
        )
        t_dec = time.perf_counter() - t_dec_start
        t_red_start = time.perf_counter()
        # Reduce over histogram axis H.
        # If channel was specified with keepdims=False, flim is (Y, X, H).
        # If keepdims=True or channel not specified, flim may be (Y, X, C, H).
        if flim.ndim == 4:
            part = flim.sum(axis=3)[..., 0]  # (Y,X,C,H)->(Y,X)
        elif flim.ndim == 3:
            part = flim.sum(axis=2)          # (Y,X,H)->(Y,X)
        else:
            raise RuntimeError(f"Unexpected flim.ndim={flim.ndim}, shape={flim.shape}")
        t_red = time.perf_counter() - t_red_start
        if img_sum is None:
            img_sum = part
        else:
            img_sum += part
        n_batches += 1
        logging.info(f"    batch H[{hs}:{he}] ch={ch}: decode={t_dec*1e3:.1f} ms, reduce={t_red*1e3:.1f} ms, shape flim={flim.shape}")
    t_total = time.perf_counter() - t_total_start
    logging.info(f"  sum_image_over_window ch={ch} H[{h_start}:{h_end}] batches={n_batches} total={t_total:.3f} s")
    return img_sum

# Decode and visualize each window for all channels
logging.info("Decoding all windows and channels for grid plot...")
channel_ids = [0, 1,2,  3]
n_win = len(windows)
imgs = [[None for _ in channel_ids] for _ in range(n_win)]
vmax_by_ch = {ch: 0.0 for ch in channel_ids}
for w_idx, (h_start, h_end) in enumerate(windows):
    t_win_start = time.perf_counter()
    logging.info(f"Decoding window {w_idx}: H[{h_start}:{h_end}]")
    for ch in channel_ids:
        t_ch_start = time.perf_counter()
        logging.info(f"  Channel {ch}")
        img = sum_image_over_window(ptu, ch, h_start, h_end, batch=1024)
        imgs[w_idx][ch] = img
        vmax_by_ch[ch] = max(vmax_by_ch[ch], float(img.max()))
        logging.info(f"  Channel {ch} done in {time.perf_counter()-t_ch_start:.3f} s")
    logging.info(f"Window {w_idx} done in {time.perf_counter()-t_win_start:.3f} s")

# Create a single figure: rows = windows, cols = channels
fig, axes = plt.subplots(n_win, len(channel_ids), figsize=(4 * len(channel_ids), 4 * n_win))
if n_win == 1:
    axes = np.array(axes).reshape(1, -1)

for w_idx, (h_start, h_end) in enumerate(windows):
    for col, ch in enumerate(channel_ids):
        ax = axes[w_idx, col]
        vmax = vmax_by_ch[ch] if vmax_by_ch[ch] > 0 else None
        im = ax.imshow(imgs[w_idx][ch], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"Win {w_idx} H[{h_start}:{h_end}] • Ch {ch}")
        fig.colorbar(im, ax=ax, shrink=0.7)

plt.tight_layout()
plt.savefig("channels_overview_grid.png", dpi=300)
plt.show()
