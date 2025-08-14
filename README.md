# PTU FLIM Channel/Window Plotter

**Author**: Steffen Ruettinger 

**Organization**: PicoQuant GmbH, Berlin, Germany 

Version: 2025.0.1


# Summary

This repository contains a small utility script to quickly visualize which FLIM channels were active in a PicoQuant PTU file and to explore time-delay (dtime) windows. It produces:

- A log-scale histogram of the PTU counts summed over channels.
- A grid of images: rows = selected dtime windows, columns = detector channels.

Script: plot_ptu_channel_windows.py

## Disclaimer

This repository contains experimental Python code originally developed for internal research purposes.  
 By using this code, you acknowledge and agree to the following:

1. **No Warranty** – Provided “as is”, without express or implied warranties (including merchantability, fitness for a particular purpose, or non-infringement). Use at your own risk.
2. **Experimental Nature** – Not fully tested, optimized, or production-ready. Intended for developers comfortable working with experimental software.
3. **Limited Support** – No guarantee of active maintenance or timely responses to issues, bugs, or pull requests.
4. **Liability** – The authors accept no responsibility for any loss or damage caused by the use of this code. You are responsible for ensuring it meets your needs and complies with applicable laws.
5. **Contributions** – Contributions are welcome. All submissions will be subject to the same license and disclaimers as the original project.

## Features

- Load a PTU file via PtuFile (https://github.com/cgohlke/ptufile).
- Compute simple, evenly spaced dtime windows (configurable count).
- Sum frames and dtime-bins into a 2D image for each channel and window.
- Save publication-ready PNGs.

## Requirements

- Python 3.9+ (tested versions may vary)
- numpy
- matplotlib
- ptufile (provides PtuFile for PTU file handling) - https://github.com/cgohlke/ptufile

## Installation

This script is based on PicoQuant and  related PTU files, therefore it is necessary to install the following dependencies:

```bash
python -m pip install -U "ptufile[all]"
```

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install numpy matplotlib ptufile
```

## Usage

1. Place or reference your PTU file (e.g., FLIM_20250715-170852/RawImage.ptu).
2. Open plot_ptu_channel_windows.py and adjust:
   - filename: path to the PTU file.
   - N_WINDOWS: number of evenly spaced dtime windows.
   - channel_ids: list of channel indices to visualize.
3. Run the script:

```bash
python plot_ptu_channel_windows.py
```

## Output

- ptu_histogram_log.png: log-scale histogram of counts summed over channels.
- channels_overview_grid.png: grid of images with rows = windows and columns = channels.

## Configuration Details

- Windowing: compute_windows_from_hist() currently splits the histogram range into N_WINDOWS equal-width windows. You can replace this with peak-based window detection if desired.
- Performance: The script decodes only the requested dtime window and integrates frames to keep memory usage manageable. Logging provides timing for decode and reduction steps.

