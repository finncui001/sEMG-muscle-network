# sEMG Muscle Network

A Python tool for building muscle functional networks from multi-channel surface EMG signals. It provides coherence analysis using Welch’s method and mutual information analysis based on Shannon entropy. It also calculates graph metrics and identifies key muscles using node contraction.

## Features

- 50 Hz notch and 20–450 Hz band-pass filtering
- Coherence and mutual information networks
- Clustering, efficiency, path length, and small-world metrics
- Network plots and key-muscle rankings

## Usage

Install dependencies:

```bash
pip install numpy scipy networkx matplotlib
```

Run the program:

```bash
python "sEMG muscle function network analysis V1.0.py"
```

Select a TXT or CSV file when prompted. Results are saved in `network_output` beside the input file.

## Note

The mutual information method is inspired by Chen et al. (2019). Default settings use 32 histogram bins and a network threshold of 0.50.
