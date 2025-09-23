handwritten_digit_recognition/
│
├── data/                     # Raw and preprocessed datasets
│   ├── raw/                  # Original USPS/MNIST dataset files
│   ├── processed/            # Preprocessed .npy or .pt files (28x28 normalized)
│   └── data_loader.py        # Utilities for loading & batching data
│
├── models/                   # Network components & full model
│   ├── layers.py             # Convolution, subsampling, dense layers, activations
│   ├── network.py            # Full architecture (H1 → H4 → output)
│   └── utils.py              # Weight initialization, gradient checks, etc.
│
├── training/                 
│   ├── train.py              # Training loop (forward, backward, update)
│   ├── loss.py               # MSE loss (with +1/-1 targets)
│   └── optim.py              # SGD, momentum (optional later)
│
├── experiments/
│   ├── run_baseline.py       # Train/evaluate the baseline network
│   ├── run_with_rejection.py # Add rejection rule
│   └── results/              # Store logs, metrics, plots
│
├── notebooks/                # Jupyter notebooks for exploration & visualization
│   ├── data_visualization.ipynb
│   ├── layer_debugging.ipynb
│   └── training_curves.ipynb
│
├── tests/                    # Unit tests (optional but useful)
│   ├── test_layers.py        # Test forward/backward consistency
│   └── test_utils.py         # Gradient checks, preprocessing checks
│
├── main.py                   # Entry point for running training end-to-end
├── requirements.txt          # Dependencies (NumPy, Matplotlib, etc.)
└── README.md                 # Project description & instructions
