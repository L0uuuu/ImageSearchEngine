# Image Search Engine Mini-Project

## Overview

This is an **Image Search Engine** built with Python and Tkinter, designed to find visually similar images in a dataset. It features a modern black-and-white GUI and supports multiple descriptors:

- **Color Histogram**: Uses `[8, 8, 8]` bins for RGB channels.
- **Grayscale Histogram**: Computes grayscale intensity histogram.
- **Correlogram**: Extracts texture features via co-occurrence matrices.
- **VGG16**: Deep learning features with GPU support (NVIDIA GTX 1660 Super).

The GUI includes a sidebar for controls and a scrollable grid for results. Search time, precision, and recall are displayed after each search.

## Project Structure

```
image_search_engine/
├── dataset/                    # Dataset images (e.g., r2_140_100.jpg)
├── main.py                     # Main script with GUI and logic
├── requirements.txt            # Dependencies
├── vgg16_features.pkl          # Cached VGG16 features
└── README.md                   # This file
```

## Usage

1. Run `main.py` to launch the GUI.
2. Select a descriptor (e.g., "VGG16").
3. Upload a query image (e.g., `r2_140_100.jpg`).
4. Click "Search" to view the top 5 similar images, along with search time and evaluation metrics.

### Evaluation
- Ground truth: Images with the exact same filename (without extension) as the query (e.g., `r2_140_100`).
- Metrics example: `Precision: 0.20, Recall: 1.00` (due to strict ground truth).

## Limitations

- Ground truth is restrictive, often leading to low precision.
- Dataset imbalance: 26 `r_` images vs. 6 `r2_` images, which may bias results.

## Future Improvements

- Redefine ground truth using filename prefixes or metadata.
- Balance dataset with more categories.
- Test with external query images.

## License

MIT License. See the [LICENSE](LICENSE) file for details.