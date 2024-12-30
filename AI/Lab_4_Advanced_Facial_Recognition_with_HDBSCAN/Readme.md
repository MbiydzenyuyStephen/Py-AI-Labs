Image Clustering with HDBSCAN

## Overview

This project aims to cluster images using the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) algorithm. The project involves loading images from a dataset, preprocessing them, reducing their dimensionality, applying the HDBSCAN clustering algorithm, visualizing the results, and evaluating the clustering quality.

## Table of Contents

1. [Project Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Detailed Steps](#detailed-steps)
   - [Step 1: Load Images](#step-1-load-images)
   - [Step 2: Visualize Images](#step-2-visualize-images)
   - [Step 3: Preprocess Images](#step-3-preprocess-images)
   - [Step 4: Dimensionality Reduction](#step-4-dimensionality-reduction)
   - [Step 5: Apply HDBSCAN Clustering](#step-5-apply-hdbscan-clustering)
   - [Step 6: Visualize and Analyze Results](#step-6-visualize-and-analyze-results)
   - [Step 7: Test on New Data](#step-7-test-on-new-data)
   - [Step 8: Identify Representative Images](#step-8-identify-representative-images)
6. [Requirements](#requirements)
7. [References](#references)

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/image-clustering-hdbscan.git
   cd image-clustering-hdbscan
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset of images in a directory (e.g., [dataSet](http://_vscodecontentref_/0)).
2. Update the `data_dir` variable in [.py](http://_vscodecontentref_/1) to point to your dataset directory.
3. Run the script:
   ```sh
   python main.py
   ```

## Project Structure
