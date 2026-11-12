# Adaptive KNN: Learning the Right Number of Neighbors

## Setup Instructions

To run the code and experiments locally, follow these steps:

1. **Clone the repository:**
git clone https://github.com/YOUR_USERNAME/adaptive-knn.git
cd adaptive-knn


2. **Create a virtual environment (using Anaconda):**
conda create -n adaptive_knn python=3.10
conda activate adaptive_knn


3. **Install dependencies:**
pip install -r requirements.txt



## Project Overview

K-Nearest Neighbors (KNN) is a simple and effective classification algorithm, but conventional KNN relies on one globally fixed neighborhood size for every query. This assumption can be inappropriate when different regions of a dataset possess varying local densities or class structures.

This project explores whether dynamically selecting $K$ based on local data characteristics can improve KNN classification performance compared to using a single, globally fixed $K$.

## Base Paper & Inspiration

This project is inspired by the following 2026 Nature Communications paper:

* **Paper:** Li, J., Xu, H., Zhang, S. (2026), "Adaptive kNN graph model," Nature Communications 17, 7491.


* **Link:** [https://www.nature.com/articles/s41467-026-74296-2](https://www.nature.com/articles/s41467-026-74296-2)


* **Original Code:** [https://github.com/Lijy207/kNN-Graph](https://github.com/Lijy207/kNN-Graph)


* **Original Data/Code Archives:** Zenodo DOI 10.5281/zenodo.19531320 (datasets) and 10.5281/zenodo.20236665 (code release).



**What the base paper does:** The original paper proposes a complex adaptive graph framework where neighborhood sizes and neighbor weights are learned for individual training samples. This learned information is then embedded into an Hierarchical Navigable Small World (HNSW) graph to shift the computational burden to the training stage.

## Our Contribution & Methodology

Instead of reproducing the complete, computationally heavy HNSW system from the reference paper, our project builds a simpler, semester-scale extension.

**What we worked on:**
Our main contribution is a lightweight adaptive-K classifier that extends the fixed-neighborhood KNN paradigm using local density calculations.

The methodology includes:

* **Local Density Estimation:** We estimate the local density around a query point by measuring distances to a small reference neighborhood.


* **Adaptive Mapping:** That density is mapped to an adaptive $K$ value. Dense regions utilize smaller neighborhoods, while sparse regions utilize larger neighborhoods, all subject to predefined bounds.


* **Voting Mechanisms:** The dynamically selected neighbors are then classified using either majority voting or distance-weighted voting.



## Experiments and Evaluation

To thoroughly test our lightweight adaptive classifier against standard baselines, we conduct a controlled empirical analysis to understand exactly when adaptive neighborhoods help or hurt.

Our evaluation pipeline includes:

* **Baseline Comparisons:** Comparing our model against standard KNN using fixed $K$ values (e.g., $K=1, 3, 5, 7, 9, 11, 15$) and distance-weighted KNN.


* **Robustness Testing:** Evaluating the method's robustness to injected feature and point noise at several different levels.


* **Imbalance & Density Analysis:** Conducting class-imbalance analysis and comparing performance across multiple public classification datasets with diverse dimensionalities, class distributions, and density structures.


* **Metrics:** Evaluating performance using Accuracy, Precision, Recall, F1-score, confusion matrices, and runtime. We also analyze the average and distribution of the dynamically selected $K$ values across the feature space. Error analysis will examine cases where adaptive selection changes the prediction relative to the best fixed-K baseline.