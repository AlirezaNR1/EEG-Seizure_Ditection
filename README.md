# EEG-Seizure-Detection
An intelligent system for detecting seizures in epileptic patients through EEG signal analysis and classification

## Project Overview

In this project I implement an intelligent system for detecting seizures in epileptic patients using EEG signals. The process involves multiple phases, including preprocessing of EEG signals, feature extraction, signal classification, feature selection, and the incorporation of clustering to improve accuracy.

## Project Phases

### Phase 1: Preprocessing and Feature Extraction

#### 1.1 Preprocessing
- Raw EEG signals are preprocessed to enhance the quality of the data for further analysis.

#### 1.2 Feature Extraction
**Categories:**
- Time Domain Features: Peek to Peek(PTP), AASS, Singular Spectrum Analysis(SSA), Log Detect, Zero Crossings(ZC)
- Statistical Features: Mean, Median, Percentile, Standard Derivation(STD), Histogram
- Entropy Features: Sample Entropy, Approximate Entropy, Spectral Entropy, Permutation Entropy, Singular Value Decomposition Entropy(SVD)

**Documentation:**
- Detailed documentation for each feature is available in the `reports/feature_extraction.pdf` folder.

#### 1.3 Signal Classification
- Implement three classification algorithms: SVM, Random Forest, KNN
- Evaluate each model using k-fold cross-validation (k=5) with metrics: Accuracy, Precision, Recall.
- Apply normalization to enhance model performance.

### Phase 2: Feature Selection

#### 2.1 Evaluating Features Independently
To understand how well each feature acts on its own, a decision tree classifier is employed to classify the data using only one feature. The output accuracy serves as a measure of feature quality.

#### 2.2 Measuring Feature Distinctiveness
The goal is to evaluate how different each feature is from the others. This is achieved through correlation analysis, conducting a two-by-two comparison of single features. A good feature is identified by having the least correlation with others.

#### 2.3 Feature Selection
The results from the independent evaluation and distinctiveness analysis are combined using a specific calculation:
score = (1 / normalized correlation * accuracy) / (1 / normalized correlation + accuracy)
Features with higher scores are selected for further stages in the project. 
For a more detailed understanding, refer to the documentation provided in the `reports/feature_selection.pdf` folder.

#### 2.4 Clustering for Accuracy Enhancement 
In this stage, the focus is on enhancing the accuracy and speed of the model through clustering. The dataset is partitioned into different parts using a clustering algorithm. The previously trained system is then applied separately to each part during testing. This results in a unique system for diagnosing seizures.

K-Means, a centroid-based algorithm, is chosen for clustering due to its ability to minimize the variance of data points within a cluster.

#### 2.5 Evaluation with Different Issues
To comprehensively assess the algorithm's performance, evaluations are conducted with different class modes. This includes creating a three-class mode initially and testing the algorithm's performance. Subsequently, three two-class modes are created, and the algorithm is tested on each.
For a more in-depth understanding of the evaluation processes, refer to the detailed document provided in the `reports/ evaluation_with_different_classes` folder.



