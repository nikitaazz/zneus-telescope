# **ZNEUS_01 - Magic Telescope (Binary Classification)**

## Week 1 - Exploratory Data Analysis and Initial Statistical Testing

### Dataset
- **Source**: [Magic Telescope Dataset](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)
- **Description**: The dataset contains 10 features from telescope observations, classified as **'g' - gamma (*signal*)** or **'h' - hadron (*background*)**.
- **Features**: 10 numerical features (e.g., fLength, fWidth, fSize, etc.) and 1 categorical target ('class').
- **Size**: 18905 rows after preprocessing (ID column dropped), dropped duplicates.
- **Preprocessing**:
  - Loaded dataset using `scipy.io.arff`.
  - Normalized column names (stripped whitespace, removed special characters, lowercased).
  - Decoded 'class' from bytes to strings.
  - Stripped whitespace from string column names.
  - Encoded 'class' to binary: 'g' → 1, 'h' → 0.

### EDA Methodology

**1. Descriptive Statistics**
- Examined dataset shape, data types, unique values, duplicates, and missing values.
- Generated summary statistics (mean, std, min, max, etc.) for numerical features.

**2. Outlier Detection and Handling**
- Visualized distributions using histograms and boxplots.
- Detected outliers using IQR method (1.5 * IQR).
- Applied Box-Cox transformation to reduce skewness and outliers for features with high outlier counts (fAsym, fWidth, fM3Long, fLength, fM3Trans, fConc1, fDist, fSize).
- Alternative approach: Dropped rows with outliers (commented out in code).

**3. Correlation Analysis**
- Computed pairwise correlations between all numerical features.
- Identified strongest correlations with 'class' (ordered by absolute value).
- Visualized correlation matrix using a heatmap.

**4. Comprehensive Visualizations**
- For each feature (excluding 'class'), created multi-panel plots:
  - Histogram + KDE for all data.
  - Overlaid histograms/KDE split by class.
  - Boxplots (all data and by class).
  - KDE with statistical tests (Shapiro-Wilk normality, Levene's variance equality, Cohen's d effect size).
  - Q-Q plots for normality assessment.
  - General summary statistics (mean, median, variance, standard deviation).
  - Additional summary statistics (Skewness, Kurtosis, Excess Kurtosis).
- Features visualized in order of correlation strength with 'class'.

**5. Hypothesis Testing**
- Tested differences in feature distributions between classes (0: 'h', 1: 'g').
- Features tested: fAlpha, fLength, fDist.
- Tests performed:
  - Shapiro-Wilk for normality.
  - Levene's test for variance equality.
  - Mann-Whitney U test for median differences (due to non-normality).
  - Cohen's d for effect size.
  - Statistical power calculation.
- Hypothesis: Class=1 ('g') has lower average values than class=0 ('h').

### Results
- **Class Distribution**: 12,332 samples for class=1 ('g'), 6,688 for class=0 ('h').
- **Outliers**: High counts in fAsym, fWidth, etc.; Box-Cox reduced skewness.
- **Correlations**: Strongest with 'class' (ordered by absolute value):
  
  | Feature   | Abs Correlation | Signed Correlation |
  |-----------|-----------------|---------------------|
  | fAlpha    | 0.460979       | -0.460979         |
  | fLength   | 0.204748       | -0.204748         |
  | fWidth    | 0.161381       | -0.161381         |
  | fM3Long   | 0.143027       | 0.143027          |
  | fAsym     | 0.139580       | 0.139580          |
  | fSize     | 0.127929       | -0.127929         |
  | fDist     | 0.058541       | -0.058541         |
  | fConc1    | 0.028271       | 0.028271          |
  | fConc     | 0.024615       | 0.024615          |
  | fM3Trans  | 0.009237       | -0.009237         |
- **Hypothesis Testing**:
  - fAlpha: Significant difference (class=0 higher), medium effect.
  - fLength: Significant difference (class=0 higher), small effect.
  - fDist: Significant difference (class=0 higher), small effect.
- Visualizations reveal distinct distributions by class, aiding feature selection for modeling.

## Week 2 - Data Preprocessing and Initial Model

### Design Decisions

**Notebook Organization**
- Separated EDA (WEEK1-EDA.ipynb) from modeling (WEEK23-MODEL.ipynb) to maintain clean, focused workflows.
- EDA notebook retained comprehensive statistical analysis and visualizations.
- Modeling notebook focuses on preprocessing pipeline, model architecture, and training.

**Outlier Handling Decision**
- We have decided to remove outliers using 1.5*IQR method poposed in our EDA

**Utility Functions (utils.py)**
- Created `load_data()` function for consistent data loading across notebooks.
- Implemented `drop_outliers_iqr()` for optional IQR-based outlier removal (used in final pipeline).
- Standardized column normalization, class encoding, and preprocessing steps.

### Data Preprocessing Pipeline

**Split Configuration**
```python
SPLIT_CONF = {
    'train_size': 0.70,    # 13,314 samples
    'val_size': 0.15,      # 2,853 samples  
    'test_size': 0.15,     # 2,853 samples
    'stratify': True       # Maintains class distribution
}
```
- Stratified splitting ensures balanced class representation across all sets.
- Two-step split: first separate test set, then split remaining into train/validation.

**Preprocessing Pipeline (sklearn)**
```python
PREPROCESSING_CONFIG = {
    'method': 'quantile',             # QuantileTransformer
    'output_distribution': 'normal',  # Gaussian output
    'remove_low_variance': True,
    'variance_threshold': 0.00
}
```
**Pipeline Steps:**
1. **Variance Threshold Filter**: Remove zero-variance features (if any).
2. **Quantile Transformer**: Non-linear transformation mapping to normal distribution.
   - Robust to outliers.
   - Ensures uniform feature scaling for neural network training.

**Feature Engineering**
- Original 10 numerical features preserved after transformation.
- No dimensionality reduction applied (all features contribute to classification).
- Features normalized to standard Gaussian distribution.

### Initial MLP Architecture
- We have proposed our first steps to

### First Experiment Results

## Week 3 - Architecture Fine-tuning and Experiment Tracking

### Configuration-Driven Experiments

**Dictionary-Based Design Pattern**
- Implemented configuration dictionaries (`MODEL_CONFIG`, `TRAINING_CONFIG`, `WANDB_CONFIG`) for reproducible experiments.
- Each experiment variant stored as separate config dictionary (e.g., `MODEL_CONFIG_01_A`, `MODEL_CONFIG_01_B`).
- Easy hyperparameter tracking and experiment comparison.

**ModularMLP & MLPTrainer Enhancements**
- **ModularMLP**: Supports dynamic architecture configuration through dictionary input.
- **MLPTrainer**: 
  - Accepts training configuration with automatic wandb integration.
  - Saves best models with full checkpoint (optimizer state, metrics, history).
  - Built-in visualization methods (`plot_history()`, `evaluate()`).

### Experiment Series

**Series 01: Adam Optimizer (4 experiments)**
- Architecture: [256, 128, 128, 64] with Leaky ReLU
- Learning Rate: 0.005
- Variants:
  - **01-A (Baseline)**: No dropout, no batch norm
  - **01-B (Dropout)**: Dropout 0.2, no batch norm
  - **01-C (BatchNorm)**: No dropout, with batch norm
  - **01-D (Both)**: Dropout 0.2 + batch norm

**Series 02: SGD Optimizer (4 experiments)**
- Architecture: [256, 128, 128, 64] with Leaky ReLU
- Learning Rate: 0.05, Momentum: 0.9
- Same regularization variants as Series 01

**Series 03: RMSprop Optimizer (4 experiments)**
- Architecture: [256, 128, 128, 64] with Leaky ReLU
- Learning Rate: 0.005
- Same regularization variants as Series 01

**Total Experiments**: 12 systematic comparisons exploring:
- Optimizer selection (Adam vs. SGD vs. RMSprop)
- Regularization techniques (Dropout, Batch Normalization, combined)
- Architecture depth and width

### Weights & Biases Integration

**Experiment Tracking Setup**
```python
WANDB_CONFIG = {
    'project': 'magic-telescope-classification',
    'entity': None,  # Personal workspace
    'name': 'exp_01_a_adam_baseline',
    'tags': ['mlp', 'adam', 'baseline'],
    'notes': 'Adam optimizer - No regularization baseline'
}
```

**Tracked Metrics (per epoch):**
- Training: loss, accuracy, precision, recall, F1-score, ROC-AUC
- Validation: loss, accuracy, precision, recall, F1-score, ROC-AUC
- Learning rate, epoch number

**Experiment Organization:**
- 12 wandb runs logged to `magic-telescope-classification` project.
- Tagged by optimizer type, regularization strategy.
- Saved model checkpoints: `saved_models/exp_*.pth`

### Visualizations

**Training History Plots**
- 6-panel visualization per experiment:
  1. Loss curves (train vs. validation)
  2. Accuracy curves
  3. Precision curves
  4. Recall curves
  5. F1-Score curves
  6. ROC-AUC curves
- Color-coded train (blue) vs. validation (orange) metrics.
- Grid layout for comprehensive performance monitoring.

**Test Evaluation Visualizations**
- **Confusion Matrix**: Heatmap with annotations (gamma vs. hadron classification).
- **Metrics Bar Chart**: Side-by-side comparison of accuracy, precision, recall, F1, AUC.
- Classification report with per-class statistics.

**Comparative Analysis**
- Enabled side-by-side comparison of 12 experiments through wandb dashboard.
- Identified best-performing configurations based on validation F1-score.

### Key Findings & Architecture Refinements

**Optimizer Performance:**
- Adam optimizer showed fastest convergence with lowest variance.
- SGD required higher learning rate but achieved competitive final performance.
- RMSprop demonstrated stable training with moderate learning rates.

**Regularization Impact:**
- Batch normalization significantly improved training stability.
- Dropout effectively reduced overfitting in deeper architectures.
- Combined regularization (dropout + batch norm) yielded best generalization.

**Architecture Insights:**
- Deeper networks [256, 128, 128, 64] outperformed shallow baseline [64, 32].
- Leaky ReLU activation preferred over tanh for gradient flow.
- Early stopping (patience=30) prevented unnecessary training epochs.

**Best Model Selection:**
- Model checkpoints saved based on validation F1-score.
- All 12 experiments archived in `saved_models/` for reproducibility.
- Full checkpoint includes model weights, optimizer state, training history, and configuration.

---

### Authors
- Matej Herzog, Nosenko Mykyta
- Project Repository: [zneus-telescope](https://github.com/nikitaazz/zneus-telescope)