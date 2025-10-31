# **ZNEUS_01 - Magic Telescope (Binary Classification)**

## Week 1 - Exploratory Data Analysis and Initial Statistical Testing

### Dataset
- **Source**: [Magic Telescope Dataset](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)
- **Description**: The dataset contains 10 features from telescope observations, classified as **'g' - gamma (*signal*)** or **'h' - hadron (*background*)**.
- **Features**: 10 numerical features (e.g., fLength, fWidth, fSize, etc.) and 1 categorical target ('class').
- **Size**: 19,020 rows after preprocessing (ID column dropped).
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

### Authors
- Matej Herzog, Nosenko Mykyta
- Project Repository: [zneus-telescope](https://github.com/nikitaazz/zneus-telescope)