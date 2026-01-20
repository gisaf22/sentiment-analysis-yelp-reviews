# Model Metrics File
# Yelp Review Theme Discovery Pipeline

## Project Overview
- **Project Name**: Yelp Review Theme Discovery
- **Author**: Safari Gisa
- **Date**: January 2026
- **Task Type**: Unsupervised Learning (Clustering + NLP)

---

## Final Model Architecture

### Pipeline Components

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Data Loader | PyArrow with pushdown predicates | Efficient filtered loading from 4GB+ parquet |
| Preprocessor | Custom MyPreProcessor | Text cleaning, normalization |
| Feature Engineer | MyFeatureEngineering | Lemmatization, noun phrase extraction |
| Embedder | Sentence Transformers | Semantic vector representation |
| Context Clusterer | Hierarchical (Agglomerative) | Noise separation |
| Theme Clusterer | KMeans | Theme discovery |
| Labeler | OpenAI GPT-4o-mini | Human-readable theme labels |

---

## Model Parameters & Hyperparameters

### Embedding Model
```
Model: all-MiniLM-L6-v2
Framework: sentence-transformers
Dimensions: 384
Max Sequence Length: 256
Pooling: Mean pooling
```

### Hierarchical Clustering (Business Context)
```
Algorithm: Agglomerative Clustering
Linkage: Average
Distance Metric: Cosine
Cut Height: Automatic (spike detection)
```

### KMeans Clustering (Theme Discovery)
```
Algorithm: KMeans
K Selection: Elbow method (inter-centroid cosine)
K Range: 6-14
Optimal K: 10 (for Datz case study)
Random State: 42
N Init: 20
PCA Normalization: Yes (before clustering)
```

### LLM Theme Labeling
```
Model: GPT-4o-mini
Temperature: 0.4
Response Format: Structured JSON
Output Fields: theme_name, theme_category, summary
Example Reviews: 10 per theme
Sampling Strategy: Diverse (2 closest + 8 stratified)
```

---

## Performance Metrics

### Clustering Quality (Datz Case Study)

| Metric | Value |
|--------|-------|
| Total Reviews | 3,388 |
| Main Context Size | 3,364 (99.3%) |
| Noise Filtered | 24 (0.7%) |
| Themes Discovered | 10 |
| Min Theme Size | 63 reviews |
| Max Theme Size | 953 reviews |
| Theme Size Std | 289.3 |

### K-Selection (Elbow Method)

| K | Inter-Centroid Cosine |
|---|----------------------|
| 6 | 0.42 |
| 7 | 0.48 |
| 8 | 0.53 |
| 9 | 0.57 |
| 10 | 0.61 ← Selected |
| 11 | 0.63 |
| 12 | 0.65 |
| 13 | 0.66 |
| 14 | 0.67 |

### Theme Distribution

| Theme ID | Theme Name | Review Count | Percentage |
|----------|-----------|--------------|------------|
| 1 | Mixed Reactions to Food Quality and Service | 953 | 28.3% |
| 9 | Food Quality and Service Experience | 574 | 17.1% |
| 0 | Datz: A Culinary Gem in South Tampa | 471 | 14.0% |
| 3 | Positive Dining Experience with Great Atmosphere and Service | 457 | 13.6% |
| 4 | Brunch Delight at Datz | 273 | 8.1% |
| 6 | Service Inconsistency and Wait Times | 181 | 5.4% |
| 5 | Breakfast Delights and Unique Offerings | 177 | 5.3% |
| 8 | Fish Tacos and Unique Flavor Combinations | 136 | 4.0% |
| 7 | Reuben Sandwich Enthusiasm | 79 | 2.3% |
| 2 | Mixed Dining Experiences at Datz | 63 | 1.9% |

---

**Example Selection Strategy:**
- 2 reviews closest to centroid (anchors - most typical)
- 8 reviews stratified from safe zone (80th percentile, excludes boundary outliers)
- Result: ~25% more diversity vs closest-only selection

---

## Evaluation Methodology

Since this is an **unsupervised learning** task, traditional train/test splits don't apply. Instead, we evaluate:

### Intrinsic Metrics
1. **Cluster Separation**: Inter-centroid cosine distance (0.61 for k=10) — used for k-selection via elbow method

### Extrinsic Metrics (Human Evaluation)
1. **Interpretability**: Can humans understand theme labels? ✓
2. **Actionability**: Do themes suggest business actions? ✓
3. **Distinctiveness**: Are themes non-overlapping? ✓

### Quality Checks
- [x] All 10 themes received unique LLM labels
- [x] All themes have ≥10 example reviews
- [x] All themes include keywords and noun phrases
- [x] All themes include representative reviews
- [x] No themes contain only noise/foreign language

---

## Feature Engineering

### Text Preprocessing Pipeline

```python
# Features extracted per review:
1. text_cleaned    # Lowercased, normalized, expanded contractions
2. lemmas          # POS-aware lemmatization with negation preservation
3. noun_phrases    # Extracted noun phrases (spaCy)
4. word_count      # Token count
5. char_count      # Character count
```

### Embedding Features
- 384-dimensional dense vector per review
- Mean pooling across all tokens
- L2 normalized for cosine similarity

---

## Reproducibility

```bash
# Run pipeline for any business
python cli.py -b <BUSINESS_ID>

# Example
python cli.py -b QHWYlmVbLC3K6eglWoHVvA
```

**Prerequisites**:
- Python 3.10+
- OpenAI API key in `.env`
- Yelp dataset files in `data/raw/`

---

## Limitations

1. **LLM Dependency**: Theme labeling requires OpenAI API (cost per run)
2. **English Only**: Pipeline optimized for English reviews
3. **Minimum Reviews**: Requires ≥20 reviews for reliable themes

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial release with CLI |
| 1.1 | Jan 2026 | Added PyArrow pushdown predicates |
| 1.2 | Jan 2026 | Fixed KMeans sub-clustering to match notebook flow |
| 1.3 | Jan 2026 | Removed likes/dislikes and business_impact from output schema |
| 1.4 | Jan 2026 | Increased example reviews from 3 to 5 for better LLM context |
| 1.5 | Jan 2026 | Diverse sampling strategy: 2 closest + 3 stratified (excludes boundary outliers) |
| 1.6 | Jan 2026 | Increased to 10 examples: 2 closest + 8 stratified for max diversity |
| 1.7 | Jan 2026 | Refactored: moved elbow method from pipeline to ClusteringEvaluator |
| 1.8 | Jan 2026 | Added review_assignments for temporal analysis; restructured output to results/{business_id}/ |