# 🎯 Yelp Review Theme Discovery

> **Automatically extract and summarize customer themes from thousands of Yelp reviews using NLP, clustering, and LLM-powered labeling.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [Solution Overview](#-solution-overview)
- [Example Results](#-example-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Future Work](#-future-work)

---

## 🔍 The Problem

Business owners are overwhelmed by customer reviews. A popular business can receive **thousands of reviews** on Yelp, each containing valuable insights about:
- What customers love
- Pain points and complaints
- Service quality issues
- Product/menu feedback

**Manually reading and categorizing these reviews is impractical.**

### Key Challenges

| Challenge | Description |
|-----------|-------------|
| 📊 Volume | 1,000+ reviews per popular business |
| 📝 Unstructured | Free-form natural language |
| 🎭 Mixed Sentiment | Single reviews contain both praise and complaints |
| 🌐 Noise | Foreign language, off-topic content |
| 🏷️ Interpretation | Raw data needs actionable labels |

---

## 💡 Solution Overview

This project implements an **end-to-end NLP pipeline** that automatically:

1. **Loads** reviews for any business using PyArrow pushdown predicates
2. **Cleans** text (normalization, lemmatization)
3. **Embeds** reviews into semantic vectors
4. **Clusters** reviews into distinct themes
5. **Labels** each theme with human-readable summaries

```
Raw Reviews → Preprocessing → Embeddings → Hierarchical → KMeans → LLM Labels
   N reviews      ✓ Clean       384-dim      Context     K themes   Actionable
                               vectors      filtering              insights
```

### What You Get

For any business, the pipeline generates:
- **Theme names** and categories
- **Summary** of customer feedback per theme
- **Keywords** and **noun phrases** extracted from the cluster
- **Representative reviews** closest to the cluster centroid

---

## 📈 Example Results

### Case Study: Datz Restaurant (Tampa, FL)

We ran the pipeline on a restaurant with 3,388 reviews to demonstrate its capabilities.

| Metric | Value |
|--------|-------|
| Reviews Analyzed | 3,388 |
| Themes Discovered | 10 |
| Processing Time | ~45 seconds |
| Main Context Coverage | 99.3% |

### Discovered Themes

| # | Theme | Reviews | Key Insight |
|---|-------|---------|-------------|
| 1 | Food Quality & Menu Variety | 953 | Diverse menu praised, but execution inconsistent |
| 2 | Service Experience | 574 | Trendy atmosphere, but slow service |
| 3 | Unique Dining Experience | 471 | Signature items are customer favorites |
| 4 | Innovative Menu | 457 | Creative dishes attract customers |
| 5 | Brunch Experience | 273 | Popular but has quality issues |

### Sample Output

```json
{
  "0": {
    "theme_name": "Vibrant Dining Experience at Datz",
    "theme_category": "Customer Experience",
    "summary": "Customers consistently praise the vibrant atmosphere and unique menu offerings...",
    "keywords": ["tampa", "restaurant", "love", "favorite", "amazing"],
    "noun_phrases": ["cheesy todd", "blue cheese drizzle", "brunch menu"],
    "representative_reviews": ["..."],
    "review_count": 471
  }
}
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-analysis-yelp.git
cd sentiment-analysis-yelp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Data Setup

Download the [Yelp Academic Dataset](https://business.yelp.com/data/resources/open-dataset/) and place files in `data/raw/`:

```
data/raw/
├── yelp_academic_dataset_business.json
└── yelp_reviews.parquet  # Convert from JSON for faster loading
```

---

## ⚡ Quick Start

### Option 1: Command Line Interface (Recommended)

```bash
# List top businesses by review count
python cli.py --list-businesses

# Run theme discovery by business ID
python cli.py -b <BUSINESS_ID>

# Run by business name (fuzzy search)
python cli.py -n "Restaurant Name" --fuzzy

# Custom output directory
python cli.py -b <BUSINESS_ID> -o ./my_results/
```

**Example:**
```bash
# Analyze Datz restaurant
python cli.py -b QHWYlmVbLC3K6eglWoHVvA

# Search by name
python cli.py -n "Acme Oyster House" --fuzzy
```

### Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/pipeline/theme_discovery_workflow.ipynb
```

---

## 📁 Project Structure

```
sentiment-analysis-yelp/
│
├── 📂 src/                          # Source code
│   ├── core/                        # Core algorithms
│   │   ├── business_context_clusterer.py
│   │   ├── clustering_evaluator.py
│   │   ├── theme_discovery_engine.py
│   │   └── theme_labeler.py
│   ├── data/                        # Data loading
│   │   └── my_reviews_loader.py
│   ├── embeddings/                  # Embedding models
│   │   ├── my_sentence_transformer.py
│   │   └── embedding_space_normalizer.py
│   └── preprocessing/               # Text preprocessing
│       ├── my_preprocessor.py
│       └── feature_engineering.py
│
├── 📂 notebooks/                    # Jupyter notebooks
│   └── pipeline/
│       └── theme_discovery_workflow.ipynb
│
├── 📂 reports/                      # Project deliverables
│   ├── project_report.ipynb
│   ├── model_metrics.md
│   └── figures/
│
├── 📂 results/                      # Pipeline outputs
│   └── {business_id}_themes.json
│
├── cli.py                           # Command-line interface
├── pipeline.py                      # Main pipeline orchestrator
└── requirements.txt
```

---

## 🔬 Methodology

### Two-Level Clustering Approach

```
                    ┌─────────────────────┐
                    │   All Reviews       │
                    │   (N reviews)       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Hierarchical       │  ← Separates noise
                    │  Clustering         │    (foreign language,
                    └──────────┬──────────┘     off-topic)
                               │
              ┌────────────────┼────────────────┐
              │                │                │
      ┌───────▼───────┐ ┌──────▼──────┐ ┌──────▼──────┐
      │ Main Context  │ │   Noise     │ │   Other     │
      │ (majority)    │ │             │ │   contexts  │
      └───────┬───────┘ └─────────────┘ └─────────────┘
              │
    ┌─────────▼─────────┐
    │  KMeans (k=auto)  │  ← Discovers themes
    └─────────┬─────────┘    (elbow method)
              │
    ┌─────────▼─────────┐
    │  LLM Labeling     │  ← Human-readable labels
    └───────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Loading | PyArrow | Efficient pushdown predicate filtering |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim semantic vectors |
| Context Clustering | Agglomerative (Ward) | Noise separation |
| Theme Clustering | KMeans + Elbow | Theme discovery |
| Labeling | GPT-4o-mini | Interpretable summaries |

---

## 📊 Model Performance

### Typical Results

| Metric | Typical Range |
|--------|---------------|
| Main Context Coverage | 95-99% |
| Themes Discovered | 6-14 |
| Min Theme Size | 50+ reviews |
| Processing Time | 30-60 seconds |

### Processing Time Breakdown

| Stage | Time |
|-------|------|
| Data Loading | <1s (PyArrow) |
| Preprocessing | ~5s |
| Embedding | ~15s |
| Clustering | ~20s |
| LLM Labeling | ~5s |
| **Total** | **~45s** |

### Comparison with Alternatives

| Method | Interpretability | Actionability | Automation |
|--------|-----------------|---------------|------------|
| **Our Pipeline** | ⭐⭐⭐ High | ⭐⭐⭐ High | ⭐⭐⭐ Full |
| LDA Topic Model | ⭐⭐ Medium | ⭐ Low | ⭐⭐⭐ Full |
| Manual Coding | ⭐⭐⭐ High | ⭐⭐⭐ High | ⭐ None |

---

## 💼 How to Use the Results

The pipeline output provides actionable insights for any business:

### Interpreting Themes

Each theme includes:
- **theme_name**: Human-readable label
- **theme_category**: Broader category (e.g., "Food Quality", "Service")
- **summary**: Overview of customer feedback
- **keywords**: TF-IDF terms that distinguish this theme
- **noun_phrases**: Common multi-word expressions from reviews
- **representative_reviews**: 3 reviews closest to cluster centroid
- **review_count**: Number of reviews in this theme

### Turning Insights into Action

1. **Identify high-priority issues**: Look for negative patterns in summaries and representative reviews
2. **Double down on strengths**: Feature items mentioned in positive noun phrases
3. **Track over time**: Re-run periodically to measure theme evolution

---

## 🔮 Future Work

- [ ] **Temporal Analysis**: Track theme evolution over time
- [ ] **Sentiment Scoring**: Correlate themes with star ratings
- [ ] **Comparative Analysis**: Compare across similar businesses
- [ ] **Real-time Processing**: Stream new reviews as they arrive
- [ ] **Web Dashboard**: Interactive UI for business owners
- [ ] **Multi-language Support**: Analyze non-English reviews

---

## � Further Research

There were a few observations and challenges that segue into phase 2 of this project.

### Observations

1. **Review text as a "unit of analysis" creates mixed signals**
   - A single review often discusses multiple topics (food, service, atmosphere, parking)
   - This leads to overlapping themes and diluted cluster coherence
   - *Phase 2 consideration*: Sentence-level or aspect-level segmentation before clustering

2. **No explicit sentiment classification on themes**
   - Themes lack an overall sentiment label (positive/negative/mixed)
   - This makes it harder to quickly identify problem areas vs. strengths
   - *Phase 2 consideration*: Add sentiment scoring per theme based on star rating distribution or lexicon analysis

3. **Theme names can be too generic**
   - Multiple themes received similar names like "Food Quality and Service Experience"
   - The LLM sometimes fails to capture what distinguishes one cluster from another
   - *Phase 2 consideration*: Include more discriminative keywords in the LLM prompt or use contrastive labeling

4. **Representative reviews don't always indicate the theme clearly**
   - Reviews closest to cluster centroids may be "average" rather than "characteristic"
   - *Phase 2 consideration*: Select reviews that maximize keyword overlap or use MMR (Maximal Marginal Relevance) for diversity

### Challenges

| Challenge | Description | Potential Solution |
|-----------|-------------|-------------------|
| Cluster interpretability | High-dimensional embeddings are hard to validate | Use UMAP visualization and silhouette analysis |
| Optimal k selection | Elbow method is subjective | Explore silhouette scores or gap statistic |
| LLM consistency | Same cluster can get different labels on re-runs | Use lower temperature or structured prompts |
| Scalability | 10k+ reviews increase embedding and clustering time | Batch processing or sampling strategies |

---

## �📚 References

- [Sentence Transformers](https://www.sbert.net/)
- [Yelp Academic Dataset](https://business.yelp.com/data/resources/open-dataset/)
- [OpenAI API](https://platform.openai.com/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Safari Gisa**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

<p align="center">
  <i>If you found this project helpful, please consider giving it a ⭐!</i>
</p>
