# Project Structure & Security Guide

## Project Organization

This project follows a standardized structure for maintainability and scalability:

```
sentiment-analysis-yelp-rebuild/
├── README.md                      # Project documentation
├── pipeline.py                    # Main orchestration script
├── .env                           # Local secrets (DO NOT COMMIT)
├── .env.example                   # Template for .env (committed)
├── .gitignore                     # Git ignore rules
│
├── src/                           # Production code
│   ├── core/                      # Core algorithms
│   │   ├── business_context_clusterer.py
│   │   ├── clustering_evaluator.py
│   │   ├── theme_discovery_engine.py
│   │   └── theme_labeler.py
│   ├── preprocessing/             # NLP & data preprocessing
│   │   ├── corpus_denoiser.py
│   │   ├── feature_engineering.py
│   │   ├── my_preprocessor.py
│   │   ├── nlp_lemmatizer.py
│   │   ├── nlp_preprocessor.py
│   │   └── sentence_unitizer.py
│   ├── embeddings/                # Embedding models & tools
│   │   ├── embedding_space_normalizer.py
│   │   ├── my_llm_embedder.py
│   │   ├── my_sentence_transformer.py
│   │   ├── my_tfidf.py
│   │   └── open_ai_review_embeddings.py
│   └── data/                      # Data loading utilities
│       └── my_reviews_loader.py
│
├── notebooks/                     # Jupyter notebooks (analysis & experiments)
│   ├── clustering/
│   │   └── theme_discovery_workflow.ipynb
│   ├── data_cleaning/
│   │   ├── businesses_cleaning.ipynb
│   │   ├── checkins_cleaning.ipynb
│   │   └── users_cleaning.ipynb
│   ├── distributions/
│   │   └── business_distribution.ipynb
│   └── embedders/
│       ├── embedding_model_performance_cosine_similarity.ipynb
│       ├── reviews_embeddings.ipynb
│       ├── test_my_sentence_transformer.ipynb
│       ├── unified_embedding_evaluation.ipynb
│       └── unified_embedding_generation.ipynb
│
├── tests/                         # Test files
│   └── notebooks/
│       ├── test_lemmatizer.ipynb
│       └── test_nlp_preprocessor.ipynb
│
├── config/                        # Configuration files
│   └── (add config files here as needed)
│
└── data/                          # Data storage
    ├── raw/                       # Original raw data
    │   ├── yelp_academic_dataset_*.json
    │   └── yelp_reviews.parquet
    ├── embeddings/                # Pre-computed embeddings
    │   └── *.npy files
    └── processed/                 # Cleaned/processed data
```

---

## Secrets Management

### ❌ What NOT to do:
- Never commit `.env` with real API keys
- Never hardcode secrets in code
- Never share `.env` files publicly

### ✅ What TO do:

#### 1. **Create `.env.example` (COMMIT THIS)**
```bash
# .env.example - Template for environment variables
OPENAI_API_KEY="your-api-key-here"
DATABASE_URL="your-database-url-here"
```

#### 2. **Keep `.env` locally (NEVER COMMIT)**
```bash
# .env - Your actual secrets (in .gitignore)
OPENAI_API_KEY="sk-proj-your-real-key-here"
```

#### 3. **Update `.gitignore`**
```bash
# Environment - NEVER commit actual secrets
.env
.env.local
.env.*.local
```

#### 4. **Load in Python**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

---

## API Key Management

### Generate a New API Key
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (shown only once)
4. Paste into your local `.env` file

### Revoke Compromised Keys
1. Go to https://platform.openai.com/api-keys
2. Click the trash icon next to any exposed keys
3. Generate a new key immediately

### If Secrets Were Committed to Git

Remove from git history:
```bash
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch .env' \
  --prune-empty --tag-name-filter cat -- --all
```

---

## Applying This Structure to Other Projects

Use this command-line process as a template:

```bash
# 1. Create directories
mkdir -p src/{core,preprocessing,embeddings,data,utils}
mkdir -p notebooks
mkdir -p tests/{unit,integration,notebooks}
mkdir -p config
mkdir -p data/{raw,processed,embeddings}

# 2. Create .env.example
cat > .env.example << 'EOF'
OPENAI_API_KEY="your-key-here"
DATABASE_URL="your-url-here"
EOF

# 3. Update .gitignore
cat >> .gitignore << 'EOF'
# Environment - NEVER commit actual secrets
.env
.env.local
.env.*.local
EOF

# 4. Move files to appropriate folders
mv *.py src/  # Then organize by subfolder

# 5. Clean git history (if .env was exposed)
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch .env' \
  --prune-empty --tag-name-filter cat -- --all

# 6. Commit changes
git add -A
git commit -m "refactor: reorganize project structure and secure secrets"
```

---

## For Developers Using This Project

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Fill in your own API keys in `.env`

3. **Never commit `.env`** - it's in `.gitignore`

---

## Quick Reference

| File | Commit? | Purpose |
|------|---------|---------|
| `.env` | ❌ No | Your actual secrets |
| `.env.example` | ✅ Yes | Template for other developers |
| `.gitignore` | ✅ Yes | Tell git what to ignore |
| `pipeline.py` | ✅ Yes | Main script |
| `/src` | ✅ Yes | Production code |
| `/notebooks` | ✅ Yes | Jupyter notebooks |
| `/data/raw` | ❌ No | Raw data files |
| `/config` | ✅ Yes | Configuration templates |

---

## Last Updated
January 10, 2026
