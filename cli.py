#!/usr/bin/env python
"""
CLI for running the theme discovery pipeline on Yelp business reviews.

Usage:
    python cli.py --business-id QHWYlmVbLC3K6eglWoHVvA
    python cli.py -b QHWYlmVbLC3K6eglWoHVvA --output results/
    python cli.py --business-name "Datz" --fuzzy
"""

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

from pipeline import ReviewThemePipeline
from src.data.my_reviews_loader import BusinessReviewLoader
from src.preprocessing.my_preprocessor import MyPreProcessor
from src.embeddings.my_sentence_transformer import MySentenceTransformer
from src.core.business_context_clusterer import BusinessContextClusterer
from src.core.theme_discovery_engine import ThemeDiscoveryEngine
from src.core.theme_labeler import ThemeLabeler

# Load environment variables
load_dotenv()


def setup_paths(project_root=None):
    """Setup data file paths."""
    if project_root is None:
        project_root = Path(__file__).parent
    
    return {
        "business": project_root / "data/raw/yelp_academic_dataset_business.json",
        "reviews": project_root / "data/raw/yelp_reviews.parquet",
    }


def initialize_pipeline():
    """Initialize and return the pipeline."""
    paths = setup_paths()
    
    print("Initializing components...")
    loader = BusinessReviewLoader(str(paths["business"]), str(paths["reviews"]))
    preprocessor = MyPreProcessor()
    embedder = MySentenceTransformer(model_name="all-MiniLM-L6-v2")
    clusterer = BusinessContextClusterer()
    theme_discovery = ThemeDiscoveryEngine()
    
    # Initialize OpenAI client
    client = OpenAI()
    theme_labeler = ThemeLabeler(client=client)
    
    pipeline = ReviewThemePipeline(
        review_loader=loader,
        preprocessor=preprocessor,
        embedder=embedder,
        context_clusterer=clusterer,
        theme_discovery=theme_discovery,
        theme_labeler=theme_labeler,
    )
    
    return pipeline, loader


def run_by_business_id(business_id, output_dir=None):
    """Run pipeline for a specific business_id."""
    print(f"\nRunning theme discovery for business_id: {business_id}")
    print("=" * 70)
    
    pipeline, _ = initialize_pipeline()
    results = pipeline.run(business_id=business_id)
    
    if not results:
        print(f"❌ No results found. Business may not exist or has no reviews.")
        return None
    
    print(f"\n✓ Theme discovery complete!")
    print(f"  Found {len(results.get('themes', results))} themes")
    
    # Save results under business folder
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(output_dir)
    
    # Create business-specific folder
    business_output_dir = output_dir / business_id
    business_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = business_output_dir / "themes.json"
    
    # Convert numpy types to native Python types (keys and values)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, (np.integer,)) else k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results = convert_numpy(results)
    
    # New format has metadata, inter_theme_similarity, themes structure
    # Sort themes by ID within the themes dict
    if 'themes' in results:
        sorted_themes = dict(sorted(results['themes'].items(), key=lambda x: int(x[0])))
        results['themes'] = sorted_themes
    else:
        # Legacy format - themes at root level
        results = dict(sorted(results.items(), key=lambda x: int(x[0])))
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved to: {output_path}")
    
    return results


def run_by_business_name(business_name, fuzzy=False, output_dir=None):
    """Run pipeline for a business by name."""
    print(f"\nLooking up business: '{business_name}' (fuzzy={fuzzy})...")
    
    _, loader = initialize_pipeline()
    
    try:
        reviews_df = loader.get_reviews_by_business_name(
            business_name,
            exact=not fuzzy,
            require_open=True
        )
        
        # Get business_id from the loader's business_df
        business_id = loader.business_df[
            loader.business_df['name'].str.lower().str.strip() == business_name.lower().strip()
        ]['business_id'].iloc[0]
        
        print(f"Found business_id: {business_id}")
        return run_by_business_id(business_id, output_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def list_top_businesses(limit=20):
    """List top businesses by review count."""
    paths = setup_paths()
    loader = BusinessReviewLoader(str(paths["business"]), str(paths["reviews"]))
    
    top_businesses = loader.business_df.nlargest(limit, 'review_count')[
        ['business_id', 'name', 'city', 'state', 'review_count', 'stars']
    ]
    
    print("\nTop Businesses by Review Count:")
    print("=" * 70)
    print(top_businesses.to_string(index=False))
    return top_businesses


def main():
    parser = argparse.ArgumentParser(
        description="Theme Discovery Pipeline for Yelp Reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run by business ID
  python cli.py -b QHWYlmVbLC3K6eglWoHVvA
  
  # Run by business name (exact match)
  python cli.py -n "Datz"
  
  # Run by business name (fuzzy match)
  python cli.py -n "Datz" --fuzzy
  
  # List top 20 businesses
  python cli.py --list-businesses
  
  # Specify output directory
  python cli.py -b QHWYlmVbLC3K6eglWoHVvA -o ./my_results/
        """
    )
    
    parser.add_argument(
        '-b', '--business-id',
        type=str,
        help='Yelp business ID'
    )
    
    parser.add_argument(
        '-n', '--business-name',
        type=str,
        help='Business name (requires exact or fuzzy match)'
    )
    
    parser.add_argument(
        '--fuzzy',
        action='store_true',
        help='Use fuzzy matching for business name (substring match)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory for results (default: ./results/)'
    )
    
    parser.add_argument(
        '--list-businesses',
        action='store_true',
        help='List top 20 businesses by review count'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top businesses to show (with --list-businesses)'
    )
    
    args = parser.parse_args()
    
    # Handle list businesses
    if args.list_businesses:
        list_top_businesses(args.top)
        return 0
    
    # Handle business ID
    if args.business_id:
        results = run_by_business_id(args.business_id, args.output)
        if results:
            return 0
        else:
            return 1
    
    # Handle business name
    if args.business_name:
        results = run_by_business_name(args.business_name, args.fuzzy, args.output)
        if results:
            return 0
        else:
            return 1
    
    # No valid option provided
    parser.print_help()
    print("\n❌ Error: Please provide either --business-id or --business-name")
    return 1


if __name__ == '__main__':
    sys.exit(main())
