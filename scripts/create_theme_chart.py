"""
Create bar chart showing % of corpus per theme.
Uses saved results from pipeline output.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    business_id = "QHWYlmVbLC3K6eglWoHVvA"
    
    # Load saved results
    results_path = PROJECT_ROOT / "results" / business_id / "themes.json"
    with open(results_path) as f:
        data = json.load(f)
    
    themes_data = data["themes"]
    total_reviews = data["metadata"]["main_context_reviews"]
    
    # Build theme list with percentages
    themes = []
    for tid, theme in themes_data.items():
        name = theme["theme_name"]
        count = theme["review_count"]
        pct = round(count / total_reviews * 100, 1)
        # Shorten name for display
        display_name = name[:35] + "..." if len(name) > 35 else name
        themes.append((display_name, pct))
    
    # Sort by percentage descending, then reverse for plot
    themes = sorted(themes, key=lambda x: x[1], reverse=True)
    themes = themes[::-1]  # smallest at bottom
    
    names = [t[0] for t in themes]
    pcts = [t[1] for t in themes]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(themes)))
    
    y_pos = np.arange(len(themes))
    bars = ax.barh(y_pos, pcts, color=colors, edgecolor="gray", linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Percentage of Corpus (%)", fontsize=11)
    ax.set_title(f"Theme Distribution: % of Reviews per Theme\n(Datz Restaurant, n={total_reviews:,})", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 35)
    
    # Add percentage labels
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct}%", va="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    
    # Save to reports/figures/{business_id}
    output_dir = PROJECT_ROOT / "reports" / "figures" / business_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "theme_corpus_percentage.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
