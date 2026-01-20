"""
Create time-series line plot showing theme prevalence over time.
Uses saved review_assignments from pipeline output - no re-clustering needed.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    business_id = "QHWYlmVbLC3K6eglWoHVvA"
    
    # -------------------------
    # 1. Load saved results
    # -------------------------
    results_path = PROJECT_ROOT / "results" / business_id / "themes.json"
    
    print(f"Loading results from: {results_path}")
    with open(results_path) as f:
        data = json.load(f)
    
    # Extract review assignments
    assignments = data["review_assignments"]
    themes = data["themes"]
    
    print(f"Loaded {len(assignments)} review assignments")
    print(f"Themes: {len(themes)}")
    
    # -------------------------
    # 2. Build DataFrame
    # -------------------------
    df = pd.DataFrame(assignments)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Theme name mapping (shorten for display)
    theme_names = {int(k): v["theme_name"] for k, v in themes.items()}
    theme_names_short = {k: v[:30] + "..." if len(v) > 30 else v for k, v in theme_names.items()}
    
    # -------------------------
    # 3. Calculate % per year per theme
    # -------------------------
    print("\nCalculating yearly theme prevalence...")
    
    # Selected themes with distinct stories:
    # 4 = Brunch Delight at Datz (273 reviews)
    # 8 = Fish Tacos and Unique Flavor Combinations (136 reviews)  
    # 6 = Service Inconsistency and Wait Times (181 reviews)
    selected_themes = [4, 8, 6]
    print(f"Selected themes: {[theme_names_short[t] for t in selected_themes]}")
    
    # Calculate raw counts per year per theme
    yearly_theme_counts = df.groupby(["year", "theme_id"]).size().unstack(fill_value=0)
    
    # Calculate percentages
    yearly_theme_pct = yearly_theme_counts.div(yearly_theme_counts.sum(axis=1), axis=0) * 100
    
    # Filter to years with enough data (at least 20 reviews)
    yearly_counts = df.groupby("year").size()
    valid_years = yearly_counts[yearly_counts >= 20].index
    yearly_theme_pct = yearly_theme_pct.loc[valid_years]
    
    print(f"Valid years: {list(valid_years)}")
    print("\nRaw counts per year (selected themes):")
    for tid in selected_themes:
        counts = yearly_theme_counts.loc[valid_years, tid].tolist()
        print(f"  {theme_names_short[tid]}: {counts}")
    
    # -------------------------
    # 4. Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Custom distinct colors for selected themes
    theme_colors = {
        4: "#E63946",  # Red - Brunch
        8: "#2A9D8F",  # Teal - Food Quality
        6: "#F4A261",  # Orange - Service Issues
    }
    
    for theme_id in selected_themes:
        if theme_id in yearly_theme_pct.columns:
            pcts = yearly_theme_pct[theme_id]
            counts = yearly_theme_counts.loc[valid_years, theme_id]
            
            ax.plot(
                pcts.index,
                pcts.values,
                marker="o",
                linewidth=2,
                markersize=6,
                label=theme_names_short[theme_id],
                color=theme_colors[theme_id],
            )
            
            # Annotate points with low counts (< 10 reviews)
            for year, count, pct in zip(pcts.index, counts.values, pcts.values):
                if count < 10:
                    ax.annotate(
                        f"n={count}",
                        (year, pct),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color="red",
                        fontweight="bold"
                    )
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("% of Reviews", fontsize=11)
    ax.set_title("Theme Prevalence Over Time (Top 3 Themes)\nDatz Restaurant\n(red labels show years with <10 reviews)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    # Set x-axis to show all years
    ax.set_xticks(yearly_theme_pct.index)
    ax.set_xticklabels(yearly_theme_pct.index, rotation=45)
    
    plt.tight_layout()
    
    # Save to reports/figures/{business_id}
    output_dir = PROJECT_ROOT / "reports" / "figures" / business_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "theme_prevalence_over_time.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Saved: {output_path}")


if __name__ == "__main__":
    main()
