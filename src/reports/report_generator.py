"""
Report Generation Module

Creates publication-ready figures for theme discovery analysis:
- Theme prevalence by evidence density
- Inter-theme semantic separation (cluster distinctiveness)
- Elbow curve for k-selection
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class ReportGenerator:
    """
    Generates figures and reports for theme discovery pipeline results.
    
    Figures are saved to: reports/figures/{business_id}/
    """

    def __init__(self, output_dir="reports/figures", business_id=None):
        if business_id:
            self.output_dir = os.path.join(output_dir, business_id)
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set consistent style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'sans-serif'

    # ------------------------------------------------------------------
    # FIGURE 1: Theme Prevalence
    # ------------------------------------------------------------------

    def create_theme_prevalence_figure(
        self,
        themes: dict,
        business_name: str = "Business",
        save_path: str = None,
    ) -> str:
        """
        Create horizontal bar chart showing theme prevalence by review count.
        
        Parameters:
        -----------
        themes : dict
            Theme results from pipeline (theme_id -> {theme_name, review_count, ...})
        business_name : str
            Name of business for title
        save_path : str, optional
            Custom save path (default: output_dir/theme_prevalence.png)
            
        Returns:
        --------
        str : Path to saved figure
        """
        # Extract theme data
        theme_data = []
        for tid, theme in themes.items():
            name = theme.get('theme_name', f'Theme {tid}')
            # Truncate long names
            display_name = name[:45] + '...' if len(name) > 45 else name
            theme_data.append({
                'theme_id': tid,
                'name': display_name,
                'full_name': name,
                'count': theme.get('review_count', 0)
            })
        
        # Sort by count (ascending for horizontal bar)
        theme_data = sorted(theme_data, key=lambda x: x['count'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(theme_data) * 0.5)))
        
        names = [t['name'] for t in theme_data]
        counts = [t['count'] for t in theme_data]
        
        # Color gradient based on count
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(theme_data)))
        
        bars = ax.barh(range(len(theme_data)), counts, color=colors, edgecolor='gray', linewidth=0.5)
        
        ax.set_yticks(range(len(theme_data)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Number of Reviews (Evidence Density)')
        ax.set_title(f'Theme Prevalence by Evidence Density\n{business_name}', 
                     fontsize=12, fontweight='bold')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(count + max(counts) * 0.02, i, f'{count}', 
                    va='center', fontsize=9, fontweight='bold')
        
        # Add total reviews annotation
        total = sum(counts)
        ax.text(0.98, 0.02, f'Total: {total:,} reviews', 
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, style='italic', color='gray')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'theme_prevalence.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Created: {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # FIGURE 2: Inter-Theme Semantic Separation
    # ------------------------------------------------------------------

    def create_inter_theme_separation_figure(
        self,
        similarity_matrix: list = None,
        themes: dict = None,
        save_path: str = None,
    ) -> str:
        """
        Create heatmap showing inter-theme semantic separation.
        
        Visualizes pairwise cosine similarity between cluster centroids
        to show how distinct the discovered themes are.
        
        Parameters:
        -----------
        similarity_matrix : list
            Pre-computed pairwise cosine similarity matrix (from pipeline output)
        themes : dict, optional
            Theme results for labels (if None, uses Theme 0, Theme 1, etc.)
        save_path : str, optional
            Custom save path
            
        Returns:
        --------
        str : Path to saved figure
        """
        if similarity_matrix is None:
            raise ValueError("similarity_matrix is required")
        
        sim_matrix = np.array(similarity_matrix)
        k = len(sim_matrix)
        
        # Get theme names
        theme_names = []
        for cid in range(k):
            if themes and str(cid) in themes:
                name = themes[str(cid)].get('theme_name', f'Theme {cid}')
            elif themes and cid in themes:
                name = themes[cid].get('theme_name', f'Theme {cid}')
            else:
                name = f'Theme {cid}'
            # Shorten for display
            name = name[:25] + '...' if len(name) > 25 else name
            theme_names.append(name)
        
        # Mean off-diagonal similarity (lower = more distinct)
        upper_tri = sim_matrix[np.triu_indices(k, k=1)]
        mean_similarity = upper_tri.mean()
        mean_separation = 1 - mean_similarity
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # LEFT: Similarity heatmap
        im1 = axes[0].imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0].set_xticks(range(k))
        axes[0].set_yticks(range(k))
        axes[0].set_xticklabels(theme_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticklabels(theme_names, fontsize=8)
        axes[0].set_title('Inter-Theme Cosine Similarity\n(Lower = More Distinct)', 
                          fontsize=11, fontweight='bold')
        
        # Add similarity values to cells
        for i in range(k):
            for j in range(k):
                color = 'white' if sim_matrix[i, j] > 0.7 else 'black'
                axes[0].text(j, i, f'{sim_matrix[i, j]:.2f}', 
                            ha='center', va='center', fontsize=7, color=color)
        
        plt.colorbar(im1, ax=axes[0], label='Cosine Similarity', shrink=0.8)
        
        # RIGHT: Separation bar chart (off-diagonal values)
        # Calculate mean separation per theme (how distinct from others)
        theme_separations = []
        for i in range(k):
            # Mean similarity to other themes
            other_sims = [sim_matrix[i, j] for j in range(k) if i != j]
            mean_sep = 1 - np.mean(other_sims)
            theme_separations.append(mean_sep)
        
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, k))
        sorted_idx = np.argsort(theme_separations)
        
        bars = axes[1].barh(
            range(k), 
            [theme_separations[i] for i in sorted_idx],
            color=[colors[i] for i in range(k)],
            edgecolor='gray',
            linewidth=0.5
        )
        
        axes[1].set_yticks(range(k))
        axes[1].set_yticklabels([theme_names[i] for i in sorted_idx], fontsize=8)
        axes[1].set_xlabel('Semantic Separation (1 - avg similarity to other themes)')
        axes[1].set_xlim(0, 1)
        axes[1].set_title('Theme Distinctiveness\n(Higher = More Unique)', 
                          fontsize=11, fontweight='bold')
        
        # Add separation value labels
        for i, idx in enumerate(sorted_idx):
            axes[1].text(theme_separations[idx] + 0.02, i, 
                        f'{theme_separations[idx]:.2f}', 
                        va='center', fontsize=8)
        
        # Add mean separation line
        axes[1].axvline(mean_separation, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[1].text(mean_separation + 0.02, k - 0.5, 
                    f'Mean: {mean_separation:.2f}', 
                    fontsize=9, color='red', fontweight='bold')
        
        plt.suptitle(f'Inter-Theme Semantic Separation Analysis (k={k})', 
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'inter_theme_separation.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Created: {save_path}")
        print(f"  Mean inter-theme similarity: {mean_similarity:.3f}")
        print(f"  Mean separation score: {mean_separation:.3f}")
        
        return save_path

    # ------------------------------------------------------------------
    # FIGURE 3: Elbow Curve for K-Selection
    # ------------------------------------------------------------------

    def create_elbow_curve_figure(
        self,
        k_values: list,
        inter_centroid_cosine: list,
        selected_k: int,
        save_path: str = None,
    ) -> str:
        """
        Create elbow curve visualization for k-selection.
        
        Parameters:
        -----------
        k_values : list
            Range of k values tested
        inter_centroid_cosine : list
            Inter-centroid cosine similarity for each k
        selected_k : int
            The selected optimal k value
        save_path : str, optional
            Custom save path
            
        Returns:
        --------
        str : Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the curve
        ax.plot(k_values, inter_centroid_cosine, 'b-o', linewidth=2, markersize=8, label='Inter-centroid similarity')
        
        # Highlight selected k
        selected_idx = k_values.index(selected_k)
        selected_value = inter_centroid_cosine[selected_idx]
        
        ax.scatter([selected_k], [selected_value], 
                   s=200, c='red', marker='*', zorder=5, label=f'Selected k={selected_k}')
        
        # Add annotation
        ax.annotate(f'Elbow: k={selected_k}\nSimilarity: {selected_value:.3f}',
                    xy=(selected_k, selected_value),
                    xytext=(selected_k + 1.5, selected_value + 0.03),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    color='red')
        
        ax.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax.set_ylabel('Inter-Centroid Cosine Similarity', fontsize=11)
        ax.set_title('Elbow Method: Optimal Number of Themes\n(Lower similarity = More distinct clusters)', 
                     fontsize=12, fontweight='bold')
        
        ax.set_xticks(k_values)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add interpretation zone
        ax.fill_between(k_values, inter_centroid_cosine, alpha=0.1, color='blue')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'elbow_curve.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Created: {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # BATCH: Generate All Figures
    # ------------------------------------------------------------------

    def generate_all_figures(
        self,
        themes: dict,
        X_embeddings: np.ndarray,
        labels: np.ndarray,
        k_values: list = None,
        inter_centroid_scores: list = None,
        selected_k: int = None,
        business_name: str = "Business",
    ) -> dict:
        """
        Generate all standard figures for a theme discovery run.
        
        Parameters:
        -----------
        themes : dict
            Theme results from pipeline
        X_embeddings : np.ndarray
            Raw embeddings
        labels : np.ndarray
            Cluster assignments
        k_values : list, optional
            K values tested (for elbow curve)
        inter_centroid_scores : list, optional
            Scores for each k (for elbow curve)
        selected_k : int, optional
            Selected k value
        business_name : str
            Business name for titles
            
        Returns:
        --------
        dict : Paths to all generated figures
        """
        paths = {}
        
        # Figure 1: Theme prevalence
        paths['theme_prevalence'] = self.create_theme_prevalence_figure(
            themes=themes,
            business_name=business_name
        )
        
        # Figure 2: Inter-theme separation
        paths['inter_theme_separation'] = self.create_inter_theme_separation_figure(
            X_embeddings=X_embeddings,
            labels=labels,
            themes=themes
        )
        
        # Figure 3: Elbow curve (if data provided)
        if k_values and inter_centroid_scores and selected_k:
            paths['elbow_curve'] = self.create_elbow_curve_figure(
                k_values=k_values,
                inter_centroid_cosine=inter_centroid_scores,
                selected_k=selected_k
            )
        
        print(f"\n{'='*50}")
        print(f"Generated {len(paths)} figures in {self.output_dir}")
        print('='*50)
        
        return paths
