"""Check if top themes have reviews in each year."""
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BUSINESS_ID = 'QHWYlmVbLC3K6eglWoHVvA'

# Load from saved results
results_path = PROJECT_ROOT / 'results' / BUSINESS_ID / 'themes.json'
with open(results_path) as f:
    results = json.load(f)

# Build dataframe from review_assignments
main_df = pd.DataFrame(results['review_assignments'])
main_df['date'] = pd.to_datetime(main_df['date'])
main_df['year'] = main_df['date'].dt.year

# Get theme names from results
theme_names = {int(k): v['theme_name'] for k, v in results['themes'].items()}

# Top 3 themes
top_3 = main_df['theme_id'].value_counts().head(3).index.tolist()

print('Reviews per Year per Theme (Top 3):')
print('='*70)
pivot = main_df[main_df['theme_id'].isin(top_3)].groupby(['year', 'theme_id']).size().unstack(fill_value=0)
pivot.columns = [theme_names[c] for c in pivot.columns]
print(pivot.to_string())
print()
print('Years with 0 reviews per theme:')
for col in pivot.columns:
    zeros = pivot[pivot[col] == 0].index.tolist()
    if zeros:
        print(f'  {col}: {zeros}')
    else:
        print(f'  {col}: None (present in all years)')
