import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('filtered_data.csv')

# Set the style for clean, modern plots
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.facecolor'] = 'white'

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Category Distribution (wandb style)
category_counts = df['category'].value_counts(normalize=True) * 100
colors = ['#2ecc71', '#3498db']
bars = ax1.bar(category_counts.index, category_counts.values, color=colors)
ax1.set_title('Distribution of Prompt Categories', pad=20, fontsize=14)
ax1.set_ylabel('Percentage (%)', fontsize=12)

# Add percentage labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

# 2. Agreement Analysis by Category
agreement_by_category = pd.crosstab(df['category'], 
                                  df['human_winner'] == df['gpt_winner'],
                                  normalize='index') * 100
agreement_by_category.columns = ['Disagree', 'Agree']
agreement_by_category.plot(kind='bar', ax=ax2, color=['#e74c3c', '#2ecc71'])
ax2.set_title('Agreement Rate by Category', pad=20, fontsize=14)
ax2.set_ylabel('Percentage (%)', fontsize=12)

# Add percentage labels
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.1f%%')

# Clean up the plots
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig('category_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Category analysis visualization saved as 'category_analysis.png'")

# Print detailed statistics
print("\nDetailed Statistics:")
print("\n1. Category Distribution:")
print(category_counts.round(1))

print("\n2. Agreement Rates by Category:")
print(agreement_by_category.round(1)) 