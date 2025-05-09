import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

df = pd.read_csv('filtered_data.csv')

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# 1. agreement analysis
ax1 = fig.add_subplot(gs[0, 0])
df['agreement'] = (df['human_winner'] == df['gpt_winner']).map({True: 'Agree', False: 'Disagree'})
agreement_counts = df['agreement'].value_counts()
colors = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax1.pie(agreement_counts, labels=agreement_counts.index, 
                                  autopct='%1.1f%%', colors=colors,
                                  textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_color('white')
ax1.set_title('Agreement between Human and GPT Judges', pad=20, fontsize=14)

# 2. performance by category
ax2 = fig.add_subplot(gs[0, 1])
category_agreement = pd.crosstab(df['category'], df['agreement'], normalize='index')
category_agreement.plot(kind='bar', ax=ax2, color=['#e74c3c', '#2ecc71'])
ax2.set_title('Agreement Rate by Category', fontsize=14)
ax2.set_xlabel('Category', fontsize=12)
ax2.set_ylabel('Proportion', fontsize=12)
ax2.legend(title='Judge Agreement', fontsize=10)
ax2.tick_params(axis='both', labelsize=10)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# 3. confusion Matrix
ax3 = fig.add_subplot(gs[1, 0])
conf_matrix = pd.crosstab(df['human_winner'], df['gpt_winner'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=conf_matrix.columns,
            yticklabels=conf_matrix.index,
            ax=ax3, annot_kws={'size': 12})
ax3.set_title('Confusion Matrix: Human vs GPT Decisions', fontsize=14)
ax3.set_xlabel('GPT Decision', fontsize=12)
ax3.set_ylabel('Human Decision', fontsize=12)
ax3.tick_params(axis='both', labelsize=10)

# 4. Stacked Percentage Distribution with improved colors
ax4 = fig.add_subplot(gs[1, 1])
winner_dist = pd.crosstab(df['category'], df['human_winner'], normalize='index') * 100
winner_dist.plot(kind='bar', stacked=True, ax=ax4, 
                color=['#3498db', '#e67e22'])
ax4.set_title('Winner Distribution by Category', fontsize=14)
ax4.set_xlabel('Category', fontsize=12)
ax4.set_ylabel('Percentage', fontsize=12)
ax4.legend(title='Winner', fontsize=10)
ax4.tick_params(axis='both', labelsize=10)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

# adjust layout and save
plt.tight_layout()
plt.savefig('paper_visualizations.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Advanced visualizations have been saved as 'paper_visualizations.png'")

print("\nStatistical Summary for Paper:")
print("\n1. Overall Agreement Rate:")
print(df['agreement'].value_counts(normalize=True).round(3) * 100)

print("\n2. Agreement Rate by Category:")
print(pd.crosstab(df['category'], df['agreement'], normalize='index').round(3) * 100)

print("\n3. Winner Distribution by Category:")
print(pd.crosstab(df['category'], df['human_winner'], normalize='index').round(3) * 100)

# chi-square test for independence
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(df['category'], df['human_winner'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print("\n4. Chi-square Test Results:")
print(f"Chi-square statistic: {chi2:.2f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}") 