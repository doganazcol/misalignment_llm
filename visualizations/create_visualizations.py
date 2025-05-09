import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('filtered_data.csv')
plt.style.use('seaborn')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. comparison of human and GPT judgments
comparison_df = pd.DataFrame({
    'Judge': ['Human', 'GPT'],
    'Count': [df['human_winner'].value_counts().get('A', 0) + df['human_winner'].value_counts().get('B', 0),
              df['gpt_winner'].value_counts().get('A', 0) + df['gpt_winner'].value_counts().get('B', 0)]
})

sns.barplot(data=comparison_df, x='Judge', y='Count', ax=ax1, palette='Set2')
ax1.set_title('Comparison of Human vs GPT Judgments')
ax1.set_ylabel('Number of Judgments')

# 2. distribution of categories (open-ended vs close-ended)
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax2, palette='Set3')
ax2.set_title('Distribution of Question Categories')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('judgement_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualizations have been saved as 'judgement_analysis.png'")

print("\nSummary Statistics:")
print("\nHuman Judgments Distribution:")
print(df['human_winner'].value_counts())
print("\nGPT Judgments Distribution:")
print(df['gpt_winner'].value_counts())
print("\nCategory Distribution:")
print(df['category'].value_counts()) 