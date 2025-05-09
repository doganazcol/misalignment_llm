import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Read the data
df = pd.read_csv('filtered_data.csv')

# Initialize wandb
wandb.init(project="chatbot-arena", name="judge-analysis")

# 1. Category Distribution Plot
plt.figure(figsize=(12, 6))
category_counts = df['category'].value_counts(normalize=True) * 100

# Create horizontal bar plot
fig1, ax1 = plt.subplots()
colors = ['#E9967A' if cat == 'Open-ended' else '#64B5F6' for cat in category_counts.index]
bars = ax1.barh(
    y=category_counts.index,
    width=category_counts.values,
    color=colors
)

ax1.set_xlabel('Percentage')
ax1.set_ylabel('Category')
ax1.set_title('Distribution of Prompt Categories')
for bar in bars:
    width = bar.get_width()
    ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{width:.1f}%', va='center')
plt.tight_layout()
plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
wandb.log({"category_distribution": wandb.Image('category_distribution.png')})

# 2. Agreement Analysis Plot
fig2, ax2 = plt.subplots(figsize=(12, 6))
agreement_data = []
for category in df['category'].unique():
    category_df = df[df['category'] == category]
    agree = (category_df['human_winner'] == category_df['gpt_winner']).mean() * 100
    disagree = 100 - agree
    agreement_data.append([category, agree, disagree])

agreement_df = pd.DataFrame(agreement_data, columns=['Category', 'Agree', 'Disagree'])
agreement_df.plot(x='Category', y=['Agree', 'Disagree'], kind='bar', ax=ax2)
ax2.set_title('Agreement Rates by Category')
ax2.set_ylabel('Percentage')
plt.tight_layout()
plt.savefig('agreement_rates.png', dpi=300, bbox_inches='tight')
wandb.log({"agreement_rates": wandb.Image('agreement_rates.png')})

# 3. Winner Distribution by Category
fig3, ax3 = plt.subplots(figsize=(12, 6))
winner_dist = pd.crosstab(df['category'], df['human_winner'], normalize='index') * 100
winner_dist.plot(kind='bar', stacked=True, ax=ax3)
ax3.set_title('Winner Distribution by Category')
ax3.set_ylabel('Percentage')
plt.legend(title='Winner', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('winner_distribution.png', dpi=300, bbox_inches='tight')
wandb.log({"winner_distribution": wandb.Image('winner_distribution.png')})

# 4. Decision Matrix
decision_matrix = pd.crosstab(df['human_winner'], df['gpt_winner'])
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(decision_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
ax4.set_title('Decision Matrix: Human vs GPT')
ax4.set_xlabel('GPT Decision')
ax4.set_ylabel('Human Decision')
plt.tight_layout()
plt.savefig('decision_matrix.png', dpi=300, bbox_inches='tight')
wandb.log({"decision_matrix": wandb.Image('decision_matrix.png')})

# 5. Log detailed metrics
metrics = {
    "open_ended_percentage": category_counts["Open-ended"],
    "closed_ended_percentage": category_counts["Closed-ended"],
    "overall_agreement_rate": (df['human_winner'] == df['gpt_winner']).mean() * 100
}

# Add category-specific metrics
for category in df['category'].unique():
    category_df = df[df['category'] == category]
    metrics[f"{category.lower().replace('-', '_')}_agreement"] = (
        (category_df['human_winner'] == category_df['gpt_winner']).mean() * 100
    )

wandb.log(metrics)

# Print summary statistics
print("\nSummary Statistics:")
print(f"\nCategory Distribution:\n{category_counts}")
print("\nAgreement Rates by Category:")
for category, agree, _ in agreement_data:
    print(f"{category}: {agree:.1f}% agreement")
print(f"\nOverall Agreement Rate: {metrics['overall_agreement_rate']:.1f}%")
print("\nDecision Matrix:")
print(decision_matrix)

wandb.finish() 