import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up the aesthetic style of the plots
sns.set_theme(style="whitegrid", context="talk")

# Load data
base_dir = r"d:\Desktop\NLP4078\project\CSE4078S26_Grp9\evaluations"
qwen_file = os.path.join(base_dir, "eval_Qwen_Qwen3.5-2B_100_predictions_20260510_115614_temp02.csv")
llama_file = os.path.join(base_dir, "eval_meta-llama_Llama-3.2-1B-Instruct_100_predictions_20260510_120307_temp02.csv")

df_qwen = pd.read_csv(qwen_file)
df_llama = pd.read_csv(llama_file)

# The metrics to plot
metrics = ['correctness', 'safety', 'completeness', 'final_score']

# Calculate means
qwen_means = df_qwen[metrics].mean()
llama_means = df_llama[metrics].mean()

# Create a DataFrame for Seaborn
data = []
for metric in metrics:
    data.append({'Model': 'Qwen3.5-2B', 'Metric': metric.capitalize().replace('_', ' '), 'Mean Score': qwen_means[metric]})
    data.append({'Model': 'Llama-3.2-1B-Instruct', 'Metric': metric.capitalize().replace('_', ' '), 'Mean Score': llama_means[metric]})

df_plot = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(12, 7))

# Define colors for the models to make it look premium
colors = ["#4361EE", "#F72585"] # vibrant blue and pink

ax = sns.barplot(
    data=df_plot, 
    x='Metric', 
    y='Mean Score', 
    hue='Model', 
    palette=colors,
    edgecolor="black",
    linewidth=1.5,
    alpha=0.9
)

# Customize the plot
plt.title("Evaluation Scores Comparison (100 Predictions)", fontsize=20, weight='bold', pad=20)
plt.ylabel("Mean Score", fontsize=16, weight='bold')
plt.xlabel("Evaluation Metric", fontsize=16, weight='bold')

# Ensure y-axis limit accommodates the max value and text above it
plt.ylim(0, max(df_plot['Mean Score']) * 1.2)

# Add values on top of bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points',
                fontsize=13,
                weight='bold')

plt.legend(title="Model", title_fontsize='14', fontsize='13', loc='upper right')
sns.despine()

# Adjust layout and save
plt.tight_layout()
output_path = os.path.join(base_dir, "score_comparison_100_preds.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {output_path}")
