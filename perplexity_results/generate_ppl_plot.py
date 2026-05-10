import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Directory containing perplexity results
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_files = glob.glob(os.path.join(results_dir, '*_perplexity_*.csv'))
    
    data = []
    
    for file in csv_files:
        filename = os.path.basename(file)
        # Extract model name from filename
        # Format: {provider}_{model_name}_perplexity_{date}_{time}.csv
        parts = filename.split('_perplexity_')
        if len(parts) == 2:
            model_full = parts[0]
            # Get the actual model name
            model_name = model_full.split('_')[-1]
            
            try:
                df = pd.read_csv(file)
                if 'perplexity' in df.columns:
                    # Filter out NaN or infinite values just in case
                    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['perplexity'])
                    df['Model'] = model_name
                    data.append(df[['Model', 'perplexity']])
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    if not data:
        print("No valid perplexity data found to plot!")
        return
        
    combined_df = pd.concat(data, ignore_index=True)
    
    # Calculate average PPL for barplot
    avg_df = combined_df.groupby('Model')['perplexity'].mean().reset_index()
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with two subplots: one for average, one for distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Bar chart of average perplexity
    sns.barplot(data=avg_df, x='Model', y='perplexity', ax=axes[0], palette="Blues_d")
    axes[0].set_title('Average Perplexity by Model', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Average Perplexity', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
    
    # Annotate the bars
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 8), textcoords='offset points', 
                         fontsize=12, fontweight='bold')
                         
    # Plot 2: Box plot of perplexity distribution
    # We use showfliers=False because perplexity often has extreme outliers that squash the boxplot
    sns.boxplot(data=combined_df, x='Model', y='perplexity', ax=axes[1], palette="Blues_d", showfliers=False)
    axes[1].set_title('Perplexity Distribution (Outliers Hidden)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(results_dir, 'ppl_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    main()
