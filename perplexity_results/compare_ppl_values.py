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
    
    print("="*50)
    print("Processing Perplexity Results")
    print("="*50)
    
    for file in csv_files:
        filename = os.path.basename(file)
        # Extract model name from filename
        parts = filename.split('_perplexity_')
        if len(parts) == 2:
            model_full = parts[0]
            model_name = model_full.split('_')[-1]
            
            try:
                df = pd.read_csv(file)
                total_rows = len(df)
                
                if 'perplexity' in df.columns:
                    # Filter out NaN or infinite values
                    df_valid = df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['perplexity'])
                    valid_rows = len(df_valid)
                    
                    df_valid = df_valid.copy()
                    df_valid['Model'] = model_name
                    data.append(df_valid[['Model', 'perplexity']])
                    
                    print(f"Model: {model_name}")
                    print(f"  - Total rows: {total_rows}")
                    print(f"  - Valid PPL rows: {valid_rows}")
                    print(f"  - Invalid/Missing rows: {total_rows - valid_rows}")
                    print(f"  - Average PPL: {df_valid['perplexity'].mean():.2f}")
                    print(f"  - Median PPL:  {df_valid['perplexity'].median():.2f}")
                    print("-" * 50)
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
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Bar chart of average perplexity
    sns.barplot(data=avg_df, x='Model', y='perplexity', ax=axes[0], hue='Model', palette="Blues_d", legend=False)
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
    sns.boxplot(data=combined_df, x='Model', y='perplexity', ax=axes[1], hue='Model', palette="Blues_d", showfliers=False, legend=False)
    axes[1].set_title('Perplexity Distribution (Outliers Hidden)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(results_dir, 'ppl_comparison_median.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated plot at: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
