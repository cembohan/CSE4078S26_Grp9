import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def main():
    evals_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Pattern to extract model name and temperature
    pattern = re.compile(r'eval_(.*?)_10_predictions_.*_temp(\d{2})\.csv')
    
    data = []
    
    csv_files = glob.glob(os.path.join(evals_dir, '*.csv'))
    for file in csv_files:
        filename = os.path.basename(file)
        match = pattern.search(filename)
        if match:
            # Extract model name
            model_full = match.group(1)
            model_name = model_full.split('_')[-1]
            
            # Extract temperature
            temp_str = match.group(2)
            temp = f"0.{temp_str[1]}" if temp_str.startswith('0') else temp_str
            
            try:
                df = pd.read_csv(file)
                if 'final_score' in df.columns:
                    # Final score average over all 10 values
                    avg_score = df['final_score'].mean()
                    data.append({'Model': model_name, 'Temperature': temp, 'Score': avg_score})
            except Exception as e:
                print(f"Error reading {file}: {e}")
            
    if not data:
        print("No valid data found to plot!")
        return
        
    # Create DataFrame and pivot for the table
    df_results = pd.DataFrame(data)
    pivot_df = df_results.pivot(index='Model', columns='Temperature', values='Score')
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Use seaborn heatmap to create a styled table
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, 
                linewidths=1, linecolor='white', ax=ax, annot_kws={"size": 14})
    
    ax.set_title('Average Final Score by Model and Temperature\n(Averaged over 10 predictions)', 
                 fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    
    # Save the table as PNG
    output_path = os.path.join(evals_dir, 'score_table.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully created and saved the table to: {output_path}")

if __name__ == "__main__":
    main()
