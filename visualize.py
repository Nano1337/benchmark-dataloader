import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set plot style
plt.style.use('ggplot')
sns.set_context("talk")

def load_data(file_path):
    """Load and process the benchmark data"""
    df = pd.read_csv(file_path)
    # Ensure numeric data types
    for col in ['num_workers', 'prefetch_factor', 'batch_size', 'throughput', 'TTFB']:
        df[col] = pd.to_numeric(df[col])
    return df

def create_grouped_bar_chart(df, output_dir):
    """Create grouped bar chart showing throughput by dataloader, grouped by worker counts"""
    plt.figure(figsize=(14, 8))
    
    # Calculate average throughput across prefetch_factor and batch_size for each dataset and num_workers
    agg_df = df.groupby(['dataset', 'num_workers']).agg({'throughput': 'mean'}).reset_index()
    
    # Create bar plot
    ax = sns.barplot(x='num_workers', y='throughput', hue='dataset', data=agg_df)
    
    plt.title('Average Throughput by Dataloader and Number of Workers', fontsize=16)
    plt.xlabel('Number of Workers', fontsize=14)
    plt.ylabel('Throughput (samples/sec)', fontsize=14)
    plt.legend(title='Dataset', fontsize=12)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grouped_bar_chart.png'), dpi=300)
    plt.close()



def create_line_chart(df, output_dir):
    """Create line chart showing TTFB scaling with workers"""
    plt.figure(figsize=(12, 8))
    
    # Calculate average TTFB across prefetch_factor and batch_size
    agg_df = df.groupby(['dataset', 'num_workers']).agg({'TTFB': 'mean'}).reset_index()
    
    # Plot lines
    sns.lineplot(
        data=agg_df,
        x='num_workers',
        y='TTFB',
        hue='dataset',
        marker='o',
        linewidth=2.5,
        markersize=10
    )
    
    plt.title('Time To First Batch (TTFB) by Number of Workers', fontsize=16)
    plt.xlabel('Number of Workers', fontsize=14)
    plt.ylabel('Average TTFB (seconds)', fontsize=14)
    plt.xticks(df['num_workers'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Dataset', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttfb_line_chart.png'), dpi=300)
    plt.close()



def main():
    # Define input and output paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'results', 'streaming', 'sweep_results.csv')
    output_dir = os.path.join(current_dir, 'results', 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    
    # Create visualizations
    print("Creating grouped bar chart...")
    create_grouped_bar_chart(df, output_dir)
    
    print("Creating TTFB line chart...")
    create_line_chart(df, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()