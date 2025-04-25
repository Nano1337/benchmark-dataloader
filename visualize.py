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

def create_batch_size_bar_grid(df, output_dir):
    """Create bar charts showing throughput by batch size for specific worker counts (4,8,16)
    in a 2x2 grid layout with the legend in the bottom-right space"""
    # Filter for only the requested worker counts
    target_workers = [4, 8, 16]
    filtered_df = df[df['num_workers'].isin(target_workers)]
    
    # Calculate average throughput across prefetch_factor for each dataset, batch_size, and filtered num_workers
    agg_df = filtered_df.groupby(['dataset', 'batch_size', 'num_workers']).agg({'throughput': 'mean'}).reset_index()
    
    # Set up a figure with a 2x2 grid layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Get the color palette used in the grouped_bar_chart for consistency
    palette = sns.color_palette()
    
    # Create a bar plot for each worker count in the first 3 positions
    for i, workers in enumerate(target_workers):
        # Filter data for this worker count
        worker_df = agg_df[agg_df['num_workers'] == workers]
        
        # Create bar plot on the appropriate subplot
        ax = axes[i]
        bars = sns.barplot(x='batch_size', y='throughput', hue='dataset', data=worker_df, ax=ax, palette=palette)
        
        # Set titles and labels
        ax.set_title(f'Workers: {workers}', fontsize=14)
        ax.set_xlabel('Batch Size', fontsize=12)
        
        # Only add y-label to the first and third subplot (left column)
        if i % 2 == 0:
            ax.set_ylabel('Throughput (samples/sec)', fontsize=12)
        else:
            ax.set_ylabel('')
        
        # Add value labels on top of bars - with special formatting for litdata
        for c_idx, container in enumerate(ax.containers):
            # Get the labels
            labels = [f"{v:.0f}" for v in container.datavalues]
            
            # Check if this container is for litdata (assuming it's one of the hue categories)
            dataset_labels = worker_df['dataset'].unique()
            is_litdata_container = False
            
            # Find which container corresponds to litdata
            if c_idx < len(dataset_labels) and 'litdata' in dataset_labels[c_idx].lower():
                is_litdata_container = True
            
            # Apply fontweight='bold' for litdata results
            if is_litdata_container:
                ax.bar_label(container, labels=labels, fontsize=10, fontweight='bold')
            else:
                ax.bar_label(container, labels=labels, fontsize=9)
        
        # Remove legends from all subplots
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Create a dedicated legend in the 4th subplot position
    ax_legend = axes[3]
    ax_legend.axis('off')  # Turn off axis
    
    # Get handles and labels from one of the other subplots
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Create a larger legend with increased font size
    legend = ax_legend.legend(handles, labels, title='Dataset', 
                             fontsize=14, title_fontsize=16,
                             loc='center', frameon=True, ncol=1, 
                             bbox_to_anchor=(0.5, 0.5),
                             markerscale=1.5)  # Make the legend markers larger
    
    # Make the legend box with a more visible border
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('gray')
    
    # Set a common title for all subplots
    fig.suptitle('Throughput by Batch Size and Number of Workers', fontsize=18, y=0.98)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_size_worker_grid.png'), dpi=300, bbox_inches='tight')
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
    print("Creating batch size bar grid for workers=4,8,16...")
    create_batch_size_bar_grid(df, output_dir)
    
    print("Creating TTFB line chart...")
    create_line_chart(df, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()