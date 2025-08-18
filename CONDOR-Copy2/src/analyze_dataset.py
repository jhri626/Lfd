import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from initializer import initialize_framework
import os

def create_directories(results_path):
    """
    Creates the requested directory and subfolders
    """
    try:
        if not os.path.exists(results_path + 'images/'):
            os.makedirs(results_path + 'images/')
            os.makedirs(results_path + 'stats/')
        print('Results directory created:', results_path)
    except FileExistsError:
        print('Results directory already exists:', results_path)

def analyze_demonstrations_detailed(data):
    """
    Detailed analysis of demonstration data structure
    """
    print("\n=== Detailed Data Analysis ===")
    
    # Print all keys in data
    print("Data keys:")
    for key in data.keys():
        print(f"  - {key}: {type(data[key])}")
    
    # Analyze demonstrations raw
    demonstrations_raw = data['demonstrations raw']
    print(f"\nDemonstrations raw:")
    print(f"  Type: {type(demonstrations_raw)}")
    print(f"  Length: {len(demonstrations_raw)}")
    
    for i, demo in enumerate(demonstrations_raw):
        print(f"  Demo {i}: shape={demo.shape}, type={type(demo)}")
        if demo.shape[0] > 0:
            print(f"    First point: {demo[0]}")
            print(f"    Last point: {demo[-1]}")
            print(f"    Min values: {np.min(demo, axis=0)}")
            print(f"    Max values: {np.max(demo, axis=0)}")
    
    # Analyze demonstrations train
    if 'demonstrations train' in data:
        demonstrations_train = data['demonstrations train']
        print(f"\nDemonstrations train:")
        print(f"  Type: {type(demonstrations_train)}")
        print(f"  Shape: {demonstrations_train.shape}")
        
    # Analyze workspace boundaries
    if 'x min' in data and 'x max' in data:
        print(f"\nWorkspace boundaries:")
        print(f"  x min: {data['x min']}")
        print(f"  x max: {data['x max']}")
    
    # Analyze goals
    if 'goals training' in data:
        print(f"\nGoals:")
        print(f"  Goals training: {data['goals training']}")

def plot_demonstrations_simple(demonstrations_raw, save_path, title="Raw Demonstrations"):
    """
    Simple plotting of raw demonstrations
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    print(f"\nPlotting {len(demonstrations_raw)} demonstrations...")
    
    all_x_coords = []
    all_y_coords = []
    
    for i, demo in enumerate(demonstrations_raw):
        print(f"Demo {i}: shape={demo.shape}")
        
        if demo.shape[1] >= 2:  # Ensure we have at least x, y coordinates
            x_coords = demo[:, 0]
            y_coords = demo[:, 1]
            
            color = colors[i % len(colors)]
            plt.plot(x_coords, y_coords, color=color, linewidth=2, 
                    label=f'Demo {i+1}', marker='o', markersize=1, alpha=0.8)
            
            # Mark start and end points
            plt.plot(x_coords[0], y_coords[0], color=color, marker='s', 
                    markersize=8, markerfacecolor='white', markeredgewidth=2)
            plt.plot(x_coords[-1], y_coords[-1], color=color, marker='*', 
                    markersize=12, markerfacecolor='yellow', markeredgewidth=2)
            
            # Collect all coordinates for axis scaling
            all_x_coords.extend(x_coords)
            all_y_coords.extend(y_coords)
    
    # Set axis limits with some padding
    if all_x_coords and all_y_coords:
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        plt.xlim(x_min - padding * x_range, x_max + padding * x_range)
        plt.ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")

def plot_demonstrations_with_info(demonstrations_raw, data, save_path):
    """
    Plot demonstrations with additional information
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Raw trajectories
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, demo in enumerate(demonstrations_raw):
        if demo.shape[1] >= 2:
            color = colors[i % len(colors)]
            ax1.plot(demo[:, 0], demo[:, 1], color=color, linewidth=2, 
                    label=f'Demo {i+1}', alpha=0.8)
            ax1.plot(demo[0, 0], demo[0, 1], color=color, marker='o', markersize=8)
            ax1.plot(demo[-1, 0], demo[-1, 1], color=color, marker='*', markersize=10)
    
    ax1.set_title('Raw Demonstrations')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Trajectory lengths
    ax2 = axes[0, 1]
    lengths = [len(demo) for demo in demonstrations_raw]
    ax2.bar(range(len(lengths)), lengths, color='skyblue', alpha=0.7)
    ax2.set_title('Trajectory Lengths')
    ax2.set_xlabel('Demonstration Index')
    ax2.set_ylabel('Number of Points')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Start and End Points
    ax3 = axes[1, 0]
    for i, demo in enumerate(demonstrations_raw):
        if demo.shape[1] >= 2:
            color = colors[i % len(colors)]
            ax3.scatter(demo[0, 0], demo[0, 1], color=color, marker='o', 
                       s=100, alpha=0.7, label=f'Start {i+1}')
            ax3.scatter(demo[-1, 0], demo[-1, 1], color=color, marker='*', 
                       s=150, alpha=0.7, label=f'End {i+1}')
    
    ax3.set_title('Start and End Points')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Data Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""Dataset Statistics:
    
Number of demonstrations: {len(demonstrations_raw)}
    
Trajectory lengths:
  Min: {min(lengths)} points
  Max: {max(lengths)} points
  Mean: {np.mean(lengths):.1f} points
    
Workspace boundaries:
  X: [{data.get('x min', ['?'])[0]:.3f}, {data.get('x max', ['?'])[0]:.3f}]
  Y: [{data.get('x min', ['?', '?'])[1]:.3f}, {data.get('x max', ['?', '?'])[1]:.3f}]
    
Goal position:
  {data.get('goals training', ['?'])[0] if 'goals training' in data else 'Not found'}
    
Dataset: {data.get('dataset_name', 'Unknown')}
Primitive ID: {data.get('selected_primitives_ids', 'Unknown')}
"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed plot saved to: {save_path}")

def main():
    # Get arguments
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, default='2nd_order_2D', help='Parameter file name')
    parser.add_argument('--results-base-directory', type=str, default='./', help='Base directory for results')
    args = parser.parse_args()
    
    print("=== Dataset Loading and Visualization ===")
    print(f"Using parameters: {args.params}")
    
    # Import parameters
    Params = getattr(importlib.import_module('params.' + args.params), 'Params')
    params = Params(args.results_base_directory)
    params.results_path += 'dataset_analysis/' + params.selected_primitives_ids + '/'
    
    # Create results directory
    create_directories(params.results_path)
    
    print(f"\\nParameters:")
    print(f"  Dataset: {params.dataset_name}")
    print(f"  Selected primitives: {params.selected_primitives_ids}")
    print(f"  Workspace dimensions: {params.workspace_dimensions}")
    print(f"  Dynamical system order: {params.dynamical_system_order}")
    print(f"  Results path: {params.results_path}")
    
    # Use existing framework to load data (this is the proven method)
    print("\\nLoading data using existing framework...")
    try:
        learner, evaluator, data = initialize_framework(params, args.params, verbose=True)
        print("? Data loaded successfully using existing framework!")
        
        # Detailed analysis
        analyze_demonstrations_detailed(data)
        
        # Extract demonstrations
        demonstrations_raw = data['demonstrations raw']
        
        # Create visualizations
        print("\\nCreating visualizations...")
        
        # Simple plot
        simple_plot_path = params.results_path + 'images/demonstrations_simple.png'
        plot_demonstrations_simple(demonstrations_raw, simple_plot_path, 
                                  f"Raw Demonstrations - {params.dataset_name}")
        
        # Detailed plot with statistics
        detailed_plot_path = params.results_path + 'images/demonstrations_detailed.png'
        plot_demonstrations_with_info(demonstrations_raw, data, detailed_plot_path)
        
        print("\\n=== Dataset Analysis Completed Successfully! ===")
        print(f"? Loaded {len(demonstrations_raw)} demonstrations")
        print(f"? Created visualizations in: {params.results_path}images/")
        print("\\nFiles created:")
        print(f"  - {simple_plot_path}")
        print(f"  - {detailed_plot_path}")
        
        # Print summary
        print("\\n=== Summary ===")
        for i, demo in enumerate(demonstrations_raw):
            print(f"Demo {i+1}: {len(demo)} points, shape={demo.shape}")
            
    except Exception as e:
        print(f"? Error loading data: {e}")
        print("\\nTrying alternative data loading method...")
        
        # Alternative: Direct data loading
        try:
            from data_preprocessing.data_preprocessor import DataPreprocessor
            data_preprocessor = DataPreprocessor(params=params, verbose=True)
            data = data_preprocessor.run()
            
            demonstrations_raw = data['demonstrations raw']
            print(f"? Alternative method successful! Loaded {len(demonstrations_raw)} demonstrations")
            
            # Create simple visualization
            simple_plot_path = params.results_path + 'images/demonstrations_alternative.png'
            plot_demonstrations_simple(demonstrations_raw, simple_plot_path, 
                                      f"Demonstrations (Alternative Loading) - {params.dataset_name}")
            
        except Exception as e2:
            print(f"? Alternative method also failed: {e2}")
            return

if __name__ == "__main__":
    main()
