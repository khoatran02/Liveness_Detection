import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def folder_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        # subindent = ' ' * 4 * (level + 1)
        
def get_path_images(root_path):
    file_paths = []
    labels = []
    for category in ['live', 'spoof']:
        category_dir = os.path.join(root_path, category)
        for file_name in os.listdir(category_dir):
            file_paths.append(os.path.join(category, file_name))
            labels.append(category)
        
    df = pd.DataFrame({'path': file_paths, 'label': labels})
    
    return df

def move_files(df,base_dir, target_dir):
    for _, row in df.iterrows():
        # Define the source path and the destination path
        src_path = os.path.join(base_dir, row['path'])
        dst_path = os.path.join(base_dir, target_dir, row['label'], os.path.basename(row['path']))
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(src_path, dst_path)
        
def visualize_data(df, base_dir, rows=5, cols=10):
    """
    Visualizes a grid of images from the dataset.

    Parameters:
    - df: pandas DataFrame containing the file paths and labels.
    - base_dir: str, the base directory where image files are located.
    - rows: int, the number of rows in the image grid.
    - cols: int, the number of columns in the image grid.
    """
    # Sample 'n' images from the dataframe (n = rows * cols)
    sampled_df = df.sample(n=rows*cols, random_state=42)
    
    # Initialize the plot
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # Flatten the axes array for easy iteration
    axes = axes.ravel()
    
    for i, (index, row) in enumerate(sampled_df.iterrows()):
        # Construct the full path to the image file
        img_path = os.path.join(base_dir, row['path'])
        # Read the image
        img = mpimg.imread(img_path)
        # Display the image
        axes[i].imshow(img)
        # Set the title to the label of the image
        axes[i].set_title(row['label'], fontsize=8)
        # Turn off the axes
        axes[i].axis('off')
    
    # Adjust the layout
    plt.tight_layout()
    plt.show()