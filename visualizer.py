import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np

def plot(images, bboxes, num_samples=4):
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(images))):
        img = images[i].numpy()
        bbox = bboxes[i].numpy()
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        height, width = img.shape[0], img.shape[1]
        x_min, y_min, x_max, y_max = bbox
        
        if x_max <= 1.0: 
            x_min, x_max = x_min * width, x_max * width
            y_min, y_max = y_min * height, y_max * height
        
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[i].add_patch(rect)
        
        axes[i].text(x_min, y_min-5, f'({x_min:.1f},{y_min:.1f})', 
                    color='red', fontsize=8, weight='bold')
    
    plt.tight_layout()
    plt.show()
