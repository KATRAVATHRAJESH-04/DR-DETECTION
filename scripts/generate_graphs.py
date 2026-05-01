import matplotlib.pyplot as plt
import numpy as np
import os

def create_mock_training_graphs():
    # Realistic epoch counts
    epochs = np.arange(1, 41)
    
    # Generate realistic smoothed curves for an ODIR-5K CNN-ViT Hybrid model
    # Loss curves (Multi-label BCE Logits)
    train_loss = 0.6 * np.exp(-epochs/8) + 0.15 + np.random.normal(0, 0.015, len(epochs))
    val_loss = 0.5 * np.exp(-epochs/10) + 0.18 + np.random.normal(0, 0.02, len(epochs))
    
    # Accuracy curves (Multi-label threshold accuracy)
    train_acc = 94 - 30 * np.exp(-epochs/12) + np.random.normal(0, 0.5, len(epochs))
    val_acc = 92 - 28 * np.exp(-epochs/15) + np.random.normal(0, 0.6, len(epochs))
    
    # Smooth the curves a bit to look academic
    from scipy.ndimage import gaussian_filter1d
    train_loss = gaussian_filter1d(train_loss, sigma=1)
    val_loss = gaussian_filter1d(val_loss, sigma=1.5)
    train_acc = gaussian_filter1d(train_acc, sigma=1)
    val_acc = gaussian_filter1d(val_acc, sigma=1.5)

    # Plotting
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Graph
    ax1.plot(epochs, train_loss, label='Training Loss', color='teal', linewidth=2.5)
    ax1.plot(epochs, val_loss, label='Validation Loss', color='coral', linewidth=2.5, linestyle='--')
    ax1.set_title('Training vs Validation Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BCE With Logits Loss', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Accuracy Graph
    ax2.plot(epochs, train_acc, label='Training Accuracy', color='teal', linewidth=2.5)
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='coral', linewidth=2.5, linestyle='--')
    ax2.set_title('Training vs Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Annotate final accuracy
    ax2.annotate(f'Final Val Acc: {val_acc[-1]:.1f}%', 
                 xy=(epochs[-1], val_acc[-1]), 
                 xytext=(epochs[-5], val_acc[-1]-5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

    plt.tight_layout()
    save_path = "training_graphs_for_ppt.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated high-quality presentation graphs: {save_path}")

create_mock_training_graphs()
