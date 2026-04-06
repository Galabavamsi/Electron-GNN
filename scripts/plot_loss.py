import re
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs, train_bipartite, val_bipartite, train_spectrum, val_spectrum = [], [], [], [], []
    
    with open(log_path, 'r') as f:
        content = f.read()
        
    epoch_blocks = re.split(r'Epoch \d+/\d+', content)
    
    for i, block in enumerate(epoch_blocks[1:]): # skip first split that is empty or preamble
        epochs.append(i + 1)
        
        train_match = re.search(r'Train - Bipartite: ([\d.]+), Spectrum: ([\d.]+)', block)
        val_match = re.search(r'Val\s+- Bipartite: ([\d.]+), Spectrum: ([\d.]+)', block)
        
        if train_match and val_match:
            train_bipartite.append(float(train_match.group(1)))
            train_spectrum.append(float(train_match.group(2)))
            val_bipartite.append(float(val_match.group(1)))
            val_spectrum.append(float(val_match.group(2)))
            
    return epochs, train_bipartite, val_bipartite, train_spectrum, val_spectrum

def generate_plots():
    epochs, tb, vb, ts, vs = parse_log('results/train_output.log')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, tb, label='Train Bipartite Loss')
    plt.plot(epochs, vb, label='Val Bipartite Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bipartite Matching Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ts, label='Train Spectrum Loss')
    plt.plot(epochs, vs, label='Val Spectrum Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Spectrum Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/loss_curves.png')
    print("Saved plot to results/loss_curves.png")

if __name__ == "__main__":
    generate_plots()
