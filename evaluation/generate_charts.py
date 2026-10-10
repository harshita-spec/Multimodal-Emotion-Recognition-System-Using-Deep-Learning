import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_evaluation_plots():
    eval_dir = os.path.dirname(__file__)
    res_path = os.path.join(eval_dir, "evaluation_results.json")
    
    if not os.path.exists(res_path):
        print("Please run evaluate_all.py first to generate evaluation_results.json")
        return
        
    with open(res_path, "r") as f:
        data = json.load(f)
        
    labels = data["confusion_matrix"]["labels"]
    cm = np.array(data["confusion_matrix"]["matrix"])
    
    # 1. Confusion Matrix Heatmap Plot (using pure Matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='left')
    ax.set_yticklabels(labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center',
                    color='white' if cm[i, j] > np.max(cm)/2 else 'black')
                    
    plt.title("Attention Fusion Model Confusion Matrix", pad=20)
    plt.xlabel("Predicted Emotion Class")
    plt.ylabel("Ground Truth Emotion Class")
    plt.tight_layout()
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 2. Modality Performance Comparison Bar Chart
    modalities = list(data["modalities"].keys())
    accuracies = [data["modalities"][m]["accuracy"] * 100 for m in modalities]
    f1_scores = [data["modalities"][m]["weighted_f1"] * 100 for m in modalities]
    
    x = np.arange(len(modalities))
    width = 0.35
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3b82f6')
    plt.bar(x + width/2, f1_scores, width, label='Weighted F1 (%)', color='#10b981')
    
    plt.ylabel('Score (%)')
    plt.title('Modality & Fusion Method Performance Benchmark Comparison')
    plt.xticks(x, modalities, rotation=25, ha='right')
    plt.ylim(50, 100)
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(eval_dir, "modality_comparison.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    
    print(f"Evaluation charts successfully saved to:\n  - {cm_path}\n  - {bar_path}\n")

if __name__ == "__main__":
    generate_evaluation_plots()
