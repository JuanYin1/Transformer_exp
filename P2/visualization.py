# CSE256 PA2 - Part 3 Visualization Module
# The visualization.py file includes:
#   1. AttentionVisualizer class: For attention pattern visualizations
#   2. TrainingVisualizer class: For training dynamics and performance comparisons
#   3. ArchitectureVisualizer class: For architectural components and position encodings
#   4. ResultsDashboard class: For comprehensive performance dashboards
#   5. create_all_visualizations function: Convenience function to run everything

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class AttentionVisualizer:
    """Class for visualizing attention patterns across different architectures"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def visualize_attention_patterns(self, models_dict, sample_text, tokenizer, save_path='attention_patterns.png'):
        """Compare attention patterns across different architectures"""
        n_models = len(models_dict)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        # Encode sample text
        tokens = tokenizer.encode(sample_text)[:16]  # Limit to 16 tokens
        token_words = sample_text.split()[:len(tokens)]
        input_tensor = torch.tensor([tokens]).to(self.device)
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'encoder'):  # Classification models
                    _, attention_maps = model.encoder(input_tensor)
                else:  # Language models - would need modification
                    print(f"Note: {model_name} attention visualization not implemented for LM")
                    continue
                    
            # Plot first layer, first head
            if attention_maps and len(attention_maps) > 0:
                # Debug: print attention structure
                print(f"Debug {model_name}: attention_maps[0] shape = {attention_maps[0].shape}")
                
                # Extract attention matrix - handle different shapes
                raw_attn = attention_maps[0]
                if len(raw_attn.shape) == 4:  # [batch, head, seq, seq]
                    attn_matrix = raw_attn[0, 0].cpu().numpy()
                elif len(raw_attn.shape) == 3:  # [head, seq, seq] or [batch, seq, seq]
                    if raw_attn.shape[0] <= 8:  # Likely heads
                        attn_matrix = raw_attn[0].cpu().numpy()
                    else:  # Likely batch
                        attn_matrix = raw_attn[0].cpu().numpy()
                else:
                    attn_matrix = raw_attn.cpu().numpy()
                
                # Ensure 2D matrix
                while len(attn_matrix.shape) > 2:
                    attn_matrix = attn_matrix[0]
                
                actual_len = min(len(tokens), attn_matrix.shape[0])
                attn_matrix = attn_matrix[:actual_len, :actual_len]
                
                print(f"Debug {model_name}: final attn_matrix shape = {attn_matrix.shape}")
                
                ax = axes[idx]
                im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
                ax.set_title(f'{model_name}\nLayer 0, Head 0')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                
                # Add token labels if few enough tokens
                if actual_len <= 10:
                    ax.set_xticks(range(actual_len))
                    ax.set_yticks(range(actual_len))
                    ax.set_xticklabels(token_words[:actual_len], rotation=45)
                    ax.set_yticklabels(token_words[:actual_len])
                
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Attention patterns saved to {save_path}")
        plt.show()
        
    def visualize_attention_heads(self, model, sample_text, tokenizer, layer=0, save_path='attention_heads.png'):
        """Visualize all attention heads in a specific layer"""
        tokens = tokenizer.encode(sample_text)[:12]  # Limit for visibility
        token_words = sample_text.split()[:len(tokens)]
        input_tensor = torch.tensor([tokens]).to(self.device)
        
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'encoder'):
                _, attention_maps = model.encoder(input_tensor)
            else:
                print("Attention head visualization not implemented for this model type")
                return
                
        if not attention_maps or layer >= len(attention_maps):
            print(f"Layer {layer} not available")
            return
            
        # Get attention for specified layer
        layer_attention = attention_maps[layer][0]  # Shape: [n_heads, seq_len, seq_len]
        n_heads = layer_attention.shape[0]
        
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        
        if n_heads == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten() if cols > 1 else [axes]
        else:
            axes = axes.flatten()
            
        for head in range(n_heads):
            attn_matrix = layer_attention[head].cpu().numpy()
            actual_len = min(len(tokens), attn_matrix.shape[0])
            attn_matrix = attn_matrix[:actual_len, :actual_len]
            
            ax = axes[head]
            im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
            ax.set_title(f'Head {head}')
            
            if actual_len <= 8:  # Only show labels for small sequences
                ax.set_xticks(range(actual_len))
                ax.set_yticks(range(actual_len))
                ax.set_xticklabels(token_words[:actual_len], rotation=45, fontsize=8)
                ax.set_yticklabels(token_words[:actual_len], fontsize=8)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for head in range(n_heads, len(axes)):
            axes[head].set_visible(False)
            
        plt.suptitle(f'All Attention Heads - Layer {layer}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Attention heads saved to {save_path}")
        plt.show()


class TrainingVisualizer:
    """Class for visualizing training dynamics and performance comparisons"""
    
    def plot_training_comparison(self, results_dict, save_path='training_comparison.png'):
        """Plot comprehensive training comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Classification Accuracy Over Time
        if 'classification' in results_dict:
            for model_name, accuracies in results_dict['classification'].items():
                epochs = range(1, len(accuracies) + 1)
                ax1.plot(epochs, accuracies, label=model_name, marker='o', linewidth=2, markersize=4)
            ax1.set_title('Classification Accuracy Over Epochs', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Test Accuracy (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
        
        # Language Model Perplexity (if available)
        if 'language_model' in results_dict:
            for model_name, perplexities in results_dict['language_model'].items():
                steps = range(0, len(perplexities) * 100, 100)  # Assuming every 100 steps
                ax2.plot(steps, perplexities, label=model_name, marker='s', linewidth=2, markersize=4)
            ax2.set_title('Language Model Perplexity During Training', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Perplexity')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Parameter Efficiency Analysis
        if 'parameters' in results_dict and 'classification' in results_dict:
            models = list(results_dict['parameters'].keys())
            params = [results_dict['parameters'][m] for m in models if m in results_dict['classification']]
            final_accs = [results_dict['classification'][m][-1] for m in models if m in results_dict['classification']]
            model_names = [m for m in models if m in results_dict['classification']]
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, (model, param, acc) in enumerate(zip(model_names, params, final_accs)):
                ax3.scatter(param, acc, s=200, alpha=0.7, color=colors[i % len(colors)], 
                           label=model, edgecolors='black', linewidth=1)
                ax3.annotate(model, (param, acc), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax3.set_title('Parameter Efficiency (Classification)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Parameters')
            ax3.set_ylabel('Final Test Accuracy (%)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        # Computational Complexity Comparison
        seq_lengths = [8, 16, 24, 32, 48, 64]
        full_attention = [s**2 for s in seq_lengths]  # O(n²)
        local_window_8 = [s * 8 for s in seq_lengths]   # O(n*w) with w=8
        local_window_16 = [s * 16 for s in seq_lengths]  # O(n*w) with w=16
        
        ax4.plot(seq_lengths, full_attention, label='Full Attention O(n²)', 
                marker='o', linewidth=2, color='red', markersize=6)
        ax4.plot(seq_lengths, local_window_8, label='Local Window O(n×w), w=8', 
                marker='s', linewidth=2, color='green', markersize=6)
        ax4.plot(seq_lengths, local_window_16, label='Local Window O(n×w), w=16', 
                marker='^', linewidth=2, color='blue', markersize=6)
        
        ax4.set_title('Attention Computational Complexity', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Operations Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training comparison saved to {save_path}")
        plt.show()
    
    def plot_loss_curves(self, loss_dict, save_path='loss_curves.png'):
        """Plot training loss curves for different models"""
        plt.figure(figsize=(12, 6))
        
        for model_name, losses in loss_dict.items():
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, label=model_name, linewidth=2, marker='o', markersize=3)
        
        plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Loss curves saved to {save_path}")
        plt.show()


class ArchitectureVisualizer:
    """Class for visualizing architectural components and comparisons"""
    
    def visualize_position_encodings(self, save_path='position_encodings.png'):
        """Compare different positional encoding approaches"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        seq_len = 32
        d_model = 64
        
        # 1. Learnable Embeddings (simulate learned patterns)
        np.random.seed(42)  # For reproducibility
        learned_pos = np.random.randn(seq_len, d_model) * 0.1
        # Add some structure to make it more realistic
        for i in range(seq_len):
            learned_pos[i] += 0.05 * np.sin(np.arange(d_model) * 2 * np.pi / d_model * (i + 1))
        
        im1 = axes[0,0].imshow(learned_pos.T, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
        axes[0,0].set_title('Learnable Position Embeddings\n(Simulated)', fontweight='bold')
        axes[0,0].set_xlabel('Position')
        axes[0,0].set_ylabel('Embedding Dimension')
        plt.colorbar(im1, ax=axes[0,0])
        
        # 2. Sinusoidal Embeddings
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        sinusoidal = torch.zeros(seq_len, d_model)
        sinusoidal[:, 0::2] = torch.sin(position * div_term)
        sinusoidal[:, 1::2] = torch.cos(position * div_term)
        
        im2 = axes[0,1].imshow(sinusoidal.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        axes[0,1].set_title('Sinusoidal Position Encodings\n(Transformer Original)', fontweight='bold')
        axes[0,1].set_xlabel('Position')
        axes[0,1].set_ylabel('Embedding Dimension')
        plt.colorbar(im2, ax=axes[0,1])
        
        # 3. AliBi Bias Visualization
        n_heads = 4
        slopes = [2**(-8*i/n_heads) for i in range(1, n_heads+1)]
        position_diffs = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        
        # Show bias for first head
        alibi_bias = slopes[0] * position_diffs
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        alibi_bias = alibi_bias.masked_fill(causal_mask == 0, float('-inf'))
        alibi_bias = alibi_bias.numpy()
        
        im3 = axes[1,0].imshow(alibi_bias, cmap='RdBu', aspect='auto')
        axes[1,0].set_title(f'AliBi Attention Bias (Head 1)\nslope = {slopes[0]:.4f}', fontweight='bold')
        axes[1,0].set_xlabel('Key Position')
        axes[1,0].set_ylabel('Query Position')
        plt.colorbar(im3, ax=axes[1,0])
        
        # 4. Local Window Attention Mask
        window_size = 8
        local_mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            local_mask[i, start:end] = 1
        
        im4 = axes[1,1].imshow(local_mask, cmap='Blues', aspect='auto')
        axes[1,1].set_title(f'Local Window Attention Mask\nwindow size = {window_size}', fontweight='bold')
        axes[1,1].set_xlabel('Key Position') 
        axes[1,1].set_ylabel('Query Position')
        plt.colorbar(im4, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Position encodings saved to {save_path}")
        plt.show()
    
    def visualize_causal_masking(self, save_path='causal_masking_comparison.png'):
        """Visualize the difference between encoder and decoder attention patterns"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        seq_len = 16
        
        # 1. Encoder: Full Bidirectional Attention
        encoder_mask = torch.ones(seq_len, seq_len)
        im1 = axes[0].imshow(encoder_mask, cmap='Blues', aspect='auto')
        axes[0].set_title('Encoder Attention\n(Bidirectional)', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Key Position')
        axes[0].set_ylabel('Query Position')
        axes[0].text(seq_len//2, seq_len//2, 'Can attend to\nALL positions', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        plt.colorbar(im1, ax=axes[0])
        
        # 2. Decoder: Causal Masking
        decoder_mask = torch.tril(torch.ones(seq_len, seq_len))
        im2 = axes[1].imshow(decoder_mask, cmap='Blues', aspect='auto')
        axes[1].set_title('Decoder Attention\n(Causal/Autoregressive)', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Key Position')
        axes[1].set_ylabel('Query Position')
        axes[1].text(seq_len//4, 3*seq_len//4, 'Can only attend to\nPAST positions', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.8))
        plt.colorbar(im2, ax=axes[1])
        
        # 3. Local Window: Sparse Attention
        window_size = 6
        local_mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            local_mask[i, start:end] = 1
        
        im3 = axes[2].imshow(local_mask, cmap='Blues', aspect='auto')
        axes[2].set_title(f'Local Window Attention\n(Window size = {window_size})', fontweight='bold', fontsize=12)
        axes[2].set_xlabel('Key Position')
        axes[2].set_ylabel('Query Position')
        axes[2].text(seq_len//2, seq_len//2, 'Can only attend to\nLOCAL window', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        plt.colorbar(im3, ax=axes[2])
        
        # Add annotations
        for i, (ax, title) in enumerate(zip(axes, ['PART 1', 'PART 2', 'PART 3'])):
            ax.text(0.02, 0.98, title, transform=ax.transAxes, fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8, edgecolor='black'),
                   verticalalignment='top', color='white')
        
        plt.suptitle('Attention Patterns: Encoder vs Decoder vs Local Window', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Causal masking comparison saved to {save_path}")
        plt.show()

    def create_architecture_diagram(self, save_path='architecture_comparison.png'):
        """Create a visual comparison of different architectures"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        # Architecture components
        components = ['Input Embedding', 'Position Encoding', 'Multi-Head Attention', 
                     'Feed Forward', 'Layer Norm', 'Output']
        
        architectures = {
            'Traditional Transformer': {
                'Position Encoding': 'Learnable/Sinusoidal',
                'Attention': 'Full O(n²)',
                'Efficiency': 'Baseline'
            },
            'AliBi Transformer': {
                'Position Encoding': 'Attention Biases',
                'Attention': 'Full O(n²)',
                'Efficiency': 'Parameter Efficient'
            },
            'Local Window Transformer': {
                'Position Encoding': 'Learnable',
                'Attention': 'Sparse O(n×w)',
                'Efficiency': 'Compute Efficient'
            }
        }
        
        colors = {'Traditional Transformer': '#1f77b4', 'AliBi Transformer': '#ff7f0e', 
                 'Local Window Transformer': '#2ca02c'}
        
        for i, (arch_name, details) in enumerate(architectures.items()):
            ax = axes[i]
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_title(arch_name, fontsize=14, fontweight='bold')
            
            # Draw architecture blocks
            blocks = [
                ('Input\nEmbedding', 2, 8, 2, 1),
                ('Position\nEncoding', 6, 8, 2, 1),
                ('Multi-Head\nAttention', 2, 6, 6, 1),
                ('Feed\nForward', 2, 4, 6, 1),
                ('Layer Norm', 2, 2, 6, 1),
                ('Output', 4, 0.5, 2, 1)
            ]
            
            for block_name, x, y, w, h in blocks:
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='black', 
                               facecolor=colors[arch_name], alpha=0.7)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, block_name, ha='center', va='center', 
                       fontsize=9, fontweight='bold')
            
            # Add architecture-specific annotations
            if 'AliBi' in arch_name:
                ax.text(7, 8.5, '❌ No Position\nEmbedding', ha='center', va='center', 
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
                ax.text(5, 6.5, '📏 Linear Biases', ha='center', va='center', 
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            
            elif 'Local Window' in arch_name:
                ax.text(5, 6.5, '🔍 Sparse\nAttention', ha='center', va='center', 
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
            
            # Add efficiency info
            ax.text(5, 0.2, f"Efficiency: {details['Efficiency']}", ha='center', va='center',
                   fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black'))
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Architecture diagram saved to {save_path}")
        plt.show()


class ResultsDashboard:
    """Class for creating comprehensive results dashboard"""
    
    def create_performance_dashboard(self, results, save_path='performance_dashboard.png'):
        """Create a comprehensive dashboard of all results"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model Comparison Table
        ax1 = fig.add_subplot(gs[0, :2])
        table_data = []
        
        for model in results.keys():
            row = [
                model,
                f"{results[model].get('params', 0):,}",
                f"{results[model].get('accuracy', 0):.1f}%",
                f"{results[model].get('perplexity', 0):.1f}" if results[model].get('perplexity', 0) > 0 else "N/A",
                f"{results[model].get('training_time', 0):.1f}s"
            ]
            table_data.append(row)
        
        table = ax1.table(cellText=table_data,
                         colLabels=['Model', 'Parameters', 'Accuracy', 'Perplexity', 'Time'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.5)
        
        # Color code the table
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
        for i in range(len(table_data)):
            for j in range(5):
                table[(i+1, j)].set_facecolor(colors[i % len(colors)])
        
        ax1.axis('off')
        ax1.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        
        # 2. Performance Radar Chart (if multiple metrics available)
        ax2 = fig.add_subplot(gs[0, 2:])
        if len(results) >= 2:
            models = list(results.keys())[:3]  # Limit to 3 models for clarity
            metrics = ['Accuracy', 'Parameter Efficiency', 'Speed']
            
            # Normalize metrics to 0-1 scale
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax2 = plt.subplot(gs[0, 2:], projection='polar')
            colors_radar = ['blue', 'red', 'green']
            
            for i, model in enumerate(models):
                values = [
                    results[model].get('accuracy', 0) / 100,  # Normalize accuracy
                    1 - (results[model].get('params', 1e6) / 1e6),  # Parameter efficiency (inverse)
                    1 / max(1, results[model].get('training_time', 1))  # Speed (inverse of time)
                ]
                values += values[:1]  # Complete the circle
                
                ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[i])
                ax2.fill(angles, values, alpha=0.25, color=colors_radar[i])
            
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(metrics)
            ax2.set_ylim(0, 1)
            ax2.set_title('Performance Radar Chart', fontsize=12, fontweight='bold', pad=20)
            ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 3. Training Progress
        ax3 = fig.add_subplot(gs[1, :])
        if any('accuracy_curve' in results[m] for m in results):
            for model in results:
                if 'accuracy_curve' in results[model]:
                    epochs = range(1, len(results[model]['accuracy_curve']) + 1)
                    ax3.plot(epochs, results[model]['accuracy_curve'], 
                            label=model, linewidth=3, marker='o', markersize=4)
            
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
            ax3.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        # 4. Architecture Innovation Highlights
        ax4 = fig.add_subplot(gs[2, :2])
        innovations = [
            "🚀 AliBi: No position embeddings needed",
            "💡 Local Window: O(n²) → O(n×w) complexity",
            "⚡ Parameter sharing: Reduced model size",
            "🎯 Enhanced architecture: Better accuracy",
            "📊 Comparative analysis: Multiple metrics"
        ]
        
        y_pos = np.arange(len(innovations))
        ax4.barh(y_pos, [1]*len(innovations), color=['gold', 'lightblue', 'lightgreen', 'orange', 'pink'])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(innovations, fontsize=11)
        ax4.set_xlabel('Innovation Impact')
        ax4.set_title('Key Architectural Innovations', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, 1.2)
        
        for i, txt in enumerate(['High', 'High', 'Medium', 'High', 'High']):
            ax4.text(1.05, i, txt, va='center', fontweight='bold')
        
        # 5. Memory and Computation Analysis
        ax5 = fig.add_subplot(gs[2, 2:])
        if results:
            models = list(results.keys())
            memory_usage = [results[m].get('params', 0) / 1000 for m in models]  # Convert to thousands
            accuracy = [results[m].get('accuracy', 0) for m in models]
            
            colors_scatter = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (model, mem, acc) in enumerate(zip(models, memory_usage, accuracy)):
                ax5.scatter(mem, acc, s=300, alpha=0.7, color=colors_scatter[i % len(colors_scatter)], 
                           label=model, edgecolors='black', linewidth=2)
                ax5.annotate(model, (mem, acc), xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, fontweight='bold')
            
            ax5.set_xlabel('Parameters (thousands)', fontsize=12)
            ax5.set_ylabel('Accuracy (%)', fontsize=12)
            ax5.set_title('Memory vs Performance Trade-off', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim(0, 100)
        
        # Add overall title
        fig.suptitle('CSE256 PA2 - Part 3: Architectural Exploration Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance dashboard saved to {save_path}")
        plt.show()


# Convenience function to run all visualizations
def create_all_visualizations(models_dict, results_dict, tokenizer, device='cpu'):
    """Run all visualization functions"""
    print("🎨 Creating comprehensive visualizations...")
    
    # Initialize visualizers
    attn_viz = AttentionVisualizer(device)
    train_viz = TrainingVisualizer()
    arch_viz = ArchitectureVisualizer()
    dashboard = ResultsDashboard()
    
    # Sample text for attention visualization
    sample_text = "The president spoke about the economy and foreign policy"
    
    try:
        # 1. Attention patterns
        if models_dict:
            attn_viz.visualize_attention_patterns(models_dict, sample_text, tokenizer)
        
        # 2. Position encodings
        arch_viz.visualize_position_encodings()
        
        # 3. Architecture comparison
        arch_viz.create_architecture_diagram()
        
        # 4. Training comparison
        if results_dict:
            train_viz.plot_training_comparison(results_dict)
        
        # 5. Performance dashboard
        if results_dict:
            dashboard.create_performance_dashboard(results_dict)
        
        print("🎉 All visualizations completed successfully!")
        
    except Exception as e:
        print(f"⚠️ Error in visualization: {e}")
        print("Continuing with available visualizations...")


if __name__ == "__main__":
    print("📊 CSE256 PA2 Visualization Module")
    print("Import this module in main.py to use visualization functions")