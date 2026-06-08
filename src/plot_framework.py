"""
Generate SCI Framework Architecture Diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title('SCIT: Staged Conditional Integration with Threshold Calibration', fontsize=14, fontweight='bold', pad=20)

def draw_box(ax, x, y, w, h, text, color='#E8F4FD', edgecolor='#2196F3', fontsize=8):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#666666'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def draw_arrow_label(ax, x1, y1, x2, y2, label, color='#666666'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mid_x, mid_y + 0.15, label, fontsize=7, ha='center', color=color)

y_top = 11

ax.text(8, y_top + 0.3, 'EEG Input Features', fontsize=11, fontweight='bold', ha='center')
draw_arrow(ax, 8, y_top, 8, y_top - 0.5)

y_modules = 9.5
draw_box(ax, 1, y_modules - 0.6, 3.5, 1.2, 'PCET Module\nPredictive Coding\np(y|x)', '#E3F2FD', '#1565C0', 8)
draw_box(ax, 6, y_modules - 0.6, 3.5, 1.2, 'SRGC Module\nUncertainty Estimation\nu(x)', '#E8F5E9', '#2E7D32', 8)
draw_box(ax, 11, y_modules - 0.6, 3.5, 1.2, 'SIED Module\nDomain Shift Detection\nd(x)', '#FFF3E0', '#EF6C00', 8)

draw_arrow(ax, 8, y_top - 0.5, 2.75, y_modules)
draw_arrow(ax, 8, y_top - 0.5, 7.75, y_modules)
draw_arrow(ax, 8, y_top - 0.5, 12.75, y_modules)

ax.text(2.75, y_modules + 0.8, 'X', fontsize=9, ha='center')
ax.text(7.75, y_modules + 0.8, 'X', fontsize=9, ha='center')
ax.text(12.75, y_modules + 0.8, 'X', fontsize=9, ha='center')

y_info = 7.5
ax.text(8, y_info + 1.2, 'Three Orthogonal Information Sources', fontsize=11, fontweight='bold', ha='center', color='#333333')

info_box_w = 4
draw_box(ax, 0.5, y_info - 0.5, info_box_w, 1, 'PCET\nClass Prediction\np(y=1|x) ∈ [0,1]', '#BBDEFB', '#1565C0', 8)
draw_box(ax, 5.5, y_info - 0.5, info_box_w, 1, 'SRGC\nUncertainty\nu(x) ∈ [0,1]', '#C8E6C9', '#2E7D32', 8)
draw_box(ax, 10.5, y_info - 0.5, info_box_w, 1, 'SIED\nDomain Shift\nd(x) ∈ [0,1]', '#FFE0B2', '#EF6C00', 8)

draw_arrow(ax, 2.75, y_modules - 0.6, 2.25, y_info + 0.5)
draw_arrow(ax, 7.75, y_modules - 0.6, 7.5, y_info + 0.5)
draw_arrow(ax, 12.75, y_modules - 0.6, 12.5, y_info + 0.5)

y_calib = 5.5
ax.text(8, y_calib + 1.5, 'Threshold Calibration', fontsize=11, fontweight='bold', ha='center', color='#333333')

draw_box(ax, 5.5, y_calib - 0.3, 4.5, 1.0, 'τ(x) = τ₀ + β*(u-μ_u)/σ_u\n         + γ*(d-μ_d)/σ_d',
         '#F3E5F5', '#7B1FA2', 8)

draw_arrow(ax, 2.25, y_info - 0.5, 5.5, y_calib + 0.7)
draw_arrow(ax, 7.5, y_info - 0.5, 7.75, y_calib + 0.7)
draw_arrow(ax, 12.5, y_info - 0.5, 9.5, y_calib + 0.7)

y_decision = 3.8
draw_box(ax, 5, y_decision - 0.5, 5.5, 1.0, 'Decision Rule\nif p(y|x) >= τ(x): pred=1\nelse: pred=0',
         '#E0E0E0', '#616161', 9)

draw_arrow(ax, 7.75, y_calib - 0.3, 7.75, y_decision + 0.5)

y_output = 2.5
draw_box(ax, 5.5, y_output - 0.4, 4.5, 0.8, 'Prediction Output\ny_pred ∈ {0, 1}', '#C5CAE9', '#303F9F', 9)
draw_arrow(ax, 7.75, y_decision - 0.5, 7.75, y_output + 0.4)

y_innovations = 1.2
ax.text(8, y_innovations + 1.5, 'Key Innovations', fontsize=11, fontweight='bold', ha='center', color='#333333')

innovations = [
    ('PCET: Error Features\n+E_recon = |X - X̂|', 0.5, '#E3F2FD'),
    ('SRGC: Uncertainty\nNot class, but u(x)', 5.5, '#E8F5E9'),
    ('SIED: Domain Shift\nNot class, but d(x)', 10.5, '#FFF3E0'),
]

for text, x, color in innovations:
    draw_box(ax, x, y_innovations - 0.6, 4.2, 0.9, text, color, '#757575', 7)

ax.text(8, 0.3, 'Average Improvement: +0.41% across all shots (3, 5, 10, 20, 50)', fontsize=10, ha='center', style='italic', color='#666666')

plt.tight_layout()
plt.savefig('d:/pycharmproject/zuco-benchmark-main/src/results/final/SCIT_framework_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('d:/pycharmproject/zuco-benchmark-main/src/results/final/SCIT_framework_diagram.pdf', bbox_inches='tight', facecolor='white')
print("Diagram saved to results/final/SCIT_framework_diagram.png and .pdf")

plt.figure(figsize=(14, 8))
ax2 = fig.add_subplot(111)
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')

y = 9.5
ax2.text(7, y, 'Performance Comparison: SCI_V7 vs PCET', fontsize=13, fontweight='bold', ha='center')

shots = ['3-shot', '5-shot', '10-shot', '20-shot', '50-shot']
pcet_acc = [59.07, 62.50, 66.95, 72.39, 78.42]
sci_acc = [59.95, 62.98, 67.36, 72.39, 78.68]

x_pos = np.arange(len(shots))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, pcet_acc, width, label='PCET', color='#1565C0', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, sci_acc, width, label='SCI_V7', color='#2E7D32', alpha=0.8)

for bar, acc in zip(bars1, pcet_acc):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
for bar, acc in zip(bars2, sci_acc):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

improvements = [sci_acc[i] - pcet_acc[i] for i in range(len(shots))]
for i, imp in enumerate(improvements):
    color = 'green' if imp > 0 else 'gray'
    ax2.text(x_pos[i] + width/2 + 0.2, sci_acc[i] + 1.5, f'+{imp:.2f}%', fontsize=8, color=color, ha='left')

ax2.set_ylabel('Accuracy (%)', fontsize=10)
ax2.set_xlabel('Shot Setting', fontsize=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(shots)
ax2.legend(loc='upper left')
ax2.set_ylim(55, 85)
ax2.grid(axis='y', alpha=0.3)

ax2.text(7, 5.5, 'Improvement range: +0.26% to +0.88%\nAll 5 shots achieve equal or better accuracy', fontsize=9, ha='center', style='italic', color='#666666')

plt.tight_layout()
plt.savefig('d:/pycharmproject/zuco-benchmark-main/src/results/final/SCIT_performance_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Performance chart saved to results/final/SCIT_performance_comparison.png")

plt.show()
print("Done!")