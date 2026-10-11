import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os

def draw_sigmoid_block(ax, x, y, size=0.5, label='σ', facecolor='#80cfff'):
    """Disegna un blocco quadrato con un'etichetta al centro (es. σ)."""
    rect = patches.Rectangle((x - size/2, y - size/2), size, size,
                             linewidth=1.5, edgecolor='black', facecolor=facecolor, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

def draw_arrow(ax, start, end, label=None, text_offset=0.2, fontweight='normal', color='black'):
    """Disegna una freccia semplice con un'etichetta opzionale."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
                                lw=1.5, color=color, zorder=2))
    if label:
        ax.text(end[0], end[1] + text_offset, label, ha='center', va='bottom',
                fontsize=11, fontweight=fontweight, color=color, zorder=4)

def draw_curved_arrow(ax, start, mid, end, color='black'):
    """Disegna una freccia curva (utilizzata per l'input X_t)."""
    verts = [start, mid, end]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=1.5, zorder=2)
    ax.add_patch(patch)
    ax.plot(end[0], end[1], marker='>', markersize=8, color=color, zorder=2)

# --- Configurazione del grafico ---
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_axis_off()
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 9)

# --- Disegno del diagramma principale ---

# 1. Contenitore principale (più alto per evitare sovrapposizioni)
main_rect = patches.FancyBboxPatch((2.5, 1.5), 6.5, 6.7, boxstyle="round,pad=0.1",
                                     linewidth=1.5, edgecolor='black', facecolor='none', zorder=1)
ax.add_patch(main_rect)

# 2. Ingressi (Etichette e Frecce)
ax.text(1, 4.5, r'Stato nascosto $\mathbf{H}_{t-1}$', ha='right', va='center', fontsize=11, zorder=4)
# Spostata appena sopra il bordo basso
ax.text(3.1, 1.8, r'Input $\mathbf{X}_t$', ha='center', va='center', fontsize=11, zorder=4)

# 3. Linee di connessione principali
ax.plot([1.1, 8.5], [4.5, 4.5], color='black', lw=1.5, zorder=2)
# La freccia curva parte da sopra la scritta e si immette nella linea orizzontale
draw_curved_arrow(ax, (3.1, 2.1), (3.1, 4.5), (3.6, 4.5))

# 4. Blocchi Sigmoidei e Uscite dei Gate (spaziati orizzontalmente, testo su 3 righe)
# Forget gate
fg_x = 3.8
draw_sigmoid_block(ax, fg_x, 5.5, label='σ')
draw_arrow(ax, (fg_x, 4.5), (fg_x, 5.1))
draw_arrow(ax, (fg_x, 5.9), (fg_x, 6.6), label='Gate di\ndimenticanza\n' + r'$\mathbf{F}_t$', fontweight='normal')

# Input gate
ig_x = 5.6
draw_sigmoid_block(ax, ig_x, 5.5, label='σ')
draw_arrow(ax, (ig_x, 4.5), (ig_x, 5.1))
draw_arrow(ax, (ig_x, 5.9), (ig_x, 6.6), label='Gate di\ningresso\n' + r'$\mathbf{I}_t$', fontweight='normal')

# Output gate
og_x = 7.6
draw_sigmoid_block(ax, og_x, 5.5, label='σ')
draw_arrow(ax, (og_x, 4.5), (og_x, 5.1))
draw_arrow(ax, (og_x, 5.9), (og_x, 6.6), label='Gate di\nuscita\n' + r'$\mathbf{O}_t$', fontweight='normal')

# --- Disegno della Legenda ---
leg_y = 0.2

draw_sigmoid_block(ax, 2, leg_y, size=0.4, label='σ')
ax.text(2.4, leg_y, 'Strato FC con\nfunzione di attivazione', ha='left', va='center', fontsize=10, zorder=4)

leg_copy_x = 5
ax.annotate("", xy=(leg_copy_x, leg_y + 0.3), xytext=(leg_copy_x - 0.3, leg_y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", lw=1.5, color='black', zorder=2))
ax.plot([leg_copy_x - 0.3, leg_copy_x + 0.3], [leg_y, leg_y], color='black', lw=1.5, zorder=2)
ax.text(leg_copy_x + 0.4, leg_y, 'Copia', ha='left', va='center', fontsize=10, zorder=4)

leg_concat_x = 7.5
ax.annotate("", xy=(leg_concat_x, leg_y + 0.2), xytext=(leg_concat_x - 0.4, leg_y + 0.2),
            arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.3", lw=1.5, color='black', zorder=2))
ax.annotate("", xy=(leg_concat_x, leg_y), xytext=(leg_concat_x - 0.4, leg_y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.3", lw=1.5, color='black', zorder=2))
ax.text(leg_concat_x + 0.2, leg_y + 0.1, 'Concatenazione', ha='left', va='center', fontsize=10, zorder=4)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'lstm_diagram.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
