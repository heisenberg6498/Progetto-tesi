import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_activation_block(ax, x, y, size=0.5, label='σ', facecolor='#80cfff'):
    """Disegna un blocco rettangolare con un'etichetta al centro (es. σ o tanh)."""
    # Crea il rettangolo
    rect = patches.Rectangle((x - size/2, y - size/2), size, size,
                             linewidth=1.5, edgecolor='black', facecolor=facecolor, zorder=3)
    ax.add_patch(rect)
    # Aggiunge il testo al centro
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

def draw_arrow(ax, start, end, label=None, text_offset=0.2, fontweight='normal', color='black'):
    """Disegna una freccia semplice con un'etichetta opzionale in alto."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
                                lw=1.5, color=color, zorder=2))
    if label:
        ax.text(end[0], end[1] + text_offset, label, ha='center', va='bottom',
                fontsize=11, fontweight=fontweight, color=color, zorder=4)

def draw_curved_arrow(ax, start, mid, end, color='black'):
    """Disegna una freccia curva verso l'alto (usata per l'ingresso X_t)."""
    verts = [
        start,      # Punto di partenza
        mid,        # Punto di controllo della curva
        end         # Punto di arrivo
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=1.5, zorder=2)
    ax.add_patch(patch)
    # Aggiunge la punta della freccia manualmente
    ax.plot(end[0], end[1], marker=(3, 0, 10), markersize=10, color=color, zorder=2)

# --- Configurazione del grafico ---
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
# Rimuove gli assi e la griglia
ax.set_axis_off()
ax.set_xlim(-1.2, 11)
ax.set_ylim(-1.5, 8)

# --- Disegno del diagramma principale ---

# 1. Contenitore principale arrotondato
main_rect = patches.FancyBboxPatch((2.2, 1.8), 7.5, 6.0, boxstyle="round,pad=0.1",
                                     linewidth=1.5, edgecolor='black', facecolor='none', zorder=1)
ax.add_patch(main_rect)

# 2. Ingressi (Etichette e Frecce)
# Stato nascosto H_{t-1}
ax.text(0.7, 4.5, r'Stato nascosto $\mathbf{H}_{t-1}$', ha='right', va='center', fontsize=11, zorder=4)
# Input X_t
ax.text(2.6, 1.3, r'Input $\mathbf{X}_t$', ha='right', va='center', fontsize=11, zorder=4)

# 3. Linee di connessione principali
# Linea orizzontale da H_{t-1}
ax.plot([0.8, 8.8], [4.5, 4.5], color='black', lw=1.5, zorder=2)

# Freccia curva da X_t che si innesta alla base dei flussi
draw_curved_arrow(ax, (2.5, 1.4), (2.6, 4.5), (3.3, 4.5))

# 4. Blocchi di attivazione e Uscite dei Gate
# Forget gate (Gate di dimenticanza)
draw_activation_block(ax, 3.6, 5.5, label='σ', facecolor='#80cfff')
draw_arrow(ax, (3.6, 4.5), (3.6, 5.1)) # Ingresso
draw_arrow(ax, (3.6, 5.9), (3.6, 6.7), label=r'Gate di' + '\n' + r'dimenticanza $\mathbf{F}_t$', fontweight='normal')

# Input gate (Gate di ingresso)
draw_activation_block(ax, 4.9, 5.5, label='σ', facecolor='#80cfff')
draw_arrow(ax, (4.9, 4.5), (4.9, 5.1)) # Ingresso
draw_arrow(ax, (4.9, 5.9), (4.9, 6.7), label=r'Gate di' + '\n' + r'ingresso $\mathbf{I}_t$', fontweight='normal')

# Input node (Nodo di ingresso / Candidato dello stato della cella)
draw_activation_block(ax, 6.4, 5.5, label='tanh', facecolor='#4da6ff', size=0.55) # Dimensione blocco leggermente aumentata per far stare la scritta
draw_arrow(ax, (6.4, 4.5), (6.4, 5.1)) # Ingresso
draw_arrow(ax, (6.4, 5.9), (6.4, 6.7), label=r'Nodo di' + '\n' + r'ingresso $\tilde{\mathbf{C}}_t$', fontweight='normal')

# Output gate (Gate di uscita)
draw_activation_block(ax, 7.9, 5.5, label='σ', facecolor='#80cfff')
draw_arrow(ax, (7.9, 4.5), (7.9, 5.1)) # Ingresso
draw_arrow(ax, (7.9, 5.9), (7.9, 6.7), label=r'Gate di' + '\n' + r'uscita $\mathbf{O}_t$', fontweight='normal')


# --- Disegno della Legenda ---
# Posizione y della legenda
leg_y = 0.5

# 1. Blocco di attivazione (FC)
draw_activation_block(ax, 1.2, leg_y, size=0.4, label='σ', facecolor='#80cfff')
ax.text(1.6, leg_y, 'Strato FC con\nfunzione di attivazione', ha='left', va='center', fontsize=10, zorder=4)

# 2. Simbolo di copia
leg_copy_x = 4.8
ax.annotate("", xy=(leg_copy_x, leg_y + 0.3), xytext=(leg_copy_x - 0.3, leg_y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.4", lw=1.5, color='black', zorder=2))
ax.plot([leg_copy_x - 0.3, leg_copy_x + 0.3], [leg_y, leg_y], color='black', lw=1.5, zorder=2)
ax.text(leg_copy_x + 0.4, leg_y, 'Copia', ha='left', va='center', fontsize=10, zorder=4)

# 3. Simbolo di concatenazione
leg_concat_x = 7.1
# Due frecce orizzontali adiacenti
ax.annotate("", xy=(leg_concat_x, leg_y + 0.2), xytext=(leg_concat_x - 0.4, leg_y + 0.2),
            arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.3", lw=1.5, color='black', zorder=2))
ax.annotate("", xy=(leg_concat_x, leg_y), xytext=(leg_concat_x - 0.4, leg_y),
            arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.3", lw=1.5, color='black', zorder=2))
ax.plot([leg_concat_x - 0.4, leg_concat_x - 0.4], [leg_y, leg_y + 0.2], color='black', lw=1.5, zorder=2)
ax.text(leg_concat_x + 0.25, leg_y + 0.1, 'Concatena', ha='left', va='center', fontsize=10, zorder=4)


# --- Didascalia Inferiore ---
# (Rossa rimossa per richiesta)

# Salvataggio e visualizzazione
plt.tight_layout()
plt.savefig('calcolo_nodo_ingresso_lstm.png', dpi=300, bbox_inches='tight')
# plt.show() rimosso per permettere l'esecuzione senza blocchi in background
