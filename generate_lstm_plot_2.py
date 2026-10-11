import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Configurazione figura
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
ax.set_xlim(-0.5, 11.5)
ax.set_ylim(-1.5, 5.5)
ax.axis("off")

# Stili grafici
c_box = "#7ec0ee"  # Azzurro per i blocchi di attivazione
c_op = "#bfe0f7"  # Azzurro chiaro per le operazioni
c_line = "black"

# 1. Contorno principale della cella LSTM
main_box = patches.FancyBboxPatch(
    (2.8, 0.4),
    5.6,
    4.5,
    boxstyle="round,pad=0.2,rounding_size=0.4",
    ec="black",
    fc="white",
    lw=1.5,
)
ax.add_patch(main_box)


# Funzioni di supporto per componenti grafici
def draw_fc_block(x, y, label, text_above, math_above):
    """Disegna un blocco FC con funzione di attivazione."""
    rect = patches.Rectangle(
        (x - 0.35, y - 0.25),
        0.7,
        0.5,
        facecolor=c_box,
        edgecolor="black",
        lw=1.2,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        fontfamily="serif",
        zorder=4,
    )
    ax.text(
        x - 0.5,
        y + 0.55,
        text_above,
        ha="center",
        va="bottom",
        fontsize=10.5,
        linespacing=1.1,
    )
    ax.text(
        x - 0.5,
        y + 0.15,
        math_above,
        ha="center",
        va="bottom",
        fontsize=11,
        fontfamily="serif",
    )


def draw_op(x, y, symbol):
    """Disegna un cerchio operatore elemento per elemento."""
    circle = patches.Circle(
        (x, y), 0.22, facecolor=c_op, edgecolor="black", lw=1.2, zorder=3
    )
    ax.add_patch(circle)
    ax.text(
        x,
        y,
        symbol,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        zorder=4,
    )


# 2. Posizionamento Blocchi e Operatori
# Porte e nodi (Layer FC)
draw_fc_block(
    3.9, 1.6, r"$\sigma$", "Gate di\ndimenticanza", r"$\mathbf{F}_t$"
)
draw_fc_block(5.0, 1.6, r"$\sigma$", "Gate di\ningresso", r"$\mathbf{I}_t$")
draw_fc_block(
    6.3, 1.6, r"$\tanh$", "Candidato\ndi stato", r"$\mathbf{\tilde{C}}_t$"
)
draw_fc_block(7.5, 1.6, r"$\sigma$", "Gate di\nuscita", r"$\mathbf{O}_t$")

# Operatori elemento per elemento
draw_op(3.9, 4.3, r"$\odot$")  # Moltiplicazione Ct-1 * Ft
draw_op(6.3, 2.9, r"$\odot$")  # Moltiplicazione It * C~t
draw_op(6.3, 4.3, "+")  # Somma in C_t

# 3. Flussi e Linee
arrow = dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12)

# Flusso Stato Cella (Ct-1 -> Ct)
ax.annotate("", xy=(3.68, 4.3), xytext=(2.6, 4.3), arrowprops=arrow)
ax.plot([4.12, 6.08], [4.3, 4.3], color=c_line, lw=1.2)
ax.annotate("", xy=(8.8, 4.3), xytext=(6.52, 4.3), arrowprops=arrow)

# Flussi verticali verso gli operatori
ax.annotate("", xy=(3.9, 4.08), xytext=(3.9, 1.85), arrowprops=arrow)
ax.annotate("", xy=(6.3, 4.08), xytext=(6.3, 3.12), arrowprops=arrow)
ax.annotate("", xy=(6.3, 2.68), xytext=(6.3, 1.85), arrowprops=arrow)
ax.annotate("", xy=(7.5, 2.3), xytext=(7.5, 1.85), arrowprops=arrow)

# Percorso Gate di ingresso (I_t) verso la moltiplicazione
ax.annotate(
    "",
    xy=(6.08, 2.9),
    xytext=(5.0, 1.85),
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        lw=1.2,
        mutation_scale=12,
        connectionstyle="angle,angleA=90,angleB=180,rad=5",
    ),
)

# Bus di ingresso inferiore (Concatenazione tra H_{t-1} e X_t)
ax.plot([2.6, 7.5], [0.95, 0.95], color=c_line, lw=1.2)
ax.plot([3.3, 3.3], [-0.1, 0.65], color=c_line, lw=1.2)
ax.plot([3.3, 3.5], [0.65, 0.95], color=c_line, lw=1.2)

# Ramificazioni dal bus ai singoli blocchi
for x in [3.9, 5.0, 6.3, 7.5]:
    ax.annotate("", xy=(x, 1.35), xytext=(x, 0.95), arrowprops=arrow)

# 4. Etichette di Testo Esterne
ax.text(
    2.5,
    4.5,
    "Stato interno\ndella cella",
    ha="right",
    va="center",
    fontsize=11,
)
ax.text(
    2.0,
    4.05,
    r"$\mathbf{C}_{t-1}$",
    ha="right",
    va="center",
    fontsize=12,
    fontfamily="serif",
)

ax.text(9.0, 4.3, r"$\mathbf{C}_t$", ha="left", va="center", fontsize=12)

ax.text(
    2.5,
    1.15,
    "Stato nascosto\nprecedente",
    ha="right",
    va="center",
    fontsize=11,
)
ax.text(
    2.0,
    0.7,
    r"$\mathbf{H}_{t-1}$",
    ha="right",
    va="center",
    fontsize=12,
    fontfamily="serif",
)

ax.text(
    3.3,
    -0.35,
    r"Input $\mathbf{X}_t$",
    ha="center",
    va="top",
    fontsize=11,
    fontfamily="serif",
)

# 5. Legenda inferiore
y_leg = -1.1
# Blocco FC
rect_leg = patches.Rectangle(
    (0.6, y_leg - 0.2), 0.6, 0.4, facecolor=c_box, edgecolor="black", lw=1.2
)
ax.add_patch(rect_leg)
ax.text(0.9, y_leg, r"$\sigma$", ha="center", va="center", fontsize=11)
ax.text(
    1.4,
    y_leg,
    "Strato fully connected\ncon funz. di attivazione",
    ha="left",
    va="center",
    fontsize=10,
)

# Operatore elemento per elemento
circ_leg = patches.Circle(
    (3.9, y_leg), 0.18, facecolor=c_op, edgecolor="black", lw=1.2
)
ax.add_patch(circ_leg)
ax.text(
    4.3,
    y_leg,
    "Operatore elemento\nper elemento",
    ha="left",
    va="center",
    fontsize=10,
)

# Copia
ax.plot([6.2, 6.5], [y_leg, y_leg], color=c_line, lw=1.2)
ax.annotate(
    "",
    xy=(6.5, y_leg + 0.3),
    xytext=(6.35, y_leg),
    arrowprops=dict(
        arrowstyle="-|>",
        color="black",
        lw=1.2,
        mutation_scale=10,
        connectionstyle="arc3,rad=-0.2",
    ),
)
ax.text(6.8, y_leg, "Copia", ha="left", va="center", fontsize=10)

# Concatena
ax.plot([8.0, 8.4], [y_leg, y_leg], color=c_line, lw=1.2)
ax.plot([8.0, 8.2], [y_leg - 0.25, y_leg], color=c_line, lw=1.2)
ax.annotate(
    "",
    xy=(8.7, y_leg),
    xytext=(8.4, y_leg),
    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10),
)
ax.text(9.0, y_leg, "Concatena", ha="left", va="center", fontsize=10)

plt.tight_layout()
plt.savefig('calcolo_stato_cella_lstm.png', dpi=300, bbox_inches='tight')
