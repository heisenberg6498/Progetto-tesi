import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Configurazione figura ad alta risoluzione
fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)

r = 0.048  # Raggio dei nodi circolari
y_top = 0.72  # Quota stati nascosti h
y_bot = 0.45  # Quota input x

# -------------------------------------------------------------
# PARTE SINISTRA: Circuito compatto con ritardo temporale
# -------------------------------------------------------------
x_left = 0.16

# Nodi x e h
circle_h = patches.Circle(
    (x_left, y_top),
    r,
    edgecolor="black",
    facecolor="white",
    lw=1.3,
    zorder=3,
)
circle_x = patches.Circle(
    (x_left, y_bot),
    r,
    edgecolor="black",
    facecolor="white",
    lw=1.3,
    zorder=3,
)
ax.add_patch(circle_h)
ax.add_patch(circle_x)

ax.text(
    x_left,
    y_top,
    r"$\boldsymbol{h}$",
    fontsize=13,
    ha="center",
    va="center",
    family="serif",
    zorder=4,
)
ax.text(
    x_left,
    y_bot,
    r"$\boldsymbol{x}$",
    fontsize=13,
    ha="center",
    va="center",
    family="serif",
    zorder=4,
)

# Freccia verticale da x a h con etichetta f
ax.annotate(
    "",
    xy=(x_left, y_top - r),
    xytext=(x_left, y_bot + r),
    arrowprops=dict(
        arrowstyle="-|>", lw=1.2, color="black", mutation_scale=12
    ),
)
ax.text(
    x_left + 0.022,
    (y_top + y_bot) / 2,
    r"$f$",
    fontsize=12,
    va="center",
    family="serif",
)

# Loop di retroazione ricorsivo che torna sul nodo stesso
ax.annotate(
    "",
    xy=(x_left + r * np.cos(-np.pi / 4), y_top + r * np.sin(-np.pi / 4)),
    xytext=(x_left + r * np.cos(np.pi / 4), y_top + r * np.sin(np.pi / 4)),
    arrowprops=dict(
        arrowstyle="-|>", lw=1.2, color="black",
        connectionstyle="arc3,rad=-2.2", mutation_scale=12
    ),
)

# Freccia di transizione
ax.annotate(
    "",
    xy=(0.30, 0.65),
    xytext=(0.23, 0.65),
    arrowprops=dict(
        arrowstyle="-|>", lw=1.6, color="black", mutation_scale=14
    ),
)

# -------------------------------------------------------------
# PARTE DESTRA: Grafo computazionale srotolato (Unfolded)
# -------------------------------------------------------------
x_unfold = [0.35, 0.48, 0.61, 0.74, 0.87]
labels_h = [
    r"$\boldsymbol{h}^{(\dots)}$",
    r"$\boldsymbol{h}^{(t-1)}$",
    r"$\boldsymbol{h}^{(t)}$",
    r"$\boldsymbol{h}^{(t+1)}$",
    r"$\boldsymbol{h}^{(\dots)}$",
]
is_dashed = [True, False, False, False, True]

# Connessioni orizzontali tra gli stati h
for i in range(len(x_unfold) - 1):
  xs = x_unfold[i] + r
  xe = x_unfold[i + 1] - r
  dashed_edge = is_dashed[i] or is_dashed[i + 1]
  ls = "--" if dashed_edge else "-"

  ax.annotate(
      "",
      xy=(xe, y_top),
      xytext=(xs, y_top),
      arrowprops=dict(
          arrowstyle="-|>",
          lw=1.2,
          color="black",
          linestyle=ls,
          mutation_scale=12,
      ),
  )
  ax.text(
      (xs + xe) / 2,
      y_top - 0.075,
      r"$f$",
      fontsize=11.5,
      ha="center",
      va="top",
      family="serif",
  )

# Nodi orizzontali h
for i, (x_pos, lbl, dash) in enumerate(zip(x_unfold, labels_h, is_dashed)):
  ls = "--" if dash else "-"
  c = patches.Circle(
      (x_pos, y_top),
      r,
      edgecolor="black",
      facecolor="white",
      lw=1.3,
      linestyle=ls,
      zorder=3,
  )
  ax.add_patch(c)
  ax.text(
      x_pos,
      y_top,
      lbl,
      fontsize=9.5 if dash else 10.5,
      ha="center",
      va="center",
      family="serif",
      zorder=4,
  )

# Ingressi inferiori x e connessioni verticali verso h
x_inputs = [0.48, 0.61, 0.74]
labels_x = [
    r"$\boldsymbol{x}^{(t-1)}$",
    r"$\boldsymbol{x}^{(t)}$",
    r"$\boldsymbol{x}^{(t+1)}$",
]

for x_pos, lbl in zip(x_inputs, labels_x):
  c_in = patches.Circle(
      (x_pos, y_bot),
      r,
      edgecolor="black",
      facecolor="white",
      lw=1.3,
      zorder=3,
  )
  ax.add_patch(c_in)
  ax.text(
      x_pos,
      y_bot,
      lbl,
      fontsize=10.5,
      ha="center",
      va="center",
      family="serif",
      zorder=4,
  )

  # Freccia verticale da x a h
  ax.annotate(
      "",
      xy=(x_pos, y_top - r),
      xytext=(x_pos, y_bot + r),
      arrowprops=dict(
          arrowstyle="-|>", lw=1.2, color="black", mutation_scale=12
      ),
  )

# Impostazioni assi e margini
ax.set_xlim(0.08, 0.94)
ax.set_ylim(0.16, 0.88)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig("rnn_unfolded_node.png", dpi=300, bbox_inches="tight")
# plt.show() rimosso
