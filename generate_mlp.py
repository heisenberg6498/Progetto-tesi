import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

radius = 0.055

# Coordinate dei nodi per ogni strato (più distanziate)
x_in = 0.10
y_inputs = np.linspace(0.8, 0.2, 5)

x_l1 = 0.38
y_l1 = [0.75, 0.50, 0.25]

x_l2 = 0.66
y_l2 = [0.88, 0.63, 0.37, 0.12]

x_out = 0.94
y_out = [0.50]

def connect(p1, p2, ax, r1=0.0, r2=radius, is_arrow=True):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  dist = np.sqrt(dx**2 + dy**2)
  start_x = p1[0] + r1 * (dx / dist)
  start_y = p1[1] + r1 * (dy / dist)
  end_x = p2[0] - r2 * (dx / dist)
  end_y = p2[1] - r2 * (dy / dist)

  arrowstyle = '->' if is_arrow else '-'
  ax.annotate(
      '',
      xy=(end_x, end_y),
      xytext=(start_x, start_y),
      arrowprops=dict(
          arrowstyle=arrowstyle, lw=0.9, color='black', mutation_scale=10
      ),
  )

for y_i in y_inputs:
  for y_j in y_l1:
    connect((x_in, y_i), (x_l1, y_j), ax, r1=0.0, r2=radius, is_arrow=True)

for y_i in y_l1:
  for y_j in y_l2:
    connect((x_l1, y_i), (x_l2, y_j), ax, r1=radius, r2=radius, is_arrow=True)

for y_i in y_l2:
  connect(
      (x_l2, y_i), (x_out, y_out[0]), ax, r1=radius, r2=radius, is_arrow=True
  )

for y in y_l1:
  ax.add_patch(
      patches.Circle((x_l1, y), radius, edgecolor='black', facecolor='white', lw=1.1, zorder=3)
  )

for y in y_l2:
  ax.add_patch(
      patches.Circle((x_l2, y), radius, edgecolor='black', facecolor='white', lw=1.1, zorder=3)
  )

ax.add_patch(
    patches.Circle((x_out, y_out[0]), radius, edgecolor='black', facecolor='white', lw=1.1, zorder=3)
)

ax.annotate(
    '',
    xy=(1.09, 0.50),
    xytext=(x_out + radius, 0.50),
    arrowprops=dict(
        arrowstyle='->', lw=0.9, color='black', mutation_scale=10
    ),
)

ax.text(
    0.0,
    0.50,
    'inputs',
    fontsize=13,
    ha='center',
    va='center',
    family='serif',
)
ax.text(
    1.11,
    0.50,
    'output',
    fontsize=13,
    ha='left',
    va='center',
    family='serif',
)

ax.set_xlim(-0.05, 1.20)
ax.set_ylim(0.0, 1.0)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'multilayer_perceptron.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Image saved to {output_path}")
