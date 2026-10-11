import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
r = 0.048
x_left, y_top = 0.5, 0.5
circle_h = patches.Circle((x_left, y_top), r, edgecolor="black", facecolor="white", lw=1.3, zorder=3)
ax.add_patch(circle_h)

# loop with arc3
# rad controls the curvature.
ax.annotate(
    "",
    xy=(x_left + r*np.cos(-np.pi/4), y_top + r*np.sin(-np.pi/4)),
    xytext=(x_left + r*np.cos(np.pi/4), y_top + r*np.sin(np.pi/4)),
    arrowprops=dict(
        arrowstyle="-|>", lw=1.2, color="red",
        connectionstyle="arc3,rad=-2.0", mutation_scale=12
    ),
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.savefig("test_loop2.png")
