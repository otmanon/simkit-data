# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Parameters
w = 400  # grid width (off by 1 due to triangle quad grid)
h = 20  # grid height
num_colors = 3
cmap_name = 'tab20b'
sat = False

# Create colormap
cmap = plt.get_cmap(cmap_name, num_colors)
if sat:
    cmap = cmap(np.linspace(0, 1, num_colors))**1.5  # Apply saturation adjustment

# Convert the colormap to a list of RGB tuples
cmap_rgb = (cmap(np.linspace(0, 1, num_colors)) * 255).astype(int)  # Normalize to 0-255

# Create a new image with Pillow
img = Image.new("RGB", (w, h), "white")  # RGB mode, white background

# Create the pixel map
pixels = img.load()

# Create regular grid of values (this will be the color index grid)
X = np.linspace(0, w, w)
Y = np.linspace(0, h, h)
X_grid, Y_grid = np.meshgrid(X, Y)

# Calculate barycenters and assign colors based on the X-coordinate
# This is similar to the previous method, but now we're working with a grid of indices.
barycenters = X_grid.flatten()
I = np.floor(barycenters / (w / num_colors)).astype(int)
I = np.clip(I, 0, num_colors - 1)

# Write the color values directly to the image's pixels
for i in range(w):
    for j in range(h):
        # Get the corresponding color index
        color_idx = I[i + j * w]
        # Set the pixel color (RGB tuple)
        pixels[i, j] = tuple(cmap_rgb[color_idx])

# Save the image
output_path = f"./data/colormaps/{cmap_name}_{num_colors}.png"
img.save(output_path)

print(f"Output saved to {output_path}")