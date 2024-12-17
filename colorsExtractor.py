import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import webcolors

def extract_dominant_colors(image, num_colors=2):
    """
    Extract dominant colors from an image using KMeans clustering.
    
    :param image: Input image (numpy array).
    :param num_colors: Number of dominant colors to extract.
    :return: Tuple of (dominant_colors, clustered_image).
    """
    image = cv2.resize(image,(120,120))
    reshaped_image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(reshaped_image)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    clustered_image = dominant_colors[labels].reshape(image.shape)
    return dominant_colors, clustered_image

def rgb_to_color_name(rgb_color):
    """
    Convert an RGB color to the closest web color name.
    
    :param rgb_color: Tuple of (R, G, B).
    :return: Closest color name.
    """
    # Convert the RGB values to a tuple
    rgb_tuple = tuple(rgb_color)
    
    try:
        # Try to find the exact color name
        color_name = webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        # If no exact match, find the closest color
        color_name = closest_color(rgb_tuple)
    
    return color_name

def closest_color(requested_color):
    """
    Find the closest color name to a given RGB value.
    
    :param requested_color: RGB tuple to find the closest color for.
    :return: Name of the closest color.
    """
    min_colors = {}
    
    for name in webcolors.names("css3"):
        print(name)
        r_c, g_c, b_c = webcolors.hex_to_rgb( webcolors.name_to_hex(name) )
        distance = ((r_c - requested_color[0]) ** 2 +
                    (g_c - requested_color[1]) ** 2 +
                    (b_c - requested_color[2]) ** 2)
        min_colors[distance] =  name #webcolors.name_to_hex(name)
    return min_colors[min(min_colors.keys())]

def plot_images_and_colors(image, clustered_image, dominant_colors):
    """
    Plot the original image, clustered image, and dominant colors with names.
    
    :param image: Original input image (numpy array).
    :param clustered_image: Image with pixels replaced by their cluster centers.
    :param dominant_colors: List of dominant colors.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3 + len(dominant_colors), figsize=(18, 6))
    
    # Plot the original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot the clustered image
    axes[1].imshow(cv2.cvtColor(clustered_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    axes[1].set_title("Clustered Image")
    axes[1].axis("off")
    
    # Plot each dominant color as a square with its name
    for i, color in enumerate(dominant_colors):
        color_name = rgb_to_color_name(color)
        axes[2 + i].imshow([[color / 255.0]])
        axes[2 + i].set_title(f"Color {i + 1}: {color_name}")
        axes[2 + i].axis("off")
    
    plt.tight_layout()
    plt.show()

# Load the image
image_path = "test_images/1.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Get the two most dominant colors and clustered image
dominant_colors, clustered_image = extract_dominant_colors(image, num_colors=5)

# Plot the original image, clustered image, and dominant colors with names
plot_images_and_colors(image, clustered_image, dominant_colors)
