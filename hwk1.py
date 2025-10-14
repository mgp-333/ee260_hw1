#### Miguel Gutierrez
#### HW 260


#import necessary libraries
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze_tiff(filename):
    """
    Load a TIFF file and analyze its properties
    """
    # Load the image
    image = io.imread(filename) ### RETURNS: img_array ndarray The different color bands/channels are stored in the third dimension, such that a gray-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.


    
    # Get basic properties
    height, width = image.shape
    dtype = image.dtype
    
    # Determine bits per pixel
    if dtype == np.uint8:
        bits_per_pixel = 8
    elif dtype == np.uint16:
        bits_per_pixel = 16
    elif dtype == np.uint32:
        bits_per_pixel = 32
    else:
        bits_per_pixel = "Unknown"
    
    # Print report
    print("=== TIFF Image Analysis ===")
    print(f"Filename: {filename}")
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Data type: {dtype}")
    print(f"Bits per pixel: {bits_per_pixel}")
    print(f"Total pixels: {width * height:,}")
    
    # Convert to float64
    image_float = image.astype('float64')
    print(f"Converted to: {image_float.dtype}")
    
    return image_float




### LINEARIZATION


def linearize_and_report(image_float, black_level, white_level):
    """
    Linearize image and report statistics
    """
    print("=== Linearization Step ===")
    print(f"Black level: {black_level}")
    print(f"White level: {white_level}")
    print(f"Input image range: [{np.min(image_float):.1f}, {np.max(image_float):.1f}]")
    
    # Apply linear transformation
    image_linear = (image_float - black_level) / (white_level - black_level)
    
    print(f"After transformation range: [{np.min(image_linear):.6f}, {np.max(image_linear):.6f}]")
    
    # Clip values
    image_linear = np.clip(image_linear, 0, 1)
    
    print(f"After clipping range: [{np.min(image_linear):.6f}, {np.max(image_linear):.6f}]")
    
    # Count clipped pixels
    below_zero = np.sum(image_linear == 0)
    above_one = np.sum(image_linear == 1)
    total_pixels = image_linear.size
    
    print(f"Pixels clipped to 0 (black): {below_zero} ({below_zero/total_pixels*100:.2f}%)")
    print(f"Pixels clipped to 1 (saturated): {above_one} ({above_one/total_pixels*100:.2f}%)")
    
    return image_linear



def plot_linearization_comparison(image_float, black_level, white_level):
    """
    Show the effect of different black levels
    """
    # Your actual linearization
    image_linear_actual = (image_float - black_level) / (white_level - black_level)
    image_linear_actual = np.clip(image_linear_actual, 0, 1)
    
    # What it would be if black=0
    image_linear_if_zero_black = image_float / white_level
    image_linear_if_zero_black = np.clip(image_linear_if_zero_black, 0, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot both histograms
    ax1.hist(image_linear_actual.flatten(), bins=100, alpha=0.7, label=f'Black={black_level}')
    ax1.hist(image_linear_if_zero_black.flatten(), bins=100, alpha=0.7, label='Black=0')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Linearized Image Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot difference
    difference = image_linear_actual - image_linear_if_zero_black
    ax2.hist(difference.flatten(), bins=100)
    ax2.set_xlabel('Difference')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Difference Between Methods')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return image_linear_actual, image_linear_if_zero_black

# Only run this if you want to compare
# actual, zero_black = plot_linearization_comparison(image_float, black_level, white_level)
def inspect_array(arr, name="array"):
    """
    Comprehensive array inspection function
    """
    print(f"\n=== Inspecting {name} ===")
    
    # Basic properties
    print(f"Type: {type(arr)}")
    print(f"Data type (dtype): {arr.dtype}")
    print(f"Shape: {arr.shape}")
    print(f"Dimensions: {arr.ndim}D")
    print(f"Total elements: {arr.size:,}")
    print(f"Memory usage: {arr.nbytes / (1024**2):.2f} MB")
    
    # Value statistics
    print(f"Min value: {np.min(arr):.6f}")
    print(f"Max value: {np.max(arr):.6f}")
    print(f"Mean value: {np.mean(arr):.6f}")
    print(f"Standard deviation: {np.std(arr):.6f}")
    
    # Check for special values
    print(f"Zeros: {np.sum(arr == 0):,} ({np.sum(arr == 0)/arr.size*100:.2f}%)")
    print(f"Ones: {np.sum(arr == 1):,} ({np.sum(arr == 1)/arr.size*100:.2f}%)")
    print(f"NaN values: {np.sum(np.isnan(arr)):,}")
    print(f"Inf values: {np.sum(np.isinf(arr)):,}")

image_float = load_and_analyze_tiff('./data/Thayer.tiff')
black_level = 0
white_level =  16383
image_linear = linearize_and_report(image_float, black_level, white_level)
inspect_array(image_linear)
#actual, zero_black = plot_linearization_comparison(image_float, black_level, white_level)
block = image_linear[0:2, 0:2]
print(block)




'''
****From assignment pdf per Professor 



import matplotlib.pyplot as plt
from skimage import io
# read two images from current directory
im1 = io.imread(‘image1.tiff’)
im2 = io.imread(‘image2.tiff’)
# display both images in a 1x2 grid
fig = plt.figure() # create a new figure
fig.add_subplot(1, 2, 1) # draw first image
plt.imshow(im1)
fig.add_subplot(1, 2, 2) # draw second image
plt.imshow(im2)
plt.savefig(’output.png’) # saves current figure as a PNG file
plt.show() # displays figure

'''
