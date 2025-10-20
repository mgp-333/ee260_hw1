#### Miguel Gutierrez
#### HW 260


#import necessary libraries
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RectBivariateSpline ## need this one for de-mosaicing per q1



'''
RAW IMAGE CONVERSION PYTHON INITIALS
take numpy2d array of unsigned integers and convert into double precision array

'''


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




'''
LINEARIZATION: 
-linearize to account for black level/dark noise offset by:
1. make all values lower than the <black> value --> black.
2. make all pixels with values above <white> --> white.
3. convert to linear array with values btween [0,1] with black maps to 0 and white maps to 1 
4. clip values greater than >1 to 1
5. round up /clip negative values (< 1) to 0.


'''


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

'''
IDENTIFY CORRECT BAYER PATTERN



'''
def image_from_odd_pixels(image_array):
    """
    Create a new image from only the odd-row, odd-column pixels of the input image.
    
    Parameters:
    -----------
    image_array : numpy array
        Input image (H x W x C for color, or H x W for grayscale)
    
    Returns:
    --------
    numpy array 
        New image containing only odd-row, odd-column pixels
    """
    # Slicing: start from index 1 (second element), take every 2nd element
    # This corresponds to "odd indices" because array indexing starts at 0.
    return image_array[1::2, 1::2]
def image_from_even_col_odd_row(image_array):
    """
    Create a new image from only the odd-row, even-column pixels of the input image.
    
    Parameters:
    -----------
    image_array : numpy array
        Input image (H x W x C for color, or H x W for grayscale)
    
    Returns:
    --------
    numpy array 
        New image containing only odd-row, even-column pixels
    """
    # Choose rows starting from index 1 (odd indices) and every 2nd one
    # Choose columns starting from index 0 (even indices) and every 2nd one
    return image_array[1::2, ::2]



def image_from_odd_col_even_row(image_array):
    """
    Create a new image from only the even-row, odd-column pixels of the input image.
    
    Parameters:
    -----------
    image_array : numpy array
        Input image (H x W x C for color, or H x W for grayscale)
    
    Returns:
    --------
    numpy array 
        New image containing only even-row, odd-column pixels
    """
    # Slicing pattern explanation:
    # Rows [::2]  → even rows (0, 2, 4, ...)
    # Cols [1::2] → odd columns (1, 3, 5, ...)
    return image_array[::2, 1::2]

def image_from_even_pixels(image_array):
    """
    Create a new image from only the even-row, even-column pixels of the input image.
    
    Parameters:
    -----------
    image_array : numpy array
        Input image (H x W x C for color, or H x W for grayscale)
    
    Returns:
    --------
    numpy array 
        New image containing only even-row, even-column pixels
    """
    # Slicing pattern: start at 0 (first element), take every second element
    # Rows -> even indices (0, 2, 4, ...)
    # Columns -> even indices (0, 2, 4, ...)
    return image_array[::2, ::2]






'''
WHITE BALANCING
For the White Balancing Portion, I used the slides from class lecture and the following sources:
Reference: https://mattmaulion.medium.com/white-balancing-an-enhancement-technique-in-image-processing-8dd773c69f6
-We needed to implement the white world and grey world white balancing algorithms.
-White World: searches for the lightest patch to use as a white reference similar to how the human visual system does
-Grey World: assumes image is grey



'''


def identify_bayer_pattern_auto(image_linear):
    """
    Automatically identify the correct Bayer pattern by testing all four
    and selecting the one with the most balanced channels
    """
    patterns = ['RGGB', 'GRBG', 'BGGR', 'GBRG']
    pattern_scores = {}
    
    print("=== Automatically Identifying Bayer Pattern ===")
    
    for pattern in patterns:
        # Extract channels based on pattern
        if pattern == 'RGGB':
            red = image_linear[0::2, 0::2]
            green1 = image_linear[0::2, 1::2]
            green2 = image_linear[1::2, 0::2]
            blue = image_linear[1::2, 1::2]
        elif pattern == 'GRBG':
            green1 = image_linear[0::2, 0::2]
            red = image_linear[0::2, 1::2]
            blue = image_linear[1::2, 0::2]
            green2 = image_linear[1::2, 1::2]
        elif pattern == 'BGGR':
            blue = image_linear[0::2, 0::2]
            green1 = image_linear[0::2, 1::2]
            green2 = image_linear[1::2, 0::2]
            red = image_linear[1::2, 1::2]
        elif pattern == 'GBRG':
            green1 = image_linear[0::2, 0::2]
            blue = image_linear[0::2, 1::2]
            red = image_linear[1::2, 0::2]
            green2 = image_linear[1::2, 1::2]
        
        # Calculate statistics
        red_mean = np.mean(red)
        green_mean = (np.mean(green1) + np.mean(green2)) / 2
        blue_mean = np.mean(blue)
        
        # Score based on gray world assumption (channels should be balanced)
        # Lower score is better
        green_similarity = np.abs(np.mean(green1) - np.mean(green2))  # Green channels should be similar
        channel_balance = np.std([red_mean, green_mean, blue_mean])   # All channels should be balanced
        
        total_score = green_similarity + channel_balance
        pattern_scores[pattern] = total_score
        
        print(f"{pattern}: Green similarity = {green_similarity:.6f}, "
              f"Channel balance = {channel_balance:.6f}, Total = {total_score:.6f}")
    
    # Find the pattern with the best (lowest) score
    best_pattern = min(pattern_scores, key=pattern_scores.get)
    print(f"\n🎯 Identified Bayer pattern: {best_pattern}")
    
    return best_pattern, pattern_scores

def white_balance_bayer(image_linear, bayer_pattern, method, r_scale=None, g_scale=None, b_scale=None):
    """
    Apply white balancing to Bayer pattern image
    """
    height, width = image_linear.shape
    
    # Extract channels based on Bayer pattern
    if bayer_pattern == 'RGGB':
        red = image_linear[0::2, 0::2]
        green1 = image_linear[0::2, 1::2]
        green2 = image_linear[1::2, 0::2]
        blue = image_linear[1::2, 1::2]
    elif bayer_pattern == 'GRBG':
        green1 = image_linear[0::2, 0::2]
        red = image_linear[0::2, 1::2]
        blue = image_linear[1::2, 0::2]
        green2 = image_linear[1::2, 1::2]
    elif bayer_pattern == 'BGGR':
        blue = image_linear[0::2, 0::2]
        green1 = image_linear[0::2, 1::2]
        green2 = image_linear[1::2, 0::2]
        red = image_linear[1::2, 1::2]
    elif bayer_pattern == 'GBRG':
        green1 = image_linear[0::2, 0::2]
        blue = image_linear[0::2, 1::2]
        red = image_linear[1::2, 0::2]
        green2 = image_linear[1::2, 1::2]
    
    # Combine green channels for statistics
    green_combined = np.concatenate([green1.flatten(), green2.flatten()])
    
    # Calculate scaling factors based on method
    if method == 'gray_world':
        red_mean = np.mean(red)
        green_mean = np.mean(green_combined)
        blue_mean = np.mean(blue)
        
        red_scale = green_mean / red_mean
        blue_scale = green_mean / blue_mean
        green_scale = 1.0
        
    elif method == 'white_world':
        red_max = np.max(red)
        green_max = np.max(green_combined)
        blue_max = np.max(blue)
        
        red_scale = green_max / red_max
        blue_scale = green_max / blue_max
        green_scale = 1.0
        
    elif method == 'camera_preset':
        red_scale = r_scale
        blue_scale = b_scale
        green_scale = g_scale
    
    # Apply white balancing
    red_balanced = red * red_scale
    green1_balanced = green1 * green_scale
    green2_balanced = green2 * green_scale
    blue_balanced = blue * blue_scale
    
    # Reconstruct the white-balanced Bayer image
    image_wb = np.zeros_like(image_linear)
    
    if bayer_pattern == 'RGGB':
        image_wb[0::2, 0::2] = red_balanced
        image_wb[0::2, 1::2] = green1_balanced
        image_wb[1::2, 0::2] = green2_balanced
        image_wb[1::2, 1::2] = blue_balanced
    elif bayer_pattern == 'GRBG':
        image_wb[0::2, 0::2] = green1_balanced
        image_wb[0::2, 1::2] = red_balanced
        image_wb[1::2, 0::2] = blue_balanced
        image_wb[1::2, 1::2] = green2_balanced
    elif bayer_pattern == 'BGGR':
        image_wb[0::2, 0::2] = blue_balanced
        image_wb[0::2, 1::2] = green1_balanced
        image_wb[1::2, 0::2] = green2_balanced
        image_wb[1::2, 1::2] = red_balanced
    elif bayer_pattern == 'GBRG':
        image_wb[0::2, 0::2] = green1_balanced
        image_wb[0::2, 1::2] = blue_balanced
        image_wb[1::2, 0::2] = red_balanced
        image_wb[1::2, 1::2] = green2_balanced
    
    return image_wb, (red_scale, green_scale, blue_scale)

def auto_white_balance_all_patterns(image_linear, r_scale, g_scale, b_scale):
    """
    Automatically identify Bayer pattern and apply all three white balance methods
    """
    print("=== Automatic Bayer Pattern White Balancing ===")
    
    # Step 1: Automatically identify the Bayer pattern
    best_pattern, all_scores = identify_bayer_pattern_auto(image_linear)
    
    # Step 2: Apply all three white balance methods using the identified pattern
    print(f"\n=== Applying White Balance Methods with {best_pattern} Pattern ===")
    
    wb_white_world, scales_white = white_balance_bayer(image_linear, best_pattern, 'white_world')
    wb_gray_world, scales_gray = white_balance_bayer(image_linear, best_pattern, 'gray_world')
    wb_camera, scales_camera = white_balance_bayer(image_linear, best_pattern, 'camera_preset', 
                                                  r_scale, g_scale, b_scale)
    
    # Create wb_results dictionary
    wb_results = {
        'white_world': wb_white_world,
        'gray_world': wb_gray_world,
        'camera_preset': wb_camera,
        'scales': {
            'white_world': scales_white,
            'gray_world': scales_gray,
            'camera_preset': scales_camera
        },
        'bayer_pattern': best_pattern,
        'all_pattern_scores': all_scores
    }
    
    print("✓ Automatic white balancing completed!")
    return wb_results

def visualize_all_bayer_patterns(image_linear, display_scale=10):
    """
    Visualize all four Bayer pattern interpretations for comparison
    """
    patterns = ['RGGB', 'GRBG', 'BGGR', 'GBRG']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, pattern in enumerate(patterns):
        # Extract channels
        if pattern == 'RGGB':
            red = image_linear[0::2, 0::2]
            green1 = image_linear[0::2, 1::2]
            green2 = image_linear[1::2, 0::2]
            blue = image_linear[1::2, 1::2]
        elif pattern == 'GRBG':
            green1 = image_linear[0::2, 0::2]
            red = image_linear[0::2, 1::2]
            blue = image_linear[1::2, 0::2]
            green2 = image_linear[1::2, 1::2]
        elif pattern == 'BGGR':
            blue = image_linear[0::2, 0::2]
            green1 = image_linear[0::2, 1::2]
            green2 = image_linear[1::2, 0::2]
            red = image_linear[1::2, 1::2]
        elif pattern == 'GBRG':
            green1 = image_linear[0::2, 0::2]
            blue = image_linear[0::2, 1::2]
            red = image_linear[1::2, 0::2]
            green2 = image_linear[1::2, 1::2]
        
        # Create simple RGB visualization (just for pattern inspection)
        height, width = red.shape
        rgb_viz = np.zeros((height, width, 3))
        rgb_viz[:, :, 0] = red * display_scale
        rgb_viz[:, :, 1] = (green1 + green2) / 2 * display_scale
        rgb_viz[:, :, 2] = blue * display_scale
        rgb_viz = np.clip(rgb_viz, 0, 1)
        
        axes[i].imshow(rgb_viz)
        axes[i].set_title(f'Pattern: {pattern}')
        axes[i].axis('off')
    
    plt.suptitle('All Bayer Pattern Interpretations (Scaled for Visibility)', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_white_balance_results(wb_results, original_image, display_scale=10):
    """
    Visualize the white balanced Bayer images
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    region = (100, 400, 100, 400)  # y1, y2, x1, x2
    y1, y2, x1, x2 = region
    
    # Original
    axes[0,0].imshow(np.clip(original_image[y1:y2, x1:x2] * display_scale, 0, 1), cmap='gray')
    axes[0,0].set_title('Original Bayer')
    axes[0,0].axis('off')
    
    # White World
    axes[0,1].imshow(np.clip(wb_results['white_world'][y1:y2, x1:x2] * display_scale, 0, 1), cmap='gray')
    axes[0,1].set_title('White World WB')
    axes[0,1].axis('off')
    
    # Gray World
    axes[1,0].imshow(np.clip(wb_results['gray_world'][y1:y2, x1:x2] * display_scale, 0, 1), cmap='gray')
    axes[1,0].set_title('Gray World WB')
    axes[1,0].axis('off')
    
    # Camera Preset
    axes[1,1].imshow(np.clip(wb_results['camera_preset'][y1:y2, x1:x2] * display_scale, 0, 1), cmap='gray')
    axes[1,1].set_title('Camera Preset WB')
    axes[1,1].axis('off')
    
    plt.suptitle(f'White Balanced Bayer Images (Auto-detected Pattern: {wb_results["bayer_pattern"]})', fontsize=14)
    plt.tight_layout()
    plt.show()

def print_white_balance_summary(wb_results):
    """
    Print comprehensive white balance results
    """
    print("\n" + "="*60)
    print("WHITE BALANCE SUMMARY")
    print("="*60)
    
    print(f"\nIdentified Bayer Pattern: {wb_results['bayer_pattern']}")
    
    print("\nPattern Analysis Scores (lower is better):")
    for pattern, score in wb_results['all_pattern_scores'].items():
        marker = " ← BEST" if pattern == wb_results['bayer_pattern'] else ""
        print(f"  {pattern}: {score:.6f}{marker}")
    
    print("\nWhite Balance Scaling Factors:")
    for method, scales in wb_results['scales'].items():
        r, g, b = scales
        print(f"  {method:15}: R = {r:7.4f}, G = {g:7.4f}, B = {b:7.4f}")
    
    print(f"\nOutput Image Shapes:")
    print(f"  Original: {image_linear.shape}")
    print(f"  White World: {wb_results['white_world'].shape}")
    print(f"  Gray World: {wb_results['gray_world'].shape}")
    print(f"  Camera Preset: {wb_results['camera_preset'].shape}")

# Complete automatic pipeline
def complete_auto_white_balance_pipeline(image_linear, r_scale, g_scale, b_scale):
    """
    Complete automatic white balancing pipeline:
    1. Visualize all Bayer patterns
    2. Automatically identify correct pattern
    3. Apply all three white balance methods
    4. Return wb_results
    """
    print("Starting Automatic White Balance Pipeline")
    print(f"Camera preset multipliers: R={r_scale:.4f}, G={g_scale:.4f}, B={b_scale:.4f}")
    
    # Step 1: Show all pattern interpretations
    print("\nStep 1: Visualizing all Bayer pattern interpretations...")
    visualize_all_bayer_patterns(image_linear)
    
    # Step 2: Automatically identify pattern and apply white balancing
    print("\nStep 2: Identifying Bayer pattern and applying white balancing...")
    wb_results = auto_white_balance_all_patterns(image_linear, r_scale, g_scale, b_scale)
    
    # Step 3: Visualize results
    print("\nStep 3: Visualizing white balanced results...")
    visualize_white_balance_results(wb_results, image_linear)
    
    # Step 4: Print summary
    print_white_balance_summary(wb_results)
    
    print("\n✓ Automatic white balance pipeline completed!")
    print("✓ wb_results is ready for demosaicing!")
    
    return wb_results

# Usage - just run this one function:
def run_automatic_white_balancing():
    """
    Run the complete automatic white balancing pipeline
    """
    # Your values
    r_scale = 2.165039
    g_scale = 1.000000
    b_scale = 1.643555
    
    # Run the complete automatic pipeline
    wb_results = complete_auto_white_balance_pipeline(
        image_linear=image_linear,  # Your linearized Bayer image
        r_scale=r_scale,
        g_scale=g_scale,
        b_scale=b_scale
    )
    
    return wb_results

# To use - just call this:
# wb_results = run_automatic_white_balancing()




"""
DEMOSAICING:


"""


def demosaic_image(image_wb, bayer_pattern):
    """
    Demosaic a white-balanced Bayer pattern image using bilinear interpolation
    
    Parameters:
    - image_wb: White balanced Bayer pattern image (2D array)
    - bayer_pattern: One of 'RGGB', 'GRBG', 'BGGR', 'GBRG'
    
    Returns:
    - Demosaiced RGB image (3D array of shape H x W x 3)
    """
    height, width = image_wb.shape
    
    # Create coordinate grids for the full image
    x_full = np.arange(width)
    y_full = np.arange(height)
    
    # Create coordinate grids for the subsampled channels
    x_even = np.arange(0, width, 2)   # Even columns: 0, 2, 4, ...
    x_odd = np.arange(1, width, 2)    # Odd columns: 1, 3, 5, ...
    y_even = np.arange(0, height, 2)  # Even rows: 0, 2, 4, ...
    y_odd = np.arange(1, height, 2)   # Odd rows: 1, 3, 5, ...
    
    # Initialize RGB channels
    red_channel = np.zeros((height, width))
    green_channel = np.zeros((height, width))
    blue_channel = np.zeros((height, width))
    
    # Extract known pixel values based on Bayer pattern
    if bayer_pattern == 'RGGB':
        # Red at (even, even)
        red_known = image_wb[0::2, 0::2]
        # Green at (even, odd) and (odd, even)
        green1_known = image_wb[0::2, 1::2]  # Even rows, odd columns
        green2_known = image_wb[1::2, 0::2]  # Odd rows, even columns
        # Blue at (odd, odd)
        blue_known = image_wb[1::2, 1::2]
        
    elif bayer_pattern == 'GRBG':
        # Red at (even, odd)
        red_known = image_wb[0::2, 1::2]
        # Green at (even, even) and (odd, odd)
        green1_known = image_wb[0::2, 0::2]  # Even rows, even columns
        green2_known = image_wb[1::2, 1::2]  # Odd rows, odd columns
        # Blue at (odd, even)
        blue_known = image_wb[1::2, 0::2]
        
    elif bayer_pattern == 'BGGR':
        # Red at (odd, odd)
        red_known = image_wb[1::2, 1::2]
        # Green at (even, odd) and (odd, even)
        green1_known = image_wb[0::2, 1::2]  # Even rows, odd columns
        green2_known = image_wb[1::2, 0::2]  # Odd rows, even columns
        # Blue at (even, even)
        blue_known = image_wb[0::2, 0::2]
        
    elif bayer_pattern == 'GBRG':
        # Red at (odd, even)
        red_known = image_wb[1::2, 0::2]
        # Green at (even, even) and (odd, odd)
        green1_known = image_wb[0::2, 0::2]  # Even rows, even columns
        green2_known = image_wb[1::2, 1::2]  # Odd rows, odd columns
        # Blue at (even, odd)
        blue_known = image_wb[0::2, 1::2]
    
    # Create interpolation functions for each channel
    # Red channel interpolation
    spline_red = RectBivariateSpline(y_even, x_even, red_known, kx=1, ky=1)
    red_channel = spline_red(y_full, x_full)
    
    # Blue channel interpolation
    spline_blue = RectBivariateSpline(y_odd, x_odd, blue_known, kx=1, ky=1)
    blue_channel = spline_blue(y_full, x_full)
    
    # Green channel interpolation (two separate interpolations then average)
    spline_green1 = RectBivariateSpline(y_even, x_odd, green1_known, kx=1, ky=1)
    green1_full = spline_green1(y_full, x_full)
    
    spline_green2 = RectBivariateSpline(y_odd, x_even, green2_known, kx=1, ky=1)
    green2_full = spline_green2(y_full, x_full)
    
    # Average the two green interpolations
    green_channel = (green1_full + green2_full) / 2
    
    # Stack channels to create RGB image
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    
    # Clip to ensure valid range
    rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image

# Apply demosaicing to all white balanced images
def demosaic_all_images(wb_results, bayer_pattern):
    """
    Apply demosaicing to all white balanced images
    """
    print("=== Applying Demosaicing ===")
    
    demosaiced_images = {}
    
    for method, image_wb in wb_results.items():
        if method == 'scales':  # Skip the scales dictionary
            continue
            
        print(f"Demosaicing {method} white balanced image...")
        rgb_image = demosaic_image(image_wb, bayer_pattern)
        demosaiced_images[method] = rgb_image
        
        print(f"  Input shape: {image_wb.shape}")
        print(f"  Output shape: {rgb_image.shape}")
    
    return demosaiced_images

# Visualize demosaiced results
def visualize_demosaiced_results(demosaiced_images):
    """
    Visualize the demosaiced RGB images
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Display settings
    display_region = (100, 400, 100, 400)  # Show a 300x300 region
    y1, y2, x1, x2 = display_region
    
    methods = ['gray_world', 'white_world', 'camera_preset']
    titles = ['Gray World', 'White World', 'Camera Preset']
    
    for i, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[i//2, i%2]
        rgb_region = demosaiced_images[method][y1:y2, x1:x2]
        
        # Scale for display (linear images are dark)
        display_image = np.clip(rgb_region * 5, 0, 1)
        
        ax.imshow(display_image)
        ax.set_title(f'{title} Demosaiced')
        ax.axis('off')
    
    # Hide the empty subplot
    axes[1, 1].axis('off')
    
    plt.suptitle('Demosaiced RGB Images (Scaled for Visibility)', fontsize=14)
    plt.tight_layout()
    plt.show()

# Complete demosaicing pipeline
def complete_demosaicing_pipeline(wb_results, bayer_pattern):
    """
    Complete demosaicing for all white balance methods
    """
    print("Starting Demosaicing Pipeline")
    print(f"Bayer Pattern: {bayer_pattern}")
    
    # Apply demosaicing to all white balanced images
    demosaiced_images = demosaic_all_images(wb_results, bayer_pattern)
    
    # Visualize results
    visualize_demosaiced_results(demosaiced_images)
    
    # Print image statistics
    print("\n=== Demosaiced Image Statistics ===")
    for method, rgb_image in demosaiced_images.items():
        print(f"{method:15}:")
        print(f"  Shape: {rgb_image.shape}")
        print(f"  Red range:   [{np.min(rgb_image[:,:,0]):.4f}, {np.max(rgb_image[:,:,0]):.4f}]")
        print(f"  Green range: [{np.min(rgb_image[:,:,1]):.4f}, {np.max(rgb_image[:,:,1]):.4f}]")
        print(f"  Blue range:  [{np.min(rgb_image[:,:,2]):.4f}, {np.max(rgb_image[:,:,2]):.4f}]")
    
    return demosaiced_images





'''
MAIN: 

'''

'''
1 Developing RAW images
'''
image_float = load_and_analyze_tiff('./data/Thayer.tiff')
black_level = 0
white_level =  16383
image_linear = linearize_and_report(image_float, black_level, white_level)
inspect_array(image_linear)
#actual, zero_black = plot_linearization_comparison(image_float, black_level, white_level)
block = image_linear[0:2, 0:2]
print(block)
# Change the coordinates to inspect different blocks
block1 = image_linear[0:2, 0:2]      # Top-left
block2 = image_linear[100:102, 200:202]  # Somewhere in the image
block3 = image_linear[500:502, 800:802]  # Another location
block4 = image_linear[2000:2002, 3000:3002]  # Bottom-right area

print("Top-left block:")
print(block1)
print("\nBlock at (100,200):")
print(block2)
print("\nBlock at (500,800):")
print(block3)
print("\nBlock at (2000,3000):")
print(block4)



'''This was part of exploratory work, but it was insufficient in order to determine the bayer pattern. After consulting with Prof, I realized I needed to stack them. 


# Extract only odd-row, odd-column pixels
odd_img = image_from_odd_pixels(image_linear)
odd_even_img =  image_from_odd_col_even_row(image_linear)
even_odd_img =  image_from_even_col_odd_row(image_linear)
even_img = image_from_even_pixels(image_linear)

#extract 

# Display result
import matplotlib.pyplot as plt

# Assuming you have these images computed already:
# image_linear (original)
# odd_odd_img (odd rows & odd cols)
# even_odd_img (even rows & odd cols)
# even_even_img (even rows & even cols)

plt.figure(figsize=(12, 8))

# Original
plt.subplot(3, 2, 1)
plt.imshow(image_linear)
plt.title('Original')
plt.axis('off')

# Odd Rows & Odd Columns
plt.subplot(3, 2, 2)
plt.imshow(odd_img)
plt.title('Odd Rows & Odd Columns')
plt.axis('off')

# Even Rows & Odd Columns
plt.subplot(3, 2, 3)
plt.imshow(even_odd_img)
plt.title('Even Rows & Odd Columns')
plt.axis('off')

# Even Rows & Odd Columns
plt.subplot(3, 2, 4)
plt.imshow(odd_even_img)
plt.title('Odd Rows & Even Columns')
plt.axis('off')


# Even Rows & Even Columns
plt.subplot(3, 2, 5)
plt.imshow(even_img)
plt.title('Even Rows & Even Columns')
plt.axis('off')


'''

#### now to stack in order to see better

# create three sub-images of im as shown in figure below(reference material is figure 4 from the homework)
im1 = image_linear[0::2, 0::2]
im2 = image_linear[0::2, 1::2]
im3 = image_linear[1::2, 0::2]

#was not stacking so needed to ensure proper size fit
print(im1.shape)
print(im2.shape)
print(im3.shape)

min_height = min(im1.shape[0], im2.shape[0], im3.shape[0])
min_width  = min(im1.shape[1], im2.shape[1], im3.shape[1])
im1_cropped = im1[:min_height, :min_width]
im2_cropped = im2[:min_height, :min_width]
im3_cropped = im3[:min_height, :min_width]


# combine the above images into an RGB image, such that im1 is the red,
# im2 is the green, and im3 is the blue channel
im_rgb1 = np.dstack((im1_cropped, im2_cropped, im3_cropped)) # stacked im1 =red, im2 = green, im3 = blue
im_rgb2 = np.dstack((im2_cropped, im1_cropped, im3_cropped))  # stacked im2 =red, im1 = green, im3 = blue
im_rgb3 = np.dstack((im2_cropped, im3_cropped, im2_cropped)) # stacked im2 =red, im3 = green, im1 = blue
im_rgb4 = np.dstack((im3_cropped, im2_cropped, im1_cropped)) # stacked im3 =red, im1 = green, im3 = blue
im_rgb5 = np.dstack((im1_cropped, im3_cropped, im2_cropped)) # stacked im1 =red, im3 = green, im2 = blue


'''
# create a new figure
fig2 = plt.figure(figsize=(25, 5))
# subplot for first image (1 row, 3 columns, position 1)
ax1 = fig2.add_subplot(1, 5, 1)
ax1.imshow(im_rgb1)
ax1.axis('off')  # optional: hide axis ticks
ax1.set_title(' im1 =red, im2 = g, im3 = b')

# subplot for second image (position 2)
ax2 = fig2.add_subplot(1, 5, 2)
ax2.imshow(im_rgb2)
ax2.axis('off')
ax2.set_title(' im2 =red, im1 = g, im3 = b')

# subplot for third image (position 3)
ax3 = fig2.add_subplot(1, 5, 3)
ax3.imshow(im_rgb3)
ax3.axis('off')
ax3.set_title(' im2 =red, im3 = g, im1 = b')

# subplot for fourth image (position 4)
ax3 = fig2.add_subplot(1, 5, 4)
ax3.imshow(im_rgb4)
ax3.axis('off')
ax3.set_title(' im3 =red, im1 = g, im2 = b')

# subplot for fifth image (position 4)
ax3 = fig2.add_subplot(1, 5, 5)
ax3.imshow(im_rgb5)
ax3.axis('off')
ax3.set_title(' im1 =red, im3 = g, im2 = b')

# save and show
plt.savefig('output.png')
plt.show()
'''



### WHITE BALANCING
# Example usage:
# Assuming you have:
# - rgb_image: your demosaiced RGB image (H x W x 3)
# - r_scale, g_scale, b_scale: values from dcraw reconnaissance

# from reconnaissance run value of dCRAW
r_scale = 2.165039  # 
g_scale = 1.000000  # 
b_scale = 1.643555  # 


# Apply WB then compare all three methods
# Apply white balancing
wb_results = complete_white_balance_analysis(image_linear, bayer_pattern, r_scale, g_scale, b_scale)

# Now you can continue with demosaicing, color correction, etc. for each white balanced image
# For example:
# demosaiced_gray = demosaic_image(wb_results['gray_world'], bayer_pattern)
# demosaiced_white = demosaic_image(wb_results['white_world'], bayer_pattern)
# demosaiced_camera = demosaic_image(wb_results['camera_preset'], bayer_pattern)


###DE-MOSAICING

# bayer_pattern = 'RGGB'  # Your identified Bayer pattern

# Apply demosaicing to all white balanced images
demosaiced_images = complete_demosaicing_pipeline(wb_results, bayer_pattern)





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
