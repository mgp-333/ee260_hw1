#### Miguel Gutierrez
#### HW 260


#import necessary libraries
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RectBivariateSpline ## need this one for de-mosaicing per q1
from skimage.color import rgb2gray # need this for brightness adjustment/ gamma encoding
from skimage.io import imsave # NEED this for compression 
import os #save





'''
############################################################################################################################################
############################################################################################################################################

RAW IMAGE CONVERSION PYTHON INITIALS
take numpy2d array of unsigned integers and convert into double precision array

############################################################################################################################################
############################################################################################################################################

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
############################################################################################################################################
############################################################################################################################################

LINEARIZATION: 
-linearize to account for black level/dark noise offset by:
1. make all values lower than the <black> value --> black.
2. make all pixels with values above <white> --> white.
3. convert to linear array with values btween [0,1] with black maps to 0 and white maps to 1 
4. clip values greater than >1 to 1
5. round up /clip negative values (< 1) to 0.

############################################################################################################################################
############################################################################################################################################

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

# Only run if wanna compare otwise ignore!!!!!
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
############################################################################################################################################
############################################################################################################################################

IDENTIFY CORRECT BAYER PATTERN

############################################################################################################################################
############################################################################################################################################

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
############################################################################################################################################
############################################################################################################################################

WHITE BALANCING
For the White Balancing Portion, I used the slides from class lecture and the following sources:
Reference: https://mattmaulion.medium.com/white-balancing-an-enhancement-technique-in-image-processing-8dd773c69f6
-We needed to implement the white world and grey world white balancing algorithms.
-White World: searches for the lightest patch to use as a white reference similar to how the human visual system does
-Grey World: assumes image is grey

############################################################################################################################################
############################################################################################################################################
'''


def white_balance_white_world(bayer_image, pattern='rggb'):
    """
    White World white balancing: assumes the brightest pixel in each channel should be white.
    """
    # extract Bayer channels based on pattern
    R, G1, G2, B = extract_bayer_channels(bayer_image, pattern)

    # find max values per channel
    r_max, g_max, b_max = np.max(R), np.max((G1 + G2) / 2), np.max(B)

    # compute normalization scales
    r_scale, g_scale, b_scale = 1 / r_max, 1 / g_max, 1 / b_max

    # apply scaling individually to each pixel
    balanced = apply_bayer_scaling(bayer_image, r_scale, g_scale, b_scale, pattern)
    return balanced, (r_scale, g_scale, b_scale)


def white_balance_gray_world(bayer_image, pattern='rggb'):
    """
    Gray World white balancing: assumes the average color should be gray.
    """
    R, G1, G2, B = extract_bayer_channels(bayer_image, pattern)

    # mean intensity per channel
    r_mean, g_mean, b_mean = np.mean(R), np.mean((G1 + G2) / 2), np.mean(B)
    overall_mean = (r_mean + g_mean + b_mean) / 3

    # compute scales to equalize channel averages
    r_scale = overall_mean / r_mean
    g_scale = overall_mean / g_mean
    b_scale = overall_mean / b_mean

    balanced = apply_bayer_scaling(bayer_image, r_scale, g_scale, b_scale, pattern)
    return balanced, (r_scale, g_scale, b_scale)


def white_balance_camera_presets(bayer_image, r_scale, g_scale, b_scale, pattern='rggb'):
    """
    White balancing using camera’s dcraw coefficients (r_scale, g_scale, b_scale).
    """
    return apply_bayer_scaling(bayer_image, r_scale, g_scale, b_scale, pattern), (r_scale, g_scale, b_scale)



#needed to be fixed per size
def extract_bayer_channels(bayer_img, pattern='rggb'):
    """
    Extract Bayer pattern channels and ensure all sizes match by cropping if necessary.
    """
    if pattern == 'rggb':
        R = bayer_img[0::2, 0::2]
        G1 = bayer_img[0::2, 1::2]
        G2 = bayer_img[1::2, 0::2]
        B = bayer_img[1::2, 1::2]
    elif pattern == 'bggr':
        B = bayer_img[0::2, 0::2]
        G1 = bayer_img[0::2, 1::2]
        G2 = bayer_img[1::2, 0::2]
        R = bayer_img[1::2, 1::2]
    elif pattern == 'grbg':
        G1 = bayer_img[0::2, 0::2]
        R = bayer_img[0::2, 1::2]
        B = bayer_img[1::2, 0::2]
        G2 = bayer_img[1::2, 1::2]
    elif pattern == 'gbrg':
        G1 = bayer_img[0::2, 0::2]
        B = bayer_img[0::2, 1::2]
        R = bayer_img[1::2, 0::2]
        G2 = bayer_img[1::2, 1::2]
    else:
        raise ValueError("Unsupported Bayer pattern.")

    # Crop all to match smallest height and width
    min_h = min(R.shape[0], G1.shape[0], G2.shape[0], B.shape[0])
    min_w = min(R.shape[1], G1.shape[1], G2.shape[1], B.shape[1])

    R = R[:min_h, :min_w]
    G1 = G1[:min_h, :min_w]
    G2 = G2[:min_h, :min_w]
    B = B[:min_h, :min_w]

    return R, G1, G2, B



def apply_bayer_scaling(bayer_img, r_scale, g_scale, b_scale, pattern='rggb'):
    """
    Apply the scaling factors to each Bayer pattern position.
    """
    balanced = bayer_img.copy()

    if pattern == 'rggb':
        balanced[0::2, 0::2] *= r_scale
        balanced[0::2, 1::2] *= g_scale
        balanced[1::2, 0::2] *= g_scale
        balanced[1::2, 1::2] *= b_scale
    elif pattern == 'bggr':
        balanced[0::2, 0::2] *= b_scale
        balanced[0::2, 1::2] *= g_scale
        balanced[1::2, 0::2] *= g_scale
        balanced[1::2, 1::2] *= r_scale
    elif pattern == 'grbg':
        balanced[0::2, 0::2] *= g_scale
        balanced[0::2, 1::2] *= r_scale
        balanced[1::2, 0::2] *= b_scale
        balanced[1::2, 1::2] *= g_scale
    elif pattern == 'gbrg':
        balanced[0::2, 0::2] *= g_scale
        balanced[0::2, 1::2] *= b_scale
        balanced[1::2, 0::2] *= r_scale
        balanced[1::2, 1::2] *= g_scale

    return np.clip(balanced, 0, 1)


def visualize_balances(original, white_world, gray_world, camera_preset):
    """
    Visualize results of three white balance algorithms for comparison.
    """
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original (Linearized Bayer)')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(white_world, cmap='gray')
    plt.title('White World')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(gray_world, cmap='gray')
    plt.title('Gray World')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(camera_preset, cmap='gray')
    plt.title('Camera Preset')
    plt.axis('off')

    plt.tight_layout()
    plt.show()







"""
############################################################################################################################################
############################################################################################################################################

DEMOSAICING: # how do i spell? this looks weird double check spelling

############################################################################################################################################
############################################################################################################################################
"""


def demosaic_bilinear(bayer_image, pattern='rggb'):
    """
    Demosaic a Bayer pattern image using bilinear interpolation via RectBivariateSpline.
    
    Parameters:
    -----------
    bayer_image : numpy array (H x W)
        White-balanced linearized Bayer image
    pattern : str
        Bayer pattern: 'rggb', 'bggr', 'grbg', or 'gbrg'
    
    Returns:
    --------
    rgb_image : numpy array (H x W x 3)
        Full-resolution RGB demosaiced image
    """
    
    H, W = bayer_image.shape
    
    # Create coordinate grids for the full output image
    full_x = np.arange(W, dtype=np.float32)
    full_y = np.arange(H, dtype=np.float32)
    
    # Extract Bayer subchannels based on pattern
    if pattern == 'rggb':
        R_sub = bayer_image[0::2, 0::2]
        G1_sub = bayer_image[0::2, 1::2]
        G2_sub = bayer_image[1::2, 0::2]
        B_sub = bayer_image[1::2, 1::2]
        r_offset, r_col_offset = 0, 0
        g1_offset, g1_col_offset = 0, 1
        g2_offset, g2_col_offset = 1, 0
        b_offset, b_col_offset = 1, 1
        
    elif pattern == 'bggr':
        B_sub = bayer_image[0::2, 0::2]
        G1_sub = bayer_image[0::2, 1::2]
        G2_sub = bayer_image[1::2, 0::2]
        R_sub = bayer_image[1::2, 1::2]
        b_offset, b_col_offset = 0, 0
        g1_offset, g1_col_offset = 0, 1
        g2_offset, g2_col_offset = 1, 0
        r_offset, r_col_offset = 1, 1
        
    elif pattern == 'grbg':
        G1_sub = bayer_image[0::2, 0::2]
        R_sub = bayer_image[0::2, 1::2]
        B_sub = bayer_image[1::2, 0::2]
        G2_sub = bayer_image[1::2, 1::2]
        g1_offset, g1_col_offset = 0, 0
        r_offset, r_col_offset = 0, 1
        b_offset, b_col_offset = 1, 0
        g2_offset, g2_col_offset = 1, 1
        
    elif pattern == 'gbrg':
        G1_sub = bayer_image[0::2, 0::2]
        B_sub = bayer_image[0::2, 1::2]
        R_sub = bayer_image[1::2, 0::2]
        G2_sub = bayer_image[1::2, 1::2]
        g1_offset, g1_col_offset = 0, 0
        b_offset, b_col_offset = 0, 1
        r_offset, r_col_offset = 1, 0
        g2_offset, g2_col_offset = 1, 1
    else:
        raise ValueError("Unsupported Bayer pattern.")
    
    # Function to interpolate one channel
    def interpolate_channel(channel_sub, row_offset, col_offset):
        """
        Interpolate a subsampled color channel to full resolution.
        
        Parameters:
        -----------
        channel_sub : numpy array
            Subsampled channel (half resolution in each dimension)
        row_offset, col_offset : int
            Starting position (0 or 1) for this channel in the full Bayer grid
        
        Returns:
        --------
        channel_full : numpy array (H x W)
            Interpolated full-resolution channel
        """
        h, w = channel_sub.shape
        
        # Coordinates of known pixels in the original full image
        # For a subsample with offset, pixels are at positions: offset + 2*index
        x_sub = col_offset + 2 * np.arange(w, dtype=np.float32)
        y_sub = row_offset + 2 * np.arange(h, dtype=np.float32)
        
        # Create the bilinear spline interpolator
        # RectBivariateSpline(y_coords, x_coords, z_values)
        spline = RectBivariateSpline(y_sub, x_sub, channel_sub, kx=1, ky=1)
        
        # Evaluate the spline at all full-resolution pixel coordinates
        # Returns shape (H, W)
        channel_full = spline(full_y, full_x, grid=True)
        
        return channel_full
    
    # Interpolate each channel to full resolution
    R_full = interpolate_channel(R_sub, r_offset, r_col_offset)
    G1_full = interpolate_channel(G1_sub, g1_offset, g1_col_offset)
    G2_full = interpolate_channel(G2_sub, g2_offset, g2_col_offset)
    B_full = interpolate_channel(B_sub, b_offset, b_col_offset)
    
    # Average the two green channels
    G_full = (G1_full + G2_full) / 2.0
    
    # Stack into RGB image (H x W x 3)
    rgb_image = np.stack((R_full, G_full, B_full), axis=2)
    
    # Clip to valid range [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image




"""
############################################################################################################################################
############################################################################################################################################

COLOR SPACE CORRECTION

############################################################################################################################################
############################################################################################################################################
"""


def color_space_correction(rgb_cam, MXYZ_to_cam, MsRGB_to_XYZ):
    """
    Convert camera RGB image to linear sRGB via color space correction.

    Parameters:
    -----------
    rgb_cam : numpy array (H x W x 3)
        Demosaiced RGB image in camera-specific color space (linear values)
    MXYZ_to_cam : numpy array (3 x 3)
        Matrix converting XYZ colorspace to camera RGB (from dcraw, reshaped and scaled)
    MsRGB_to_XYZ : numpy array (3 x 3)
        Matrix converting linear sRGB to XYZ (fixed standard matrix)

    Returns:
    --------
    rgb_sRGB : numpy array (H x W x 3)
        Linear sRGB color corrected image
    """

    # Step 1: Compute MsRGB->cam matrix
    MsRGB_to_cam = MXYZ_to_cam @ MsRGB_to_XYZ

    # Step 2: Normalize matrix rows to sum to 1
    MsRGB_to_cam_norm = MsRGB_to_cam / MsRGB_to_cam.sum(axis=1, keepdims=True)

    # Step 3: Invert the matrix to get cam->sRGB
    Mcam_to_sRGB = np.linalg.inv(MsRGB_to_cam_norm)

    # Step 4: Flatten image for matrix multiplication
    H, W, _ = rgb_cam.shape
    rgb_cam_flat = rgb_cam.reshape(-1, 3).T  # shape: (3, H*W)

    # Step 5: Apply inverse matrix
    rgb_sRGB_flat = Mcam_to_sRGB @ rgb_cam_flat  # (3, H*W)

    # Step 6: Reshape back to image shape
    rgb_sRGB = rgb_sRGB_flat.T.reshape(H, W, 3)

    # Step 7: Clip negative values if any and ensure no overflow beyond 1.0
    rgb_sRGB = np.clip(rgb_sRGB, 0, 1)

    return rgb_sRGB


'''
############################################################################################################################################
############################################################################################################################################

Brightness adjustment and gamma encoding

############################################################################################################################################
############################################################################################################################################
'''



# rgb_linear: your input linear RGB image, values typically 0.0 to >1.0

def brightness_adjust_and_gamma_encode(rgb_linear, target_mean=0.25):
    # Convert RGB to grayscale for luminance measurement
    gray = rgb2gray(rgb_linear)
    mean_intensity = np.mean(gray)

    # Compute scale factor
    scale = target_mean / mean_intensity if mean_intensity > 0 else 1.0

    # Scale image brightness
    rgb_scaled = rgb_linear * scale

    # Clip values to max 1.0
    rgb_scaled = np.clip(rgb_scaled, 0, 1)

    # Define sRGB gamma encoding function vectorized
    def gamma_encode_channel(c):
        a = 0.055
        threshold = 0.0031308
        below_thresh = c <= threshold
        above_thresh = ~below_thresh

        gc = np.zeros_like(c)
        gc[below_thresh] = 12.92 * c[below_thresh]
        gc[above_thresh] = (1 + a) * np.power(c[above_thresh], 1/2.4) - a
        return gc

    # Apply gamma encoding channel-wise
    rgb_gamma = np.zeros_like(rgb_scaled)
    for i in range(3):  # R,G,B channels
        rgb_gamma[:, :, i] = gamma_encode_channel(rgb_scaled[:, :, i])

    return rgb_gamma

"""
COMPRESSION

"""




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


print("Now i am starting White-Balancing")

# from reconnaissance run value of dCRAW
r_scale = 2.165039  # 
g_scale = 1.000000  # 
b_scale = 1.643555  # 

# Example usage ---------------------------------------------------------
# Assume 'image_linear' is your 2D numpy array (linearized Bayer image) normalized [0,1]
# and 'r_scale', 'g_scale', 'b_scale' come from dcraw reconnaissance run.



pattern = 'gbrg'  # adjust to match your Bayer pattern

# Apply white balancing
white_world_img, _ = white_balance_white_world(image_linear, pattern)
gray_world_img, _ = white_balance_gray_world(image_linear, pattern)
camera_preset_img, _ = white_balance_camera_presets(image_linear, r_scale, g_scale, b_scale, pattern)

# Visualize
visualize_balances(image_linear, white_world_img, gray_world_img, camera_preset_img)



###DE-MOSAICING

# bayer_pattern = 'RGGB'  # 
# Apply demosaicing to all white balanced images
# 


print("Now i am starting De-Mosaicing")
pattern = 'gbrg'  # checked this and this was the best
# Apply demosaicing
rgb_demosaiced = demosaic_bilinear(gray_world_img, pattern=pattern)

# Visualize
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_world_img, cmap='gray')
plt.title('White Balanced Bayer (Before Demosaicing)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb_demosaiced)
plt.title('Demosaiced RGB Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Optional: save the result
# from PIL import Image
# rgb_uint8 = (np.clip(rgb_demosaiced, 0, 1) * 255).astype(np.uint8)
# Image.fromarray(rgb_uint8).save('demosaiced.png')



#### COLOR SPACE CORRECTION 
print("Now i am starting COLOR SPACE CORRECTION ")

# ------- Fixed MsRGB->XYZ matrix as per sRGB standard -------
MsRGB_to_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

#  vector values from dcraw --> got this from slack
dcraw_vector = [
    24542,-10860,-3401,-1490,11370,-297,2858,-605,3225 
]
MXYZ_to_cam = np.array(dcraw_vector).reshape((3, 3)) / 10000.0


rgb_sRGB = color_space_correction(rgb_demosaiced, MXYZ_to_cam, MsRGB_to_XYZ)

plt.imshow(rgb_sRGB)
plt.title("Color Corrected Linear sRGB")
plt.axis('off')
plt.show()




### Brightness Adjustment and Gamma Encoding

print("Now i am starting Brightness Adjustment and Gamma Encoding ")
#rgb_corrected = brightness_adjust_and_gamma_encode(rgb_sRGB, target_mean=0.25)
for target in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
    rgb_corrected = brightness_adjust_and_gamma_encode(rgb_sRGB, target_mean=target)
    
    plt.figure()
    plt.imshow(rgb_corrected)
    plt.title(f'Brightness Adjusted & Gamma Encoded (target mean={target})')
    plt.axis('off')
    plt.show()


#### Compression

print("Now i am starting Compression Comparison ")

# Assume rgb_img is your final processed image with values in [0, 1]

# Convert to uint8 [0, 255] for saving formats
rgb_uint8 = (rgb_corrected * 255).astype('uint8')

# Save as PNG (lossless)
imsave('output_image.png', rgb_uint8)

# Save as JPEG with quality=95 (lossy compression)
imsave('output_image_quality95.jpg', rgb_uint8, quality=95)












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
