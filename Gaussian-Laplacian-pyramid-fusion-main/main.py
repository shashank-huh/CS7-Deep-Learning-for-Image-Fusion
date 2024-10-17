import matplotlib.pyplot as plt
import cv2
from numpy import uint8
import os
import time 

from pyramids import *
from weight_map import *

from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
from skimage.transform import resize

def main_multimodal_fusion(im_vis, im_ir, kernel, levels, window_size):
    """
    A function to fuse two images of different modalities, in this example we use visible and NIR images.

    :param im_vis: The visible image, a numpy array of floats within [0, 1] of shape (N, M, 3)
    :param im_ir: The NIR image, a numpy array of floats within [0, 1] of shape (N, M)
    :param kernel: The Gaussian kernel used to build pyramids
    :param levels: The desired levels in the pyramids
    :param window_size: The window size used to compute the local entropy and the local contrast
    """

    im_vis = convert_image_to_floats(im_vis)
    im_ir = convert_image_to_floats(im_ir[:,:,1])

    im_vis_hsv = rgb2hsv(im_vis)
    value_channel = im_vis_hsv[:, :, 2]

    # kernels to compute visibility
    kernel1 = classical_gaussian_kernel(5, 2)
    kernel2 = classical_gaussian_kernel(5, 2)

    # Computation of local entropy, local contrast and visibility for value channel
    local_entropy_value = normalized_local_entropy(value_channel, window_size)
    local_contrast_value = local_contrast(value_channel, window_size)
    visibility_value = visibility(value_channel, kernel1, kernel2)
    # Combination of local entropy, local contrast and visibility for value channel
    weight_value = weight_combination(local_entropy_value, local_contrast_value, visibility_value, 1, 1, 1)
    weight_value[weight_value==0] = 1e-6;

    # Computation of local entropy, local contrast and visibility for IR image
    local_entropy_ir = normalized_local_entropy(im_ir, window_size)
    local_contrast_ir = local_contrast(im_ir, window_size)
    visibility_ir = visibility(im_ir, kernel1, kernel2)
    # Combination of local entropy, local contrast and visibility for IR image
    weight_ir = weight_combination(local_entropy_ir, local_contrast_ir, visibility_ir, 1, 1, 1)
    weight_ir[weight_ir==0] = 1e-6;

    # Normalising weights of value channel and IR image
    weightN_value, weightN_ir = weight_normalization(weight_value, weight_ir)

    # Creating Gaussian pyramids of the weights maps of respectively the value channel and IR image
    gauss_pyr_value_weights = gaussian_pyramid(weightN_value, kernel, levels)
    gauss_pyr_ir_weights = gaussian_pyramid(weightN_ir, kernel, levels)

    # Creating Laplacian pyramids of respectively the value channel and IR image
    lap_pyr_value = laplacian_pyramid(value_channel, kernel, levels)
    lap_pyr_ir = laplacian_pyramid(im_ir, kernel, levels)

    # Creating the fused Laplacian of the two modalities
    lap_pyr_fusion = fused_laplacian_pyramid(gauss_pyr_value_weights, gauss_pyr_ir_weights, lap_pyr_value, lap_pyr_ir)

    # Creating the Gaussian pyramid of value channel in order to collapse the fused Laplacian pyramid
    gauss_pyr_value = gaussian_pyramid(value_channel, kernel, levels)
    collapsed_image = collapse_pyramid(lap_pyr_fusion, gauss_pyr_value)

    # Replacing the value channel in HSV visible image by the collapsed image
    im_vis_hsv_fusion = im_vis_hsv.copy()
    im_vis_hsv_fusion[:, :, 2] = collapsed_image
    im_vis_rgb_fusion = hsv2rgb(im_vis_hsv_fusion)

    return im_vis_rgb_fusion

def main_gaussian_laplacian_pyramids(image, kernel, levels):
    """
    A function to build the Gaussian and Laplacian pyramids of an image
    :param image: A grayscale or 3 channels image, a numpy array of floats within [0, 1] of shape (N, M) or (N, M, 3)
    :param kernel: The Gaussian kernel used to build pyramids
    :param levels: The desired levels in the pyramids
    """

    image = convert_image_to_floats(image)

    # Building the Gaussian and Laplacian pyramids
    gauss_pyr = gaussian_pyramid(image, kernel, levels)
    lap_pyr = laplacian_pyramid(image, kernel, levels)

    # Displaying pyramids
    i = 1
    for p in gauss_pyr:
        plt.subplot(1, len(gauss_pyr), i)
        plt.imshow(p, cmap='gray')
        i += 1
    plt.show()

    i = 1
    for p in lap_pyr:
        plt.subplot(1, len(lap_pyr), i)
        plt.imshow(p, cmap='gray')
        i += 1
    plt.show()

    # Building and displaying collapsed image
    collapsed_image = collapse_pyramid(lap_pyr, gauss_pyr)
    plt.imshow(collapsed_image, cmap='gray')
    plt.show()

def convert_image_to_floats(image):
    """
    A function to convert an image to a numpy array of floats within [0, 1]

    :param image: The image to be converted
    :return: The converted image
    """

    if np.max(image) <= 1.0:
        return image
    else:
        return image / 255.0

kernel = smooth_gaussian_kernel(0.4)
levels = 4
window_size = 5

dataset = 'D:/DL/M3FD_Fusion/'
files = [f.name for f in os.scandir(dataset+'vis/')]

for f in files:
    n = 31
    vis_path = dataset + 'vis/' + files[n]
    ir_path = dataset + 'ir/' + files[n]
    fused_path = 'D:/DL/pyramids_fused/' +files[n]

    # vis_path = 'D:/DL/M3FD_Fusion/vis/03878.png'
    # ir_path = 'D:/DL/M3FD_Fusion/ir/03878.png'
    # fused_path = 'D:/DL/pyramids_fused/03878.png'
    

    image_vis = cv2.imread(vis_path, cv2.IMREAD_UNCHANGED)
    image_ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)

    reduce = 1
    vis_rs = resize(image_vis, (int(image_vis.shape[0]/reduce), int(image_vis.shape[1]/reduce)))
    ir_rs = resize(image_ir, (int(image_ir.shape[0]/reduce), int(image_ir.shape[1]/reduce), 3))

    tstart = time.time()
    fus = main_multimodal_fusion(vis_rs, ir_rs, kernel, levels, window_size)
    print('Elapsed: %s' % (time.time() - tstart))
    fus = np.array(fus * 255.0/fus.max())
    fus = fus.astype(uint8)

    # plt.subplot(1,1,1)
    # plt.imshow(fus)
    # plt.show() 
    cv2.imwrite(filename=fused_path, img=fus)
    # Display the images
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title('Visible Image')
    plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Infrared Image')
    plt.imshow(cv2.cvtColor(image_ir, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Fused Image')
    plt.imshow(cv2.cvtColor(fus, cv2.COLOR_BGR2RGB))  # Use the fused image data
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    exit()