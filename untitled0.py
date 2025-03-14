import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import felzenszwalb, slic
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import os

def process_image(image_path):
    image = io.imread(image_path, as_gray=True)

    # Add Gaussian Noise
    noise = 0.2 * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)

    # Convert to uint8 for OpenCV processing
    noisy_uint8 = (noisy_image * 255).astype(np.uint8)

    # Apply Bilateral Filtering (better edge preservation than Gaussian)
    filtered_image = cv2.bilateralFilter(noisy_uint8, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply Adaptive Histogram Equalization (CLAHE) for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_image = clahe.apply(filtered_image)

    # Apply Otsu’s Thresholding for better segmentation
    _, otsu_threshold = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # K-Means Segmentation
    flat_image = enhanced_image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(flat_image)
    kmeans_segmented = kmeans.labels_.reshape(enhanced_image.shape)

    # Mean Shift Segmentation
    bandwidth = estimate_bandwidth(flat_image, quantile=0.2, n_samples=500)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift.fit(flat_image)
    mean_shift_segmented = mean_shift.labels_.reshape(enhanced_image.shape)

    # Graph-Based Segmentation
    graph_segmented = felzenszwalb(enhanced_image.astype(np.float32) / 255, scale=100)

    # SLIC Superpixel Segmentation (Fixed channel_axis for grayscale images)
    slic_segmented = slic(enhanced_image, n_segments=250, compactness=10, channel_axis=None)

    # Region Growing (Threshold-Based)
    seed_point = (enhanced_image.shape[0] // 2, enhanced_image.shape[1] // 2)
    region_growing_segmented = np.zeros_like(enhanced_image, dtype=np.uint8)
    threshold = 30
    seed_value = enhanced_image[seed_point]

    mask = np.abs(enhanced_image - seed_value) < threshold
    region_growing_segmented[mask] = 255

    # Connected Components
    _, connected_component_image = cv2.connectedComponents(otsu_threshold)

    # Compute Metrics
    def iou(seg, gt):
        intersection = np.logical_and(seg, gt).sum()
        union = np.logical_or(seg, gt).sum()
        return intersection / union if union > 0 else 0

    def dice_coeff(seg, gt):
        intersection = np.logical_and(seg, gt).sum()
        return 2 * intersection / (seg.sum() + gt.sum()) if (seg.sum() + gt.sum()) > 0 else 0

    gt_mask = (image > 0.5).astype(np.uint8)  # Use original image as a GT mask approximation

    iou_score = iou(kmeans_segmented, gt_mask)
    dice_score = dice_coeff(kmeans_segmented, gt_mask)
    ssim_score = ssim(kmeans_segmented, gt_mask, data_range=1)

    return (image, noisy_image, enhanced_image, kmeans_segmented, mean_shift_segmented, 
            region_growing_segmented, graph_segmented, slic_segmented, connected_component_image, 
            iou_score, dice_score, ssim_score)

# Directory containing images
image_dir = r"C:\crab_detection\images.cv_ra67e9eg6eccki07q0kc3o\data\train"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")][:20]

iou_scores, dice_scores, ssim_scores = [], [], []

fig, axes = plt.subplots(len(image_files), 8, figsize=(24, len(image_files) * 3))

for idx, img_path in enumerate(image_files):
    (image, noisy_image, enhanced_image, kmeans_segmented, mean_shift_segmented, 
     region_growing_segmented, graph_segmented, slic_segmented, connected_component_image,
     iou, dice, ssim_val) = process_image(img_path)

    iou_scores.append(iou)
    dice_scores.append(dice)
    ssim_scores.append(ssim_val)

    axes[idx, 0].imshow(image, cmap="gray")
    axes[idx, 0].set_title("Original Image")

    axes[idx, 1].imshow(noisy_image, cmap="gray")
    axes[idx, 1].set_title("Noisy Image")

    axes[idx, 2].imshow(enhanced_image, cmap="gray")
    axes[idx, 2].set_title("Enhanced Image")

    axes[idx, 3].imshow(kmeans_segmented, cmap="gray")
    axes[idx, 3].set_title("K-Means Segmentation")

    axes[idx, 4].imshow(mean_shift_segmented, cmap="gray")
    axes[idx, 4].set_title("Mean Shift Segmentation")

    axes[idx, 5].imshow(region_growing_segmented, cmap="gray")
    axes[idx, 5].set_title("Region Growing")

    axes[idx, 6].imshow(graph_segmented, cmap="nipy_spectral")
    axes[idx, 6].set_title("Graph-Based Segmentation")

    axes[idx, 7].imshow(connected_component_image, cmap="nipy_spectral")
    axes[idx, 7].set_title("Connected Components")

    print(f"Processed {img_path}: IoU={iou:.4f}, Dice={dice:.4f}, SSIM={ssim_val:.4f}")

plt.tight_layout()
plt.show()

# Compute Averages
avg_iou = np.mean(iou_scores)
avg_dice = np.mean(dice_scores)
avg_ssim = np.mean(ssim_scores)
print(f"Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}, Average SSIM: {avg_ssim:.4f}")
