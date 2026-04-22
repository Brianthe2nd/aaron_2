import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ----------- CORE ANALYSIS FUNCTION -----------

def analyze_image_difference(original, modified):
    """Compute LAB differences: brightness, contrast, warmth."""
    
    lab1 = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(modified, cv2.COLOR_BGR2LAB)

    L1, A1, B1 = cv2.split(lab1)
    L2, A2, B2 = cv2.split(lab2)

    # Brightness diff
    delta_L = np.mean(L2) - np.mean(L1)

    # Contrast diff
    delta_contrast = np.std(L2) - np.std(L1)

    # Color/warmth diff
    delta_A = np.mean(A2) - np.mean(A1)
    delta_B = np.mean(B2) - np.mean(B1)

    # Structural similarity
    ssim_val = ssim(original, modified, channel_axis=2)

    return delta_L, delta_contrast, delta_A, delta_B, ssim_val


# ----------- APPLY CORRECTIONS TO IMAGE A -----------

def apply_lab_correction(image, delta_L, delta_contrast, delta_A, delta_B):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    L = L.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # Brightness shift
    L += delta_L

    # Contrast adjustment (scaled to avoid overcorrection)
    mean_L = np.mean(L)
    contrast_scale = 1 + (delta_contrast / 50.0)  # divisor 50 makes it realistic
    L = (L - mean_L) * contrast_scale + mean_L

    # Warmth shifts
    A += delta_A
    B += delta_B

    # Clip
    L = np.clip(L, 0, 255).astype(np.uint8)
    A = np.clip(A, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    corrected = cv2.merge((L, A, B))
    corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

    return corrected


# ----------- MAIN PIPELINE: MAKE A LOOK LIKE B -----------

def match_image(imgA, imgB):
    """Automatically adjust imgA so it matches imgB."""
    imgA = cv2.imread("g.png")
    imgB = cv2.imread("templates/x_logo.png")
    
    # Make sure both images are the same size
    imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

    # 1. Get differences
    dL, dContrast, dA, dB, ssim_before = analyze_image_difference(imgA, imgB)

    # 2. Apply correction to A
    imgA_corrected = apply_lab_correction(imgA, dL, dContrast, dA, dB)

    # 3. Measure similarity after correction
    _, _, _, _, ssim_after = analyze_image_difference(imgA_corrected, imgB)

    print("\n===== MATCHING SUMMARY =====")
    print(f"Brightness ΔL: {dL:.3f}")
    print(f"Contrast Δ:    {dContrast:.3f}")
    print(f"Warmth ΔA:     {dA:.3f}")
    print(f"Warmth ΔB:     {dB:.3f}")
    print(f"SSIM before:   {ssim_before:.4f}")
    print(f"SSIM after:    {ssim_after:.4f}")
    
    return imgA_corrected


import cv2
import matplotlib.pyplot as plt

# --- Load images ---
imgA = cv2.imread("g.png")
imgB = cv2.imread("templates/x_logo.png")

# Make sure both are BGR → RGB for display
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Match A to B ---
matchedA = match_image(imgA, imgB)

# --- Display side-by-side ---
plt.figure(figsize=(15, 8))

plt.subplot(1, 3, 1)
plt.title("Original A")
plt.imshow(to_rgb(imgA))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("A Corrected → B")
plt.imshow(to_rgb(matchedA))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Target B")
plt.imshow(to_rgb(imgB))
plt.axis("off")

plt.tight_layout()
plt.show()
