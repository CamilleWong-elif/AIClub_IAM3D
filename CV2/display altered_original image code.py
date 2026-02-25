original = img.copy()  # save original BEFORE augment loop

for aug in cv2_augmentations:
    img = aug(img)

# ---- DISPLAY BLOCK ----
if DEBUG_VISUALIZE and i < 5:  # only show first 5 images
    combined = np.hstack((original, img))  # side-by-side
    cv2.imshow("Original | Augmented", combined)
    cv2.waitKey(500)   # show for 500ms (0.5 sec)
    cv2.destroyAllWindows()