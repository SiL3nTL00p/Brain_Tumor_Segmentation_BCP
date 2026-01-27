import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import matplotlib.patches as patches


def display_brats_patient_gray(patient_dir):
    """
    Displays FLAIR, T1, T1CE, T2 (gray) for the axial slice
    with the largest tumor area, including mask overlay,
    bounding box, and label on the box.
    """

    modalities = {
        "FLAIR": "flair",
        "T1":    "_t1.",
        "T1CE":  "t1ce",
        "T2":    "_t2.",
    }

    # load segmentation
    seg_file = [f for f in os.listdir(patient_dir) if "seg" in f][0]
    seg = nb.load(os.path.join(patient_dir, seg_file)).get_fdata()

    # choose slice with largest tumor
    sums = seg.sum(axis=(0, 1))
    z = int(np.argmax(sums)) if sums.max() > 0 else seg.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (name, key) in zip(axes, modalities.items()):
        img_file = [f for f in os.listdir(patient_dir) if key in f][0]
        img = nb.load(os.path.join(patient_dir, img_file)).get_fdata()

        slice_img = np.rot90(img[:, :, z])
        slice_seg = np.rot90(seg[:, :, z])

        ax.imshow(slice_img, cmap="gray")

        ax.imshow(
            np.ma.masked_where(slice_seg == 0, slice_seg),
            cmap="Reds",
            alpha=0.45
        )

        ys, xs = np.where(slice_seg > 0)
        if len(xs) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            width = x_max - x_min
            height = y_max - y_min

            tumor_area = len(xs)
            tumor_size = max(width, height)

            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=1.5,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)

            label = f"tumor,area:{tumor_area},size:{tumor_size}"
            ax.text(
                x_min,
                y_min - 6,
                label,
                color="white",
                fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="red", edgecolor="red", pad=1.5)
            )

        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(
        f"{os.path.basename(patient_dir)} | Axial slice z = {z}",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()



