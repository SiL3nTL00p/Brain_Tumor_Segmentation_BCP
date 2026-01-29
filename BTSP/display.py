import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import matplotlib.patches as patches


class BraTSGrayViewer:
    """
    class to visualize BraTS MRI modalities (gray) with
    tumor mask, bounding box, and header-on-box label.
    """

    def __init__(self, patient_dir):
        self.patient_dir = patient_dir
        self.modalities = {
            "FLAIR": "flair",
            "T1":    "_t1.",
            "T1CE":  "t1ce",
            "T2":    "_t2.",
        }
        self.seg = self._load_segmentation()
        self.z = self._select_slice()

    def _load_segmentation(self):
        seg_file = [f for f in os.listdir(self.patient_dir) if "seg" in f][0]
        return nb.load(os.path.join(self.patient_dir, seg_file)).get_fdata()

    def _select_slice(self):
        sums = self.seg.sum(axis=(0, 1))
        return int(np.argmax(sums)) if sums.max() > 0 else self.seg.shape[2] // 2

    def _load_image(self, key):
        img_file = [f for f in os.listdir(self.patient_dir) if key in f][0]
        return nb.load(os.path.join(self.patient_dir, img_file)).get_fdata()

    def display(self):
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        for ax, (name, key) in zip(axes, self.modalities.items()):
            img = self._load_image(key)

            slice_img = np.rot90(img[:, :, self.z])
            slice_seg = np.rot90(self.seg[:, :, self.z])

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
            f"{os.path.basename(self.patient_dir)} | Axial slice z = {self.z}",
            fontsize=16
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()
