from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """BraTSDataset: loads BraTS-MET 3D MRI images and segmentation labels."""

    def __init__(self, data_dir: str, patch_size: tuple = (128, 128, 128), training: bool = True, with_label: bool = True, class_crop_probability: float = 0.85):
        """Initialize BraTSDataset with data directory, patch size, training mode, and label settings."""
        super().__init__()
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.patch_size = tuple(patch_size)
        self.training = training
        self.with_label = with_label
        self.class_crop_probability = class_crop_probability
        self.modalities = ("t1n", "t1c", "t2w", "t2f")
        self.valid_labels = (0, 1, 2, 3, 4)
        self.excluded_patient_ids = {
            "BraTS-MET-01094-002",
            "BraTS-MET-01184-002"
        }
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")
        candidate_dirs = set()
        if self.data_dir.is_dir() and self.data_dir.name.startswith("BraTS-MET-"):
            candidate_dirs.add(self.data_dir)
        for patient_dir in self.data_dir.rglob("BraTS-MET-*"):
            if patient_dir.is_dir():
                candidate_dirs.add(patient_dir)
        patient_map = {}
        for patient_dir in sorted(candidate_dirs):
            patient_id = patient_dir.name
            if patient_id in self.excluded_patient_ids:
                continue
            modality_paths = [patient_dir / f"{patient_id}-{modality}.nii.gz" for modality in self.modalities]
            if not all(image_path.exists() for image_path in modality_paths):
                continue
            if self.with_label:
                label_path = patient_dir / f"{patient_id}-seg.nii.gz"
                if not label_path.exists():
                    continue
            if patient_id not in patient_map:
                patient_map[patient_id] = patient_dir
            elif len(patient_dir.parts) < len(patient_map[patient_id].parts):
                patient_map[patient_id] = patient_dir
        self.patient_dirs = [patient_map[patient_id] for patient_id in sorted(patient_map)]
        if len(self.patient_dirs) == 0:
            raise RuntimeError(f"No valid patient folders found in: {self.data_dir}")

    def __len__(self):
        """Return the number of patients in the dataset."""
        return len(self.patient_dirs)

    def normalize(self, image: np.ndarray):
        """Normalize image by foreground mean and standard deviation."""
        image = image.astype(np.float32, copy=False)
        foreground_mask = image != 0
        if not np.any(foreground_mask):
            return image
        foreground = image[foreground_mask]
        mean = foreground.mean()
        std = foreground.std()
        if std < 1e-8:
            std = 1.0
        normalized = np.zeros_like(image, dtype=np.float32)
        normalized[foreground_mask] = (image[foreground_mask] - mean) / std
        return normalized

    def get_padded_shape(self, image_shape: tuple):
        """Return padded shape accommodating both image and patch sizes."""
        return tuple(max(image_size, patch_size) for image_size, patch_size in zip(image_shape, self.patch_size))

    def get_start_from_center(self, center: np.ndarray, image_shape: tuple):
        """Calculate crop start coordinates from a center point within the image."""
        crop_start = []
        for dimension in range(3):
            start = int(center[dimension]) - self.patch_size[dimension] // 2
            maximum_start = image_shape[dimension] - self.patch_size[dimension]
            start = max(0, min(start, maximum_start))
            crop_start.append(start)
        return tuple(crop_start)

    def choose_crop_start(self, image_shape: tuple, label: np.ndarray = None):
        """Choose crop start based on label foreground or random sampling."""
        if label is None:
            image_center = np.asarray(image_shape) // 2
            return self.get_start_from_center(image_center, image_shape)
        foreground_classes = np.unique(label)
        foreground_classes = foreground_classes[(foreground_classes >= 1) & (foreground_classes <= 4)]
        if not self.training:
            foreground_coordinates = np.where(label > 0)
            if len(foreground_coordinates[0]) > 0:
                tumor_center = np.asarray([(axis.min() + axis.max()) // 2 for axis in foreground_coordinates])
                return self.get_start_from_center(tumor_center, image_shape)
            image_center = np.asarray(image_shape) // 2
            return self.get_start_from_center(image_center, image_shape)
        use_class_crop = len(foreground_classes) > 0 and np.random.random() < self.class_crop_probability
        if use_class_crop:
            selected_class = int(np.random.choice(foreground_classes))
            class_indices = np.flatnonzero(label == selected_class)
            if len(class_indices) > 0:
                selected_index = class_indices[np.random.randint(len(class_indices))]
                class_center = np.asarray(np.unravel_index(selected_index, label.shape))
                return self.get_start_from_center(class_center, image_shape)
        return tuple(np.random.randint(0, image_size - patch_size + 1) for image_size, patch_size in zip(image_shape, self.patch_size))

    def extract_patch(self, data, original_shape: tuple, crop_start: tuple, dtype):
        """Extract and pad a patch from data at the given crop_start position."""
        source_slices = []
        padding = []
        for start, image_size, patch_size in zip(crop_start, original_shape, self.patch_size):
            source_start = min(start, image_size)
            source_end = min(start + patch_size, image_size)
            source_size = max(source_end - source_start, 0)
            source_slices.append(slice(source_start, source_end))
            padding.append((0, patch_size - source_size))
        patch = np.asarray(data[tuple(source_slices)], dtype=dtype)
        if any(padding_size > 0 for _, padding_size in padding):
            patch = np.pad(patch, pad_width=tuple(padding), mode="constant", constant_values=0)
        if patch.shape != self.patch_size:
            raise RuntimeError(f"Incorrect patch shape: got {patch.shape}, expected {self.patch_size}")
        return patch

    def __getitem__(self, index: int):
        """Load and return image patch and optional label patch by patient index."""
        patient_dir = self.patient_dirs[index]
        patient_id = patient_dir.name
        first_image_path = patient_dir / f"{patient_id}-{self.modalities[0]}.nii.gz"
        first_nifti = nib.load(str(first_image_path), mmap=True, keep_file_open=False)
        original_shape = first_nifti.shape
        if len(original_shape) != 3:
            raise ValueError(f"Expected a 3D image, but {patient_id} has shape {original_shape}")
        padded_shape = self.get_padded_shape(original_shape)
        target = None
        if self.with_label:
            label_path = patient_dir / f"{patient_id}-seg.nii.gz"
            label_nifti = nib.load(str(label_path), mmap=True, keep_file_open=False)
            if label_nifti.shape != original_shape:
                raise ValueError(f"Shape mismatch for {patient_id}: label shape is {label_nifti.shape}, image shape is {original_shape}")
            label = np.asarray(label_nifti.dataobj, dtype=np.int16)
            invalid_mask = (label < 0) | (label > 4)
            if np.any(invalid_mask):
                label = label.copy()
                label[invalid_mask] = 0
            crop_start = self.choose_crop_start(padded_shape, label)
            label_patch = self.extract_patch(label, original_shape, crop_start, np.int64)
            target = torch.from_numpy(np.ascontiguousarray(label_patch))
        else:
            crop_start = self.choose_crop_start(padded_shape)
        image_patches = []
        for modality in self.modalities:
            image_path = patient_dir / f"{patient_id}-{modality}.nii.gz"
            if not image_path.exists():
                raise FileNotFoundError(f"MRI file does not exist: {image_path}")
            nifti_image = nib.load(str(image_path), mmap=True, keep_file_open=False)
            if nifti_image.shape != original_shape:
                raise ValueError(f"Shape mismatch for {patient_id}: {modality} shape is {nifti_image.shape}, reference shape is {original_shape}")
            image_patch = self.extract_patch(nifti_image.dataobj, original_shape, crop_start, np.float32)
            image_patch = self.normalize(image_patch)
            image_patches.append(image_patch)
        image = np.stack(image_patches, axis=0).astype(np.float32, copy=False)
        image = torch.from_numpy(np.ascontiguousarray(image))
        if self.with_label:
            return image, target
        return image, patient_id
