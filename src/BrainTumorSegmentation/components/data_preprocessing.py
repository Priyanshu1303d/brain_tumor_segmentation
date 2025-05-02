import os
import random

import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
)
from tqdm import tqdm

from BrainTumorSegmentation import logger
from BrainTumorSegmentation.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    """
    Data Preprocessing class to handle the preprocessing of brain tumor segmentation data.
    """

    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize the DataPreprocessing class with the configuration.

        Args:
            config (DataPreprocessingConfig): Configuration for data preprocessing.
        """
        self.config = config
        self.transform = Compose(
            [
                LoadImage(),
                EnsureChannelFirst(),
                Resize(self.config.img_size),
                ScaleIntensity(),
            ]
        )

    def preprocess_and_save(self, input_path, output_path):
        """
        Apply preprocessing to the input image and save it to the output path.
        """
        try:
            data = self.transform(input_path)
        except Exception as e:
            logger.error(f"Error transforming image {input_path}: {e}")
            return
        data_np = (
            data.squeeze().numpy() if data.shape[0] == 1 else data.numpy()
        )  # handles both mask and image
        affine = nib.load(input_path).affine
        nib.save(nib.Nifti1Image(data_np, affine), output_path)

    def preprocess_data(self):
        """
        Run the preprocessing pipeline across train, val, and test splits.
        """
        # Ensure output folders exist
        for split in ["train", "val", "test"]:
            os.makedirs(
                os.path.join(self.config.preprocessed_data_path, split), exist_ok=True
            )

        # List and filter patients
        all_patients = [
            f
            for f in os.listdir(self.config.data_path)
            if os.path.isdir(os.path.join(self.config.data_path, f))
            and os.path.exists(
                os.path.join(self.config.data_path, f, f + "-seg.nii.gz")
            )
        ]

        random.seed(42)
        random.shuffle(all_patients)

        # Split - 70% train, 15% val, 15% test
        l = len(all_patients)
        train_patients = all_patients[: int(0.7 * l)]
        val_patients = all_patients[int(0.7 * l) : int(0.85 * l)]
        test_patients = all_patients[int(0.85 * l) :]

        splits = {"train": train_patients, "val": val_patients, "test": test_patients}

        for split, patient_list in splits.items():
            logger.info(f"Preprocessing {split} set ({len(patient_list)} patients)...")
            for pid in tqdm(patient_list):
                patient_input_path = os.path.join(self.config.data_path, pid)
                patient_output_path = os.path.join(
                    self.config.preprocessed_data_path, split, pid
                )
                os.makedirs(patient_output_path, exist_ok=True)

                for modality in ["t1n", "t1c", "t2w", "t2f", "seg"]:
                    in_file = os.path.join(
                        patient_input_path, f"{pid}-{modality}.nii.gz"
                    )
                    out_file = os.path.join(
                        patient_output_path, f"{pid}-{modality}.nii.gz"
                    )

                    if not os.path.exists(in_file):
                        logger.warning(f"Missing file: {in_file}")
                        continue

                    try:
                        self.preprocess_and_save(in_file, out_file)
                    except Exception as e:
                        logger.error(f"Failed to preprocess {in_file}: {e}")

        logger.info(
            f"✅ Preprocessing complete. Data saved in: {self.config.preprocessed_data_path}"
        )

    def run(self):
        """
        Run the data preprocessing pipeline.
        """
        logger.info("Starting data preprocessing...")
        self.preprocess_data()
        logger.info("Data preprocessing completed.")
