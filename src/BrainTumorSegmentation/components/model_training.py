import os
import random
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import SwinUNETR
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from BrainTumorSegmentation import logger
from BrainTumorSegmentation.entity.config_entity import ModelTrainingConfig


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for preprocessed brain tumor segmentation data.
    """

    def __init__(self, patient_list: List[str], data_root: str):
        """
        Initialize the brain tumor dataset.

        Args:
            patient_list (List[str]): List of patient IDs
            data_root (str): Root directory of the preprocessed data for a specific split
        """
        self.patient_list = patient_list
        self.data_root = data_root

    def __len__(self) -> int:
        """Return the number of patients in the dataset."""
        return len(self.patient_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the preprocessed multimodal images and segmentation mask for a patient.

        Args:
            idx (int): Index of the patient in the list

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor (4 channels) and mask tensor
        """
        pid = self.patient_list[idx]
        ppath = os.path.join(self.data_root, pid)

        # Load preprocessed modalities
        modalities = ["t1n", "t1c", "t2w", "t2f"]
        images = []

        for mod in modalities:
            # Load preprocessed NIfTI file
            img_path = os.path.join(ppath, f"{pid}-{mod}.nii.gz")
            nii_img = nib.load(img_path)
            img_data = nii_img.get_fdata()

            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(img_data).float()

            # Add channel dimension if not present
            if len(img_tensor.shape) == 3:  # If it's just spatial dimensions
                img_tensor = img_tensor.unsqueeze(0)

            images.append(img_tensor)

        # Stack all modalities along channel dimension
        image = torch.cat(images, dim=0)  # Shape: (4, H, W, D)

        # Load segmentation mask
        mask_path = os.path.join(ppath, f"{pid}-seg.nii.gz")
        nii_mask = nib.load(mask_path)
        mask_data = nii_mask.get_fdata()

        # Convert to PyTorch tensor and ensure it's a long tensor for labels
        mask = torch.from_numpy(mask_data).long()

        # Add channel dimension if not present, then squeeze to remove it
        # (CrossEntropyLoss expects class indices not one-hot)
        if len(mask.shape) == 3:  # If it's just spatial dimensions
            mask = mask.unsqueeze(0)

        return image, mask.squeeze(0)  # Return as (C, H, W, D) and (H, W, D)


class ModelTraining:
    """
    Model Training class to handle the training of brain tumor segmentation model.
    """

    def __init__(self, config: ModelTrainingConfig):
        """
        Initialize the ModelTraining class with the configuration.

        Args:
            config (ModelTrainingConfig): Configuration for model training.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

        logger.info(f"Using device: {self.device}")

    def _get_patient_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get patient lists from preprocessed train, validation, and test directories.

        Returns:
            Tuple[List[str], List[str], List[str]]: Train, validation, and test patient lists
        """
        # Get patients from each split directory
        train_patients = [
            f
            for f in os.listdir(os.path.join(self.config.data_path, "train"))
            if os.path.isdir(os.path.join(self.config.data_path, "train", f))
        ]

        val_patients = [
            f
            for f in os.listdir(os.path.join(self.config.data_path, "val"))
            if os.path.isdir(os.path.join(self.config.data_path, "val", f))
        ]

        test_patients = [
            f
            for f in os.listdir(os.path.join(self.config.data_path, "test"))
            if os.path.isdir(os.path.join(self.config.data_path, "test", f))
        ]

        logger.info(
            f"Found patients: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test"
        )
        return train_patients, val_patients, test_patients

    def _create_dataloaders(
        self, train_patients: List[str], val_patients: List[str]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training and validation using preprocessed data.

        Args:
            train_patients (List[str]): List of training patient IDs
            val_patients (List[str]): List of validation patient IDs

        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation DataLoaders
        """
        # Create datasets with the correct paths for preprocessed data
        train_ds = BrainTumorDataset(
            train_patients, os.path.join(self.config.data_path, "train")
        )

        val_ds = BrainTumorDataset(
            val_patients, os.path.join(self.config.data_path, "val")
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def _create_model(self) -> torch.nn.Module:
        """
        Create the SwinUNETR model.

        Returns:
            torch.nn.Module: Initialized model
        """
        model = SwinUNETR(
            img_size=self.config.img_size,
            in_channels=4,  # 4 modalities
            out_channels=4,  # 4 classes (background + 3 tumor types),
            feature_size=self.config.feature_size,
        ).to(self.device)
        return model

    def _calculate_accuracy(self, outputs, labels):
        """
        Calculate classification accuracy.

        Args:
            outputs: Model predictions
            labels: Ground truth labels

        Returns:
            float: Accuracy score
        """
        # Convert outputs to predicted class
        preds = torch.argmax(outputs, dim=1)

        # Calculate accuracy
        correct = (preds == labels).sum().item()
        total = labels.numel()

        return correct / total if total > 0 else 0

    def train_model(self):
        """
        Train the model using the specified configuration.
        """
        # Get patient splits
        train_patients, val_patients, _ = self._get_patient_splits()

        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders(
            train_patients, val_patients
        )

        # Initialize model, optimizer, and loss function
        model = self._create_model()
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = CrossEntropyLoss()

        # Training loop
        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_acc = 0

            for imgs, masks in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Training",
            ):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Calculate metrics
                batch_acc = self._calculate_accuracy(outputs, masks)
                train_acc += batch_acc / len(train_loader)
                train_loss += loss.item() / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0
            val_acc = 0

            with torch.no_grad():
                for imgs, masks in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{self.config.num_epochs} - Validation",
                ):
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)

                    # Calculate metrics
                    batch_acc = self._calculate_accuracy(outputs, masks)
                    val_acc += batch_acc / len(val_loader)
                    val_loss += loss.item() / len(val_loader)

            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_acc:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    self.config.root_dir, "swin_unetr_model.pth"
                )
                torch.jit.script(model).save(best_model_path)
                logger.info(f"✅ Saved best model with validation loss: {val_loss:.4f}")

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        logger.info(f"✅ Model saved at: {best_model_path}")

    def run(self):
        """
        Run the model training pipeline.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.train_model()
