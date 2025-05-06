import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from BrainTumorSegmentation import logger
from BrainTumorSegmentation.entity.config_entity import ModelEvaluationConfig
from BrainTumorSegmentation.utils.common import save_json


class BrainTumorDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for preprocessed brain tumor segmentation data.
    """

    def __init__(self, patient_list, data_root):
        """
        Initialize the brain tumor dataset.

        Args:
            patient_list (List[str]): List of patient IDs
            data_root (str): Root directory of the preprocessed data for a specific split
        """
        self.patient_list = patient_list
        self.data_root = data_root

    def __len__(self):
        """Return the number of patients in the dataset."""
        return len(self.patient_list)

    def __getitem__(self, idx):
        """
        Get the preprocessed multimodal images and segmentation mask for a patient.

        Args:
            idx (int): Index of the patient in the list

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: Image tensor (4 channels), mask tensor, and patient ID
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

        return (
            image,
            mask.squeeze(0),
            pid,
        )  # Return as (C, H, W, D), (H, W, D), and patient ID


class ModelEvaluation:
    """
    Model Evaluation class to handle the evaluation of brain tumor segmentation model.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation class with the configuration.

        Args:
            config (ModelEvaluationConfig): Configuration for model evaluation.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Evaluation using device: {self.device}")

        # Create output directory
        os.makedirs(self.config.root_dir, exist_ok=True)
        # Create visualization directory
        os.makedirs(os.path.join(self.config.root_dir, "visualizations"), exist_ok=True)

    def _load_model(self):
        """
        Load the trained model from the specified path.

        Returns:
            torch.nn.Module: Loaded model
        """
        logger.info(f"Loading model from: {self.config.model_path}")
        try:
            # Try loading as TorchScript model first
            try:
                model = torch.jit.load(self.config.model_path)
                model.to(self.device)
            except:
                # If that fails, try loading as state dict
                from monai.networks.nets import SwinUNETR

                model = SwinUNETR(
                    img_size=self.config.img_size,
                    in_channels=4,  # 4 modalities
                    out_channels=4,  # 4 classes (background + 3 tumor types)
                    feature_size=self.config.feature_size,
                    use_checkpoint=False,
                ).to(self.device)
                model.load_state_dict(
                    torch.load(self.config.model_path, map_location=self.device)
                )

            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _check_patient_files(self, patient, data_dir):
        """
        Check if a patient has all required modality files.

        Args:
            patient (str): Patient ID
            data_dir (str): Data directory path

        Returns:
            bool: True if all files exist, False otherwise
        """
        base = os.path.join(data_dir, patient)
        files = [
            f"{patient}-{mod}.nii.gz" for mod in ["t1n", "t1c", "t2w", "t2f", "seg"]
        ]
        return all(os.path.exists(os.path.join(base, f)) for f in files)

    def _get_test_dataloader(self):
        """
        Create DataLoader for the test set.

        Returns:
            DataLoader: Test data loader
        """
        # Check if using preprocessed structure or original structure
        if os.path.exists(os.path.join(self.config.data_path, "test")):
            # Using preprocessed structure with test folder
            test_patients = [
                f
                for f in os.listdir(os.path.join(self.config.data_path, "test"))
                if os.path.isdir(os.path.join(self.config.data_path, "test", f))
            ]
            data_root = os.path.join(self.config.data_path, "test")
        else:
            # Original structure - filter and select test patients (last 15%)
            all_patients = [
                f
                for f in os.listdir(self.config.data_path)
                if os.path.isdir(os.path.join(self.config.data_path, f))
            ]
            valid_patients = [
                p
                for p in all_patients
                if self._check_patient_files(p, self.config.data_path)
            ]

            # Ensure consistent ordering for splits by sorting
            valid_patients.sort()

            # Take the last 15% as test set
            test_patients = valid_patients[int(0.85 * len(valid_patients)) :]
            data_root = self.config.data_path

        logger.info(f"Found {len(test_patients)} test patients")

        # Create dataset and dataloader
        test_ds = BrainTumorDataset(test_patients, data_root)

        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return test_loader, test_patients

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

    def _save_prediction_image(
        self, image, true_mask, pred_mask, patient_id, slice_idx
    ):
        """
        Create and save a visualization of the prediction results.

        Args:
            image (torch.Tensor): Input image (first modality)
            true_mask (torch.Tensor): Ground truth segmentation
            pred_mask (torch.Tensor): Predicted segmentation
            patient_id (str): Patient identifier
            slice_idx (int): Slice index to visualize
        """
        # Create figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Extract middle slice for visualization
        image_slice = (
            image[0, :, :, slice_idx].cpu().numpy()
        )  # Use first modality (T1n)
        true_slice = true_mask[:, :, slice_idx].cpu().numpy()
        pred_slice = pred_mask[:, :, slice_idx].cpu().numpy()

        # Display images
        axes[0].imshow(image_slice, cmap="gray")
        axes[0].set_title("T1n Image")
        axes[0].axis("off")

        axes[1].imshow(true_slice, cmap="inferno")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(image_slice, cmap="gray")
        axes[2].imshow(pred_slice, cmap="inferno", alpha=0.5)
        axes[2].set_title("Predicted Mask Overlay")
        axes[2].axis("off")

        # Save figure
        fig.suptitle(f"Patient: {patient_id}")
        plt.tight_layout()

        # Save the visualization
        viz_dir = os.path.join(self.config.root_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(
            os.path.join(viz_dir, f"{patient_id}_slice{slice_idx}.png"), dpi=300
        )
        plt.close(fig)

    def evaluate_model(self):
        """
        Evaluate the model on the test dataset.
        """
        # Load model
        model = self._load_model()

        # Get test dataloader
        test_loader, _ = self._get_test_dataloader()

        # Initialize metrics storage
        all_metrics = []

        # Setup loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize overall counters
        overall_loss = 0.0
        overall_correct = 0
        overall_total = 0

        # Evaluation loop
        logger.info("Starting model evaluation...")
        with torch.no_grad():
            for idx, (imgs, masks, pid) in enumerate(
                tqdm(test_loader, desc="Evaluating")
            ):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                # Forward pass
                outputs = model(imgs)

                # Calculate loss
                loss = criterion(outputs, masks)

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                # Calculate accuracy
                correct = (preds == masks).sum().item()
                total = masks.numel()
                accuracy = correct / total if total > 0 else 0

                # Update overall metrics
                overall_loss += loss.item()
                overall_correct += correct
                overall_total += total

                # Log per-patient results
                for i in range(len(imgs)):
                    patient_id = pid[i] if isinstance(pid, list) else pid
                    logger.info(
                        f"Patient: {patient_id} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}"
                    )

                    # Store metrics for this patient
                    patient_metrics = {
                        "patient_id": patient_id,
                        "loss": loss.item(),
                        "accuracy": accuracy,
                    }
                    all_metrics.append(patient_metrics)

                # Save visualization for each patient
                for i in range(len(imgs)):
                    # Get prediction
                    pred_mask = preds[i]
                    middle_slice = imgs.shape[-1] // 2

                    # Get patient ID
                    patient_id = pid[i] if isinstance(pid, list) else pid

                    # Save visualization
                    self._save_prediction_image(
                        imgs[i],
                        masks[i],
                        pred_mask,
                        patient_id,
                        middle_slice,
                    )

        # Calculate overall metrics
        avg_loss = overall_loss / len(test_loader)
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0

        # Create DataFrame from all metrics
        metrics_df = pd.DataFrame(all_metrics)

        # Calculate and compile overall metrics
        overall_metrics = {
            "avg_loss": avg_loss,
            "accuracy": overall_accuracy,
            "std_accuracy": metrics_df["accuracy"].std() if len(metrics_df) > 0 else 0,
        }

        # Save metrics to CSV
        metrics_df.to_csv(
            os.path.join(self.config.root_dir, "patient_metrics.csv"), index=False
        )

        # Log overall results
        logger.info(
            f"✅ Evaluation completed. Results saved to: {self.config.root_dir}"
        )
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(
            f"Overall Accuracy: {overall_accuracy:.4f} ± {overall_metrics['std_accuracy']:.4f}"
        )

        return overall_metrics

    def run(self):
        """
        Run the model evaluation pipeline.
        """
        logger.info("Starting model evaluation...")
        overall_metrics = self.evaluate_model()

        # save overall metrics to json
        metrics_path = os.path.join(self.config.root_dir, "overall_metrics.json")
        save_json(Path(metrics_path), overall_metrics)
