"""Test deterministic behavior of training and embedding extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from biofm.dataio.toydata import generate_toy_dataset
from biofm.models.foundation import BioFMModel
from biofm.training.loop import LoopConfig, train_model
from biofm.utils.determinism import set_all_seeds, get_deterministic_dataloader_kwargs


def test_deterministic_training(tmp_path: Path) -> None:
    """Test that training produces deterministic results with fixed seed."""
    
    # Set seed for deterministic behavior
    set_all_seeds(42)
    
    # Generate toy dataset
    data_dir = tmp_path / "data"
    bundle = generate_toy_dataset(
        data_dir=data_dir,
        n_samples=8,  # Small dataset for fast test
        image_size=32,
        n_genes=16,
        seed=42,
    )
    
    # Create dataloader with deterministic behavior
    from biofm.dataio.loader import create_dataloader
    
    torch = pytest.importorskip("torch")
    
    dataloader_kwargs = get_deterministic_dataloader_kwargs()
    dataloader = create_dataloader(
        bundle=bundle,
        batch_size=4,
        shuffle=False,  # Deterministic order
        **dataloader_kwargs
    )
    
    # Create model and configure for minimal training
    model = BioFMModel(
        image_size=32,
        n_genes=16,
        vision_hidden_size=128,
        rna_hidden_size=64,
        projection_size=32,
    )
    
    # Training configuration for exactly 20 steps
    config = LoopConfig(
        epochs=10,  # Should give us about 20 steps with 8 samples, batch_size=4
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        amp_mode="off",  # Disable AMP for deterministic behavior
        checkpoint_dir=tmp_path / "checkpoints"
    )
    
    # Run training
    device = torch.device("cpu")  # CPU only for deterministic behavior
    set_all_seeds(42)  # Reset seed before training
    summary = train_model(model, dataloader, config, device=device)
    
    # Extract embeddings from the first 10 images
    model.eval()
    embeddings_list = []
    with torch.no_grad():
        step_count = 0
        for batch in dataloader:
            if step_count >= 10:
                break
            pixel_values = batch["pixel_values"].to(device)
            # Get vision embeddings (before projection)
            vision_embeds = model.vision_encoder(pixel_values)
            embeddings_list.append(vision_embeds.cpu().numpy())
            step_count += 1
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings_list, axis=0)[:10]  # First 10 images
    
    # Compute SHA256 hash of the embeddings
    embeddings_bytes = embeddings.astype(np.float32).tobytes()
    sha256_hash = hashlib.sha256(embeddings_bytes).hexdigest()
    
    # This is the expected hash for the deterministic run
    # Note: This will need to be updated if model architecture changes
    # For now, just verify the hash is consistent by running twice
    
    # Second run to verify consistency
    set_all_seeds(42)
    
    # Regenerate data with same seed
    bundle2 = generate_toy_dataset(
        data_dir=tmp_path / "data2",
        n_samples=8,
        image_size=32,
        n_genes=16,
        seed=42,
    )
    
    dataloader2 = create_dataloader(
        bundle=bundle2,
        batch_size=4,
        shuffle=False,
        **dataloader_kwargs
    )
    
    # Create new model with same architecture
    model2 = BioFMModel(
        image_size=32,
        n_genes=16,
        vision_hidden_size=128,
        rna_hidden_size=64,
        projection_size=32,
    )
    
    # Train with same config
    config2 = LoopConfig(
        epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        amp_mode="off",
        checkpoint_dir=tmp_path / "checkpoints2"
    )
    
    summary2 = train_model(model2, dataloader2, config2, device=device)
    
    # Extract embeddings again
    model2.eval()
    embeddings_list2 = []
    with torch.no_grad():
        step_count = 0
        for batch in dataloader2:
            if step_count >= 10:
                break
            pixel_values = batch["pixel_values"].to(device)
            vision_embeds = model2.vision_encoder(pixel_values)
            embeddings_list2.append(vision_embeds.cpu().numpy())
            step_count += 1
    
    embeddings2 = np.concatenate(embeddings_list2, axis=0)[:10]
    
    # Compute hash again
    embeddings_bytes2 = embeddings2.astype(np.float32).tobytes()
    sha256_hash2 = hashlib.sha256(embeddings_bytes2).hexdigest()
    
    # Assert that both runs produce identical results
    assert sha256_hash == sha256_hash2, (
        f"Deterministic training failed! "
        f"First hash: {sha256_hash}, Second hash: {sha256_hash2}"
    )
    
    # For reference, log the golden hash value
    print(f"Golden SHA256 hash for deterministic embeddings: {sha256_hash}")
    
    # Verify that we actually completed training
    assert len(summary.losses) > 0
    assert len(summary2.losses) > 0
    assert summary.losses == summary2.losses  # Losses should also be identical


def test_set_all_seeds() -> None:
    """Test that set_all_seeds function works correctly."""
    
    torch = pytest.importorskip("torch")
    
    # Test that same seed produces same results
    set_all_seeds(123)
    random_val1 = torch.rand(1).item()
    np_val1 = np.random.rand()
    
    set_all_seeds(123)
    random_val2 = torch.rand(1).item()
    np_val2 = np.random.rand()
    
    assert random_val1 == random_val2
    assert np_val1 == np_val2
    
    # Test that different seeds produce different results
    set_all_seeds(456)
    random_val3 = torch.rand(1).item()
    np_val3 = np.random.rand()
    
    assert random_val1 != random_val3
    assert np_val1 != np_val3