#!/usr/bin/env python3
"""
Test script for camouflage pattern synthesis functionality.
Tests the key enhancements added to the synthesis system.
"""

import sys
import numpy as np
from pathlib import Path
import tempfile
import os

# Add current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthesize_camo_pattern_prov4 import CamouflagePatternSynthesizer


def test_multi_scale_macro_field():
    """Test multi-scale macro field generation."""
    print("Testing multi-scale macro field generation...")
    
    synthesizer = CamouflagePatternSynthesizer(tile_size=128)
    
    # Test with multiple scales
    scales = [20, 10, 5]
    amplitudes = [1.0, 0.5, 0.25]
    
    field = synthesizer.legacy_macro_signed_blobs(128, 128, scales, amplitudes)
    
    # Basic validation
    assert field.shape == (128, 128), f"Expected shape (128, 128), got {field.shape}"
    assert not np.all(field == 0), "Field should not be all zeros"
    assert np.isfinite(field).all(), "Field should contain only finite values"
    
    print("✓ Multi-scale macro field test passed")


def test_adaptive_parameters():
    """Test adaptive parameter setting from environmental profile."""
    print("Testing adaptive parameter setting...")
    
    # Test with profile
    synthesizer = CamouflagePatternSynthesizer(
        tile_size=128,
        profile_path="profile_plus.yaml"
    )
    
    # Check that dominant scales were loaded
    assert hasattr(synthesizer, 'dominant_scales_px'), "Should have dominant_scales_px"
    assert len(synthesizer.dominant_scales_px) > 0, "Should have at least one dominant scale"
    
    # Check adaptive parameters were set
    assert synthesizer.macro_density > 0, "Macro density should be positive"
    assert synthesizer.macro_r_px > 0, "Macro radius should be positive"
    
    print("✓ Adaptive parameters test passed")


def test_large_region_loss():
    """Test large region suppression loss computation."""
    print("Testing large region loss computation...")
    
    synthesizer = CamouflagePatternSynthesizer(tile_size=64, large_region_lambda=0.1)
    
    # Create a pattern with a large connected region
    pattern = np.zeros((64, 64))
    # Create a large high-value region (should trigger penalty)
    pattern[16:48, 16:48] = 2.0  # High values that will create large connected region
    
    # Test loss computation
    loss = synthesizer._compute_large_region_loss(pattern)
    
    assert loss > 0, "Should have positive loss for large connected region"
    
    # Test with no large regions
    small_pattern = np.random.randn(64, 64) * 0.1  # Small random values
    small_loss = synthesizer._compute_large_region_loss(small_pattern)
    
    assert loss > small_loss, "Large region should have higher loss than small regions"
    
    print("✓ Large region loss test passed")


def test_connected_component_analysis():
    """Test connected component analysis functionality."""
    print("Testing connected component analysis...")
    
    synthesizer = CamouflagePatternSynthesizer(tile_size=32)
    
    # Create a test pattern with a large connected component
    # 32x32 = 1024 pixels, 8% = ~82 pixels, so create a 10x10 region (100 pixels)
    pattern = np.zeros((32, 32))
    pattern[5:15, 5:15] = 3.0    # Large region (100 pixels > 82 pixel threshold)
    
    # Debug: Check sizes
    tile_area = 32 * 32
    size_threshold = int(0.08 * tile_area)
    region_size = 10 * 10
    print(f"Tile area: {tile_area}, Size threshold: {size_threshold}, Region size: {region_size}")
    
    # Compute gradient
    gradient = synthesizer._compute_region_gradient(pattern)
    
    print(f"Gradient max: {gradient.max()}")
    print(f"Gradient min: {gradient.min()}")
    print(f"Gradient nonzero count: {(gradient != 0).sum()}")
    
    assert gradient.shape == pattern.shape, "Gradient should have same shape as pattern"
    assert np.any(gradient != 0), "Gradient should be non-zero for regions above threshold"
    
    print("✓ Connected component analysis test passed")


def test_optimization_step():
    """Test optimization step with enhanced loss."""
    print("Testing optimization step...")
    
    synthesizer = CamouflagePatternSynthesizer(tile_size=32, large_region_lambda=0.1)
    
    # Create simple target and current patterns
    target = np.random.randn(32, 32) * 0.5
    current = np.random.randn(32, 32) * 0.5
    
    # Run optimization step
    loss, updated = synthesizer._optimize_at_res(target, current, learning_rate=0.01)
    
    assert isinstance(loss, (float, np.floating)), "Loss should be a float"
    assert loss >= 0, "Loss should be non-negative"
    assert updated.shape == current.shape, "Updated pattern should have same shape"
    assert not np.array_equal(updated, current), "Pattern should be updated"
    
    print("✓ Optimization step test passed")


def test_cli_parameters():
    """Test that CLI parameters are properly handled."""
    print("Testing CLI parameter integration...")
    
    # Test with custom large_region_lambda
    synthesizer = CamouflagePatternSynthesizer(
        tile_size=64,
        large_region_lambda=0.2
    )
    
    assert synthesizer.large_region_lambda == 0.2, "Large region lambda should be set correctly"
    
    print("✓ CLI parameters test passed")


def test_full_synthesis():
    """Test full pattern synthesis pipeline."""
    print("Testing full synthesis pipeline...")
    
    synthesizer = CamouflagePatternSynthesizer(
        tile_size=64,
        profile_path="profile_plus.yaml",
        large_region_lambda=0.1
    )
    
    # Run synthesis with few iterations for speed
    pattern = synthesizer.synthesize_pattern(iterations=5)
    
    assert pattern.shape == (64, 64), f"Expected shape (64, 64), got {pattern.shape}"
    assert np.isfinite(pattern).all(), "Pattern should contain only finite values"
    assert not np.all(pattern == 0), "Pattern should not be all zeros"
    
    print("✓ Full synthesis test passed")


def main():
    """Run all tests."""
    print("Running camouflage synthesis tests...\n")
    
    try:
        test_multi_scale_macro_field()
        test_adaptive_parameters() 
        test_large_region_loss()
        test_connected_component_analysis()
        test_optimization_step()
        test_cli_parameters()
        test_full_synthesis()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)