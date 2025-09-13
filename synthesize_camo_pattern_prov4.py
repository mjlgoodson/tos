#!/usr/bin/env python3
"""
Camouflage Pattern Synthesis - Provisional Version 4

Enhanced synthesis with multi-scale macro fields, adaptive parameters,
and large region suppression for improved concealment patterns.
"""

import argparse
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import scipy.ndimage
from scipy.ndimage import label
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class CamouflagePatternSynthesizer:
    """Main class for synthesizing camouflage patterns with advanced features."""
    
    def __init__(self, 
                 tile_size: int = 512,
                 profile_path: Optional[str] = None,
                 large_region_lambda: float = 0.1):
        """
        Initialize the synthesizer.
        
        Args:
            tile_size: Size of the output tile in pixels
            profile_path: Path to environmental profile YAML file
            large_region_lambda: Weight for large region suppression loss
        """
        self.tile_size = tile_size
        self.large_region_lambda = large_region_lambda
        self.profile_data = None
        
        if profile_path:
            self.load_environmental_profile(profile_path)
            
        # Default parameters (can be overridden by profile)
        self.macro_density = 0.02
        self.macro_r_px = 20
        self.dominant_scales_px = [8, 16, 32]
        
        # Apply adaptive parameters if profile is available
        self._setup_adaptive_parameters()
    
    def load_environmental_profile(self, profile_path: str) -> None:
        """Load environmental profile configuration from YAML file."""
        try:
            with open(profile_path, 'r') as f:
                self.profile_data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Profile file {profile_path} not found. Using defaults.")
            self.profile_data = None
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing profile file: {e}. Using defaults.")
            self.profile_data = None
    
    def _setup_adaptive_parameters(self) -> None:
        """Setup adaptive parameters based on environmental profile."""
        if not self.profile_data:
            return
            
        # Use dominant scales from profile if available
        if 'dominant_scales_px' in self.profile_data:
            self.dominant_scales_px = self.profile_data['dominant_scales_px']
            
            # Adapt macro density based on dominant scales
            avg_scale = np.mean(self.dominant_scales_px)
            self.macro_density = max(0.01, 0.05 - avg_scale / 1000.0)
            
            # Adapt blob radius based on largest dominant scale
            max_scale = max(self.dominant_scales_px)
            self.macro_r_px = int(max_scale * 0.8)
    
    def legacy_macro_signed_blobs(self, 
                                  width: int, 
                                  height: int, 
                                  scales: Optional[List[int]] = None,
                                  amplitudes: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate multi-scale macro field with signed blobs at various scales.
        
        Enhanced version that sums macro fields at multiple scales for better
        pattern breakup and reduced clustering.
        
        Args:
            width: Width of the field
            height: Height of the field  
            scales: List of blob scales (radii) to generate
            amplitudes: Corresponding amplitudes for each scale
            
        Returns:
            Combined macro field as numpy array
        """
        if scales is None:
            # Use adaptive scales based on dominant_scales_px
            scales = [self.macro_r_px, self.macro_r_px // 2, self.macro_r_px // 4]
        
        if amplitudes is None:
            # Decreasing amplitudes for smaller scales
            amplitudes = [1.0, 0.5, 0.25]
            
        # Ensure we have matching scales and amplitudes
        min_len = min(len(scales), len(amplitudes))
        scales = scales[:min_len]
        amplitudes = amplitudes[:min_len]
        
        # Initialize combined field
        combined_field = np.zeros((height, width), dtype=np.float32)
        
        for scale, amplitude in zip(scales, amplitudes):
            # Generate blob field for this scale
            field = self._generate_single_scale_blobs(width, height, scale, amplitude)
            combined_field += field
            
        # Add tileable noise for additional fine-scale breakup
        if len(scales) > 1:  # Only add noise if we have multiple scales
            noise_field = self._generate_tileable_noise(width, height, 
                                                      scale=min(scales) // 2,
                                                      amplitude=min(amplitudes) * 0.5)
            combined_field += noise_field
            
        return combined_field
    
    def _generate_single_scale_blobs(self, 
                                   width: int, 
                                   height: int, 
                                   blob_radius: int,
                                   amplitude: float) -> np.ndarray:
        """Generate signed blobs at a single scale."""
        field = np.zeros((height, width), dtype=np.float32)
        
        # Calculate number of blobs based on adaptive density
        area = width * height
        n_blobs = int(area * self.macro_density * amplitude)
        
        # Generate blob centers
        blob_centers = np.random.rand(n_blobs, 2)
        blob_centers[:, 0] *= width
        blob_centers[:, 1] *= height
        
        # Generate blob strengths (signed)
        blob_strengths = np.random.choice([-1, 1], size=n_blobs) * amplitude
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        for i, (center, strength) in enumerate(zip(blob_centers, blob_strengths)):
            cx, cy = center
            
            # Calculate distance from blob center
            dx = x - cx
            dy = y - cy
            
            # Handle tiling by considering wrapped distances
            dx = np.minimum(np.abs(dx), width - np.abs(dx))
            dy = np.minimum(np.abs(dy), height - np.abs(dy))
            
            dist_sq = dx**2 + dy**2
            
            # Apply Gaussian blob
            blob_mask = dist_sq <= (blob_radius**2)
            blob_values = strength * np.exp(-dist_sq / (2 * (blob_radius/3)**2))
            
            field += blob_values * blob_mask
            
        return field
    
    def _generate_tileable_noise(self, 
                               width: int, 
                               height: int, 
                               scale: int, 
                               amplitude: float) -> np.ndarray:
        """Generate tileable noise field for fine-scale pattern breakup."""
        # Use a simple approach with multiple octaves of noise
        noise_field = np.zeros((height, width), dtype=np.float32)
        
        # Generate noise at different frequencies
        frequencies = [1, 2, 4]
        for freq in frequencies:
            current_scale = max(1, scale // freq)
            freq_amplitude = amplitude / freq
            
            # Generate random noise at reduced resolution
            noise_h = max(1, height // current_scale)
            noise_w = max(1, width // current_scale)
            
            noise = np.random.randn(noise_h, noise_w) * freq_amplitude
            
            # Upsample to full resolution with smoothing
            if noise_h < height or noise_w < width:
                from scipy.ndimage import zoom
                zoom_factors = (height / noise_h, width / noise_w)
                noise = zoom(noise, zoom_factors, order=1)
            
            noise_field += noise
            
        return noise_field
    
    def _optimize_at_res(self, 
                        target_pattern: np.ndarray,
                        current_pattern: np.ndarray,
                        learning_rate: float = 0.01) -> Tuple[float, np.ndarray]:
        """
        Optimization step with enhanced patch suppression loss.
        
        Args:
            target_pattern: Target pattern to match
            current_pattern: Current synthesized pattern
            learning_rate: Learning rate for optimization
            
        Returns:
            Tuple of (total_loss, updated_pattern)
        """
        # Basic coherent patch loss
        coherent_loss = np.mean((current_pattern - target_pattern) ** 2)
        
        # Enhanced large region suppression loss
        large_region_loss = self._compute_large_region_loss(current_pattern)
        
        # Combine losses
        total_loss = coherent_loss + self.large_region_lambda * large_region_loss
        
        # Compute gradients and update pattern
        pattern_gradient = 2 * (current_pattern - target_pattern)
        
        # Add gradient from large region loss
        region_gradient = self._compute_region_gradient(current_pattern)
        pattern_gradient += self.large_region_lambda * region_gradient
        
        # Update pattern
        updated_pattern = current_pattern - learning_rate * pattern_gradient
        
        return total_loss, updated_pattern
    
    def _compute_large_region_loss(self, pattern: np.ndarray) -> float:
        """
        Compute loss for large connected regions using connected component analysis.
        
        Args:
            pattern: Pattern field to analyze
            
        Returns:
            Loss value for large connected regions
        """
        # Convert pattern to probability field (sigmoid activation)
        prob_field = 1 / (1 + np.exp(-np.clip(pattern, -10, 10)))  # Clip for numerical stability
        
        # Threshold to get high-probability areas
        threshold = 0.6  # Lower threshold to be more sensitive
        high_prob_mask = prob_field > threshold
        
        # Find connected components
        labeled_regions, num_regions = label(high_prob_mask)
        
        if num_regions == 0:
            return 0.0
            
        # Calculate region sizes
        region_sizes = np.bincount(labeled_regions.ravel())[1:]  # Skip background (0)
        
        # Calculate threshold size (8% of tile area)
        tile_area = pattern.shape[0] * pattern.shape[1]
        size_threshold = int(0.08 * tile_area)
        
        # Compute penalty for oversized regions
        oversized_regions = region_sizes > size_threshold
        if not np.any(oversized_regions):
            return 0.0
            
        # Penalty increases quadratically with size excess
        excess_sizes = region_sizes[oversized_regions] - size_threshold
        penalty = np.sum((excess_sizes / tile_area) ** 2)
        
        return penalty
    
    def _compute_region_gradient(self, pattern: np.ndarray) -> np.ndarray:
        """
        Compute gradient for large region suppression loss.
        
        Args:
            pattern: Pattern field
            
        Returns:
            Gradient array
        """
        gradient = np.zeros_like(pattern)
        
        # Convert to probability field
        prob_field = 1 / (1 + np.exp(-np.clip(pattern, -10, 10)))  # Clip for numerical stability
        
        # Find high-probability regions
        threshold = 0.6  # Lower threshold to catch more regions
        high_prob_mask = prob_field > threshold
        
        # Find connected components
        labeled_regions, num_regions = label(high_prob_mask)
        
        if num_regions == 0:
            return gradient
            
        # Calculate region sizes
        region_sizes = np.bincount(labeled_regions.ravel())[1:]
        
        # Size threshold
        tile_area = pattern.shape[0] * pattern.shape[1]
        size_threshold = int(0.08 * tile_area)
        
        # Add gradient for oversized regions
        for region_id in range(1, num_regions + 1):
            region_size = region_sizes[region_id - 1]
            if region_size > size_threshold:
                region_mask = labeled_regions == region_id
                
                # Gradient magnitude proportional to size excess
                excess = region_size - size_threshold
                grad_magnitude = 2 * excess / (tile_area ** 2)
                
                # Apply gradient to suppress this region
                # Derivative of sigmoid: sigmoid * (1 - sigmoid)
                sigmoid_deriv = prob_field * (1 - prob_field)
                gradient[region_mask] -= grad_magnitude * sigmoid_deriv[region_mask]
                
        return gradient
    
    def synthesize_pattern(self, 
                          iterations: int = 100,
                          output_path: Optional[str] = None) -> np.ndarray:
        """
        Main synthesis method that creates the camouflage pattern.
        
        Args:
            iterations: Number of optimization iterations
            output_path: Optional path to save the result
            
        Returns:
            Synthesized pattern as numpy array
        """
        print(f"Synthesizing camouflage pattern ({self.tile_size}x{self.tile_size})")
        print(f"Using macro_density={self.macro_density:.4f}, macro_r_px={self.macro_r_px}")
        print(f"Dominant scales: {self.dominant_scales_px}")
        print(f"Large region lambda: {self.large_region_lambda}")
        
        # Generate initial macro field with multi-scale approach
        macro_field = self.legacy_macro_signed_blobs(
            self.tile_size, 
            self.tile_size,
            scales=[self.macro_r_px, self.macro_r_px // 2, self.macro_r_px // 4],
            amplitudes=[1.0, 0.6, 0.3]
        )
        
        # Initialize pattern
        current_pattern = macro_field.copy()
        
        # Create a synthetic target (in practice this would be environment-based)
        target_pattern = self._generate_target_pattern()
        
        print(f"Starting optimization with {iterations} iterations...")
        
        # Optimization loop
        for iteration in range(iterations):
            loss, current_pattern = self._optimize_at_res(
                target_pattern, current_pattern, learning_rate=0.01
            )
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
        
        print(f"Optimization complete. Final loss: {loss:.6f}")
        
        # Save result if path provided
        if output_path:
            self._save_pattern(current_pattern, output_path)
            
        return current_pattern
    
    def _generate_target_pattern(self) -> np.ndarray:
        """Generate a synthetic target pattern for demonstration."""
        # Create a target with varied spatial frequencies
        target = np.zeros((self.tile_size, self.tile_size))
        
        y, x = np.ogrid[:self.tile_size, :self.tile_size]
        
        # Add multiple frequency components
        for scale in self.dominant_scales_px:
            freq = 2 * np.pi / scale
            target += 0.3 * np.sin(freq * x) * np.cos(freq * y)
            target += 0.2 * np.cos(freq * x + np.pi/4) * np.sin(freq * y + np.pi/3)
        
        # Add some randomness
        target += 0.1 * np.random.randn(self.tile_size, self.tile_size)
        
        return target
    
    def _save_pattern(self, pattern: np.ndarray, output_path: str) -> None:
        """Save pattern to file."""
        # Normalize pattern for saving
        normalized = ((pattern - pattern.min()) / 
                     (pattern.max() - pattern.min()) * 255).astype(np.uint8)
        
        # Try to save as image if PIL is available
        try:
            from PIL import Image
            Image.fromarray(normalized).save(output_path)
            print(f"Pattern saved to {output_path}")
        except ImportError:
            # Fallback to numpy save
            np.save(output_path.replace('.png', '.npy'), pattern)
            print(f"Pattern saved to {output_path.replace('.png', '.npy')} (numpy format)")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description='Synthesize camouflage patterns')
    
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Size of output tile (default: 512)')
    
    parser.add_argument('--profile', type=str, default=None,
                       help='Path to environmental profile YAML file')
    
    parser.add_argument('--large_region_lambda', type=float, default=0.1,
                       help='Strength of large region penalty (default: 0.1)')
    
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of optimization iterations (default: 100)')
    
    parser.add_argument('--output', type=str, default='camo_pattern.png',
                       help='Output file path (default: camo_pattern.png)')
    
    args = parser.parse_args()
    
    # Create synthesizer
    synthesizer = CamouflagePatternSynthesizer(
        tile_size=args.tile_size,
        profile_path=args.profile,
        large_region_lambda=args.large_region_lambda
    )
    
    # Synthesize pattern
    pattern = synthesizer.synthesize_pattern(
        iterations=args.iterations,
        output_path=args.output
    )
    
    print("Pattern synthesis complete!")


if __name__ == '__main__':
    main()