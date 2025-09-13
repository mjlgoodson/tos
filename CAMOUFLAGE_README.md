# Camouflage Pattern Synthesis - Provisional Version 4

This repository contains an enhanced camouflage pattern synthesis system with multi-scale macro fields, adaptive parameters, and large region suppression for improved pattern concealment.

## Features

The enhanced synthesis system includes:

1. **Multi-scale macro field logic**: Generates macro fields at multiple scales and sums them for better pattern breakup
2. **Adaptive macro density and blob size**: Parameters responsive to environmental spatial statistics
3. **Enhanced patch suppression loss**: Uses connected component analysis to penalize large connected regions
4. **CLI parameter control**: Full control over synthesis parameters via command-line arguments

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies:
- numpy>=1.20.0
- scipy>=1.7.0
- PyYAML>=6.0
- Pillow>=8.0.0

## Usage

### Basic Usage

Generate a camouflage pattern with default parameters:

```bash
python synthesize_camo_pattern_prov4.py
```

### Advanced Usage

Use environmental profile and custom parameters:

```bash
python synthesize_camo_pattern_prov4.py \
  --tile_size 512 \
  --profile profile_plus.yaml \
  --large_region_lambda 0.15 \
  --iterations 200 \
  --output forest_camo.png
```

### CLI Parameters

- `--tile_size`: Size of output tile in pixels (default: 512)
- `--profile`: Path to environmental profile YAML file
- `--large_region_lambda`: Strength of large region penalty (default: 0.1)
- `--iterations`: Number of optimization iterations (default: 100)  
- `--output`: Output file path (default: camo_pattern.png)

## Environmental Profile

The `profile_plus.yaml` file defines environmental spatial statistics:

```yaml
# Dominant spatial scales for adaptive parameter tuning
dominant_scales_px:
  - 8      # Fine texture scale
  - 16     # Medium feature scale  
  - 32     # Large feature scale
  - 64     # Macro feature scale

environment_type: "forest"
lighting_conditions: "mixed"
base_macro_density: 0.03
max_region_size_percent: 8.0
```

When provided, the system automatically adapts:
- `macro_density` based on average dominant scale
- `macro_r_px` based on largest dominant scale
- Multi-scale blob generation using the dominant scales

## Key Enhancements

### 1. Multi-Scale Macro Fields

The `legacy_macro_signed_blobs` function now generates macro fields at multiple scales:

- Large-scale blobs for primary pattern structure
- Medium-scale blobs for secondary features
- Small-scale blobs for fine texture breakup
- Optional tileable noise for additional fine detail

### 2. Adaptive Parameters

Parameters automatically adapt to environmental statistics:

- Macro density decreases for environments with larger dominant scales
- Blob radius scales with the largest environmental feature size
- Multi-scale generation uses environment-appropriate scales

### 3. Large Region Suppression

Enhanced loss function penalizes large connected regions:

- Uses `scipy.ndimage.label` for connected component analysis
- Penalizes regions exceeding 8% of tile area (configurable)
- Gradient computation ensures regions are broken up during optimization

## Testing

Run the test suite:

```bash
python test_synthesis.py
```

Tests validate:
- Multi-scale macro field generation
- Adaptive parameter setting from profiles
- Large region loss computation
- Connected component analysis
- Full synthesis pipeline

## Implementation Details

### Multi-Scale Generation

```python
# Generate at multiple scales with decreasing amplitudes
scales = [macro_r_px, macro_r_px // 2, macro_r_px // 4]
amplitudes = [1.0, 0.6, 0.3]

# Combine fields
for scale, amplitude in zip(scales, amplitudes):
    field += generate_single_scale_blobs(scale, amplitude)

# Add fine-scale noise
field += generate_tileable_noise(min_scale // 2, min_amplitude * 0.5)
```

### Large Region Loss

```python
# Find connected components in high-probability areas
prob_field = sigmoid(pattern)
high_prob_mask = prob_field > threshold
labeled_regions, num_regions = scipy.ndimage.label(high_prob_mask)

# Penalize oversized regions
for region in oversized_regions:
    penalty += (region_size - threshold_size)² / tile_area²
```

### Adaptive Parameters

```python
# Adapt to environmental scales
avg_scale = mean(dominant_scales_px)
macro_density = max(0.01, 0.05 - avg_scale / 1000.0)

max_scale = max(dominant_scales_px)  
macro_r_px = int(max_scale * 0.8)
```

## Output

The system generates grayscale camouflage patterns that:

- Avoid large contiguous regions through suppression loss
- Include multi-scale features matching environmental statistics
- Maintain coherent visual structure while breaking up clustering
- Are tileable for seamless application

Pattern files are saved as PNG images with normalized 8-bit values.