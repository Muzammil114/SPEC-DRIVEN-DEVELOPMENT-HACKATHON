---
sidebar_position: 3
---

# Synthetic Data Generation for AI Training

## Learning Objectives
- Understand the importance of synthetic data in robotics AI
- Learn to configure synthetic data generation in Isaac Sim
- Generate various types of synthetic data (images, point clouds, semantic segmentation)
- Apply domain randomization techniques
- Evaluate synthetic data quality for real-world transfer

## Introduction to Synthetic Data in Robotics

Synthetic data generation is a crucial capability in modern robotics AI development. It allows for:

- **Large-scale dataset creation** without real-world data collection costs
- **Controlled environment simulation** with known ground truth
- **Edge case generation** that might be rare in real data
- **Domain randomization** to improve model robustness
- **Safety during development** without physical robot risks

In NVIDIA Isaac Sim, synthetic data generation leverages the high-fidelity rendering pipeline to create realistic training data for perception and navigation systems.

## Synthetic Data Types

### 1. RGB Images
High-quality color images for visual perception tasks:

```python
import omni
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.sensor import Camera

# Create a camera for RGB image capture
camera = Camera(
    prim_path="/World/Camera",
    frequency=30,  # Hz
    resolution=(640, 480)
)

# Capture RGB images
rgb_data = camera.get_rgb()
```

### 2. Depth Maps
Depth information for 3D perception and navigation:

```python
# Capture depth information
depth_data = camera.get_depth()

# Convert depth to point cloud
import numpy as np
def depth_to_point_cloud(depth_image, camera_intrinsics):
    """Convert depth image to point cloud"""
    height, width = depth_image.shape
    points = []

    for v in range(height):
        for u in range(width):
            z = depth_image[v, u]
            if z < 1000:  # Valid depth
                x = (u - camera_intrinsics[0, 2]) * z / camera_intrinsics[0, 0]
                y = (v - camera_intrinsics[1, 2]) * z / camera_intrinsics[1, 1]
                points.append([x, y, z])

    return np.array(points)
```

### 3. Semantic Segmentation
Pixel-level object classification for scene understanding:

```python
# Capture semantic segmentation
semantic_data = camera.get_semantic_segmentation()

# Define semantic labels
semantic_labels = {
    0: "background",
    1: "robot",
    2: "table",
    3: "chair",
    4: "wall",
    5: "floor"
}
```

### 4. Instance Segmentation
Object instance identification for multi-object tracking:

```python
# Capture instance segmentation
instance_data = camera.get_instance_segmentation()

# Get unique object IDs
unique_instances = np.unique(instance_data)
for instance_id in unique_instances:
    if instance_id != 0:  # Skip background
        mask = instance_data == instance_id
        print(f"Instance {instance_id}: {np.sum(mask)} pixels")
```

## Isaac Sim Synthetic Data Pipeline

### 1. USD Scene Setup for Data Generation
```python
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.sensor import Camera

def setup_data_generation_scene():
    world = World(stage_units_in_meters=1.0)

    # Add ground plane
    world.scene.add_default_ground_plane()

    # Add objects with semantic labels
    cube = create_primitive(
        prim_path="/World/Cube",
        primitive_props={
            "prim_type": "Cube",
            "position": [1.0, 0.0, 0.5],
            "scale": [0.3, 0.3, 0.3]
        }
    )

    # Set semantic label for the cube
    from omni.isaac.core.utils.semantics import add_semantics
    add_semantics(prim_path="/World/Cube", semantic_label="object")

    # Add camera
    camera = Camera(
        prim_path="/World/Camera",
        position=[2.0, 0.0, 1.0],
        frequency=30
    )

    return world, camera
```

### 2. Data Capture Configuration
```python
# Configure camera with multiple sensor types
def configure_camera_for_data_capture(camera):
    # Enable RGB capture
    camera.add_render_product()

    # Enable depth
    camera.add_depth_to_render_product()

    # Enable semantic segmentation
    camera.add_semantic_segmentation_to_render_product()

    # Enable instance segmentation
    camera.add_instance_segmentation_to_render_product()

    # Enable normal maps
    camera.add_normals_to_render_product()
```

### 3. Data Processing Pipeline
```python
import cv2
import numpy as np
import os

class SyntheticDataProcessor:
    def __init__(self, output_dir="synthetic_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories for different data types
        self.rgb_dir = os.path.join(output_dir, "rgb")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.semantic_dir = os.path.join(output_dir, "semantic")

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.semantic_dir, exist_ok=True)

        self.frame_counter = 0

    def save_frame_data(self, camera):
        """Save all sensor data for a single frame"""
        # Capture data
        rgb = camera.get_rgb()
        depth = camera.get_depth()
        semantic = camera.get_semantic_segmentation()

        # Save RGB image
        rgb_filename = os.path.join(self.rgb_dir, f"frame_{self.frame_counter:06d}.png")
        cv2.imwrite(rgb_filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Save depth map
        depth_filename = os.path.join(self.depth_dir, f"frame_{self.frame_counter:06d}.npy")
        np.save(depth_filename, depth)

        # Save semantic segmentation
        semantic_filename = os.path.join(self.semantic_dir, f"frame_{self.frame_counter:06d}.png")
        cv2.imwrite(semantic_filename, semantic.astype(np.uint8))

        # Create metadata
        metadata = {
            "frame_id": self.frame_counter,
            "timestamp": camera.get_current_frame_time(),
            "camera_pose": camera.get_world_pose()
        }

        self.frame_counter += 1

        return metadata
```

## Domain Randomization

Domain randomization is a technique to make synthetic data more robust for real-world transfer by varying environmental parameters:

### 1. Lighting Randomization
```python
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdLux, Gf

def randomize_lighting():
    """Randomize lighting conditions in the scene"""
    # Get the default light (usually from the ground plane)
    light_prim = get_prim_at_path("/World/light")

    if light_prim:
        # Randomize light intensity (100 to 1000)
        intensity = np.random.uniform(100, 1000)
        light_prim.GetAttribute("intensity").Set(intensity)

        # Randomize light color temperature (3000K to 8000K)
        color_temp = np.random.uniform(3000, 8000)
        # Convert to RGB approx
        rgb = color_temperature_to_rgb(color_temp)
        light_prim.GetAttribute("color").Set(Gf.Vec3f(*rgb))

def color_temperature_to_rgb(temp):
    """Convert color temperature to RGB (approximation)"""
    temp = temp / 100
    if temp <= 66:
        red = 255
        green = temp
        green = 99.4708025861 * math.log(green) - 161.1195681661
    else:
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        green = temp - 60
        green = 288.1221695283 * (green ** -0.0755148492)

    blue = temp - 10
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = temp - 10
        blue = 138.5177312231 * math.log(blue) - 305.0447927307

    return [max(0, min(255, x)) / 255.0 for x in [red, green, blue]]
```

### 2. Texture Randomization
```python
def randomize_textures(world):
    """Randomize surface textures in the scene"""
    # Get all objects in the scene
    prims = world.scene.get_viewport_frame().get_prims()

    for prim in prims:
        if prim.GetTypeName() in ["Cube", "Sphere", "Cylinder"]:
            # Randomize material properties
            material = create_random_material()
            set_material_to_prim(prim, material)

def create_random_material():
    """Create a random material with varied properties"""
    import random

    material_properties = {
        "albedo": [
            random.uniform(0.1, 1.0),  # Red
            random.uniform(0.1, 1.0),  # Green
            random.uniform(0.1, 1.0)   # Blue
        ],
        "metallic": random.uniform(0.0, 1.0),
        "roughness": random.uniform(0.0, 1.0),
        "specular": random.uniform(0.0, 1.0)
    }

    return material_properties
```

### 3. Camera Pose Randomization
```python
def randomize_camera_pose(camera, workspace_bounds):
    """Randomize camera position and orientation"""
    import random

    # Random position within workspace bounds
    x = random.uniform(workspace_bounds[0][0], workspace_bounds[0][1])
    y = random.uniform(workspace_bounds[1][0], workspace_bounds[1][1])
    z = random.uniform(workspace_bounds[2][0], workspace_bounds[2][1])

    # Random rotation
    roll = random.uniform(-0.2, 0.2)    # Small roll variation
    pitch = random.uniform(-0.5, 0.5)   # Pitch variation
    yaw = random.uniform(-3.14, 3.14)   # Full yaw rotation

    # Set new pose
    camera.set_world_pose(position=[x, y, z], orientation=[roll, pitch, yaw])
```

## Large-Scale Data Generation Pipeline

### Automated Data Collection Script
```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_primitive
from omni.isaac.sensor import Camera
import numpy as np
import random
import time

class AutomatedDataCollector:
    def __init__(self, num_frames=10000, output_dir="synthetic_dataset"):
        self.num_frames = num_frames
        self.output_dir = output_dir
        self.processor = SyntheticDataProcessor(output_dir)
        self.frame_count = 0

    def setup_scene(self):
        """Setup the scene with random objects"""
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # Add multiple random objects
        for i in range(10):
            # Random object type
            obj_type = random.choice(["Cube", "Sphere", "Cylinder"])
            position = [
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(0.5, 2)
            ]

            create_primitive(
                prim_path=f"/World/Object_{i}",
                primitive_props={
                    "prim_type": obj_type,
                    "position": position,
                    "scale": [random.uniform(0.2, 0.5)] * 3
                }
            )

        # Add camera
        self.camera = Camera(
            prim_path="/World/Camera",
            position=[3, 0, 2],
            frequency=30
        )

        self.world.reset()

    def collect_data(self):
        """Collect synthetic data with domain randomization"""
        for frame_idx in range(self.num_frames):
            # Randomize scene
            if frame_idx % 50 == 0:  # Randomize every 50 frames
                self.randomize_scene()

            # Step simulation
            self.world.step(render=True)

            # Save frame data
            metadata = self.processor.save_frame_data(self.camera)

            # Print progress
            if frame_idx % 100 == 0:
                print(f"Collected {frame_idx}/{self.num_frames} frames")

    def randomize_scene(self):
        """Randomize various scene properties"""
        # Randomize lighting
        randomize_lighting()

        # Randomize object positions slightly
        for i in range(10):
            obj_prim = get_prim_at_path(f"/World/Object_{i}")
            if obj_prim:
                current_pos = obj_prim.GetAttribute("xformOp:translate").Get()
                new_pos = [
                    current_pos[0] + random.uniform(-0.1, 0.1),
                    current_pos[1] + random.uniform(-0.1, 0.1),
                    current_pos[2] + random.uniform(-0.1, 0.1)
                ]
                obj_prim.GetAttribute("xformOp:translate").Set(new_pos)

# Usage example
if __name__ == "__main__":
    collector = AutomatedDataCollector(num_frames=5000, output_dir="my_dataset")
    collector.setup_scene()
    collector.collect_data()
    print("Data collection completed!")
```

## Data Quality Assessment

### Evaluating Synthetic vs Real Data Similarity
```python
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import linalg

def calculate_fid(real_images, synthetic_images):
    """Calculate Fréchet Inception Distance between real and synthetic data"""
    # This is a simplified implementation
    # In practice, you'd use pre-trained Inception features

    # Calculate basic statistics
    real_mean = np.mean(real_images, axis=0)
    real_cov = np.cov(real_images.reshape(len(real_images), -1).T)

    synth_mean = np.mean(synthetic_images, axis=0)
    synth_cov = np.cov(synthetic_images.reshape(len(synthetic_images), -1).T)

    # Calculate FID
    diff = real_mean - synth_mean
    covmean = linalg.sqrtm(real_cov.dot(synth_cov))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(real_cov + synth_cov - 2 * covmean)
    return fid

def assess_synthetic_data_quality(real_samples, synthetic_samples):
    """Assess quality of synthetic data"""
    metrics = {}

    # SSIM comparison
    ssim_scores = []
    for real, synth in zip(real_samples, synthetic_samples):
        score = ssim(real, synth, multichannel=True, data_range=255)
        ssim_scores.append(score)

    metrics['ssim_mean'] = np.mean(ssim_scores)
    metrics['ssim_std'] = np.std(ssim_scores)

    # FID score
    metrics['fid'] = calculate_fid(real_samples, synthetic_samples)

    # Perceptual quality
    metrics['perceptual_score'] = calculate_perceptual_score(real_samples, synthetic_samples)

    return metrics

def calculate_perceptual_score(real_images, synth_images):
    """Calculate perceptual similarity using VGG features"""
    # Implementation would use pre-trained VGG network
    # to extract high-level features and compare them
    pass
```

## Integration with AI Training Pipelines

### Data Loading for PyTorch
```python
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class SyntheticDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load file lists
        self.rgb_files = sorted(os.listdir(os.path.join(data_dir, "rgb")))
        self.semantic_files = sorted(os.path.join(data_dir, "semantic"))

        # Verify same number of files
        assert len(self.rgb_files) == len(self.semantic_files)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = os.path.join(self.data_dir, "rgb", self.rgb_files[idx])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load semantic segmentation
        sem_path = os.path.join(self.data_dir, "semantic", self.semantic_files[idx])
        semantic = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            rgb = self.transform(rgb)
            semantic = self.transform(semantic)

        return {
            'rgb': torch.tensor(rgb, dtype=torch.float32),
            'semantic': torch.tensor(semantic, dtype=torch.long),
            'filename': self.rgb_files[idx]
        }

# Usage example
dataset = SyntheticDataset("synthetic_dataset")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    rgb_batch = batch['rgb']
    semantic_batch = batch['semantic']

    # Train your model
    # outputs = model(rgb_batch)
    # loss = criterion(outputs, semantic_batch)
```

## Performance Optimization

### Batch Processing for Efficiency
```python
def batch_synthetic_data_generation(world, cameras, num_batches=100, batch_size=10):
    """Generate synthetic data in batches for efficiency"""
    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx+1}/{num_batches}")

        for frame_idx in range(batch_size):
            # Randomize scene
            randomize_scene()

            # Step simulation
            world.step(render=True)

            # Capture from all cameras
            for cam_idx, camera in enumerate(cameras):
                save_frame_data(camera, f"batch_{batch_idx}_cam_{cam_idx}_frame_{frame_idx}")

def multi_camera_setup():
    """Setup multiple cameras for comprehensive data collection"""
    cameras = []

    # Front camera
    front_cam = Camera(
        prim_path="/World/FrontCamera",
        position=[0, 0, 1.5],
        frequency=30
    )
    cameras.append(front_cam)

    # Overhead camera
    overhead_cam = Camera(
        prim_path="/World/OverheadCamera",
        position=[0, 0, 3.0],
        frequency=30
    )
    cameras.append(overhead_cam)

    # Side camera
    side_cam = Camera(
        prim_path="/World/SideCamera",
        position=[2.0, 0, 1.5],
        frequency=30
    )
    cameras.append(side_cam)

    return cameras
```

## Best Practices for Synthetic Data

### 1. Quality Control
- Regularly assess synthetic vs real similarity
- Monitor domain randomization effectiveness
- Validate data diversity and coverage

### 2. Annotation Accuracy
- Ensure ground truth data is accurate
- Validate semantic segmentation labels
- Cross-check with real-world annotations when possible

### 3. Computational Efficiency
- Optimize scene complexity for generation speed
- Use appropriate rendering quality settings
- Implement efficient data storage and retrieval

## Summary

Synthetic data generation in Isaac Sim provides:
- High-quality training data for robotics AI
- Domain randomization for robust model training
- Scalable data collection without physical constraints
- Accurate ground truth for evaluation

Proper implementation of synthetic data pipelines accelerates AI development while reducing real-world data collection costs.

## Exercises

1. Create a synthetic data pipeline for object detection
2. Implement domain randomization for lighting and textures
3. Generate a dataset for semantic segmentation
4. Compare synthetic vs real data quality metrics