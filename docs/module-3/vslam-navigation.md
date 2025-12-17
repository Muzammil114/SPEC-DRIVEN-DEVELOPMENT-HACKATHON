---
sidebar_position: 5
---

# Hardware-Accelerated VSLAM and Navigation

## Learning Objectives
- Understand Visual SLAM (VSLAM) concepts and algorithms
- Learn to implement hardware-accelerated VSLAM using Isaac Sim and Isaac ROS
- Configure and optimize VSLAM for different environments
- Integrate VSLAM with navigation systems
- Evaluate VSLAM performance and accuracy

## Introduction to Visual SLAM

Visual Simultaneous Localization and Mapping (VSLAM) is a critical technology for autonomous robots that enables them to build a map of an unknown environment while simultaneously tracking their position within it using visual sensors.

### Key VSLAM Concepts
- **Feature Detection**: Identifying distinctive points in images
- **Feature Matching**: Associating features across frames
- **Pose Estimation**: Determining camera/robot pose
- **Mapping**: Building a representation of the environment
- **Loop Closure**: Recognizing previously visited locations

### VSLAM vs Other SLAM Approaches
- **LIDAR SLAM**: Uses LiDAR sensors, accurate but expensive
- **Visual SLAM**: Uses cameras, cost-effective but sensitive to lighting
- **Visual-Inertial SLAM**: Combines cameras and IMUs for robustness
- **Multi-sensor SLAM**: Integrates multiple sensor types

## VSLAM Algorithms

### 1. Feature-Based Methods
Feature-based VSLAM extracts and tracks distinctive features across frames:

```python
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class FeatureBasedVSLAM:
    def __init__(self):
        # Feature detector (GPU-accelerated)
        self.detector = cv2.cuda.SIFT_create() if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.SIFT_create()

        # Feature matcher
        self.matcher = cv2.BFMatcher()  # Could be FLANN for GPU

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None

        # Pose tracking
        self.current_pose = np.eye(4)
        self.keyframes = []

    def detect_features(self, image):
        """Detect features in the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Use GPU if available
        if isinstance(self.detector, cv2.cuda_SIFT):
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(gray)
            keypoints, descriptors = self.detector.detectAndCompute(gpu_image, None)
            # Download results to CPU
            keypoints = [cv2.KeyPoint(x=k.pt[0], y=k.pt[1], _size=k.size, _angle=k.angle) for k in keypoints]
        else:
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_pose(self, kp1, kp2, matches):
        """Estimate relative pose between frames"""
        if len(matches) < 10:
            return None, False

        # Get matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix,
                                     method=cv2.RANSAC, threshold=1.0)

        if E is None:
            return None, False

        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        return T, True
```

### 2. Direct Methods
Direct methods work with pixel intensities rather than features:

```python
class DirectVSLAM:
    def __init__(self):
        self.reference_frame = None
        self.depth_map = None
        self.camera_pose = np.eye(4)

    def photometric_error(self, img1, img2, depth, T):
        """Calculate photometric error between frames"""
        # Project 3D points from reference frame to current frame
        h, w = img1.shape[:2]
        y, x = np.mgrid[:h, :w]

        # Back-project to 3D
        x3d = (x - self.cx) * depth / self.fx
        y3d = (y - self.cy) * depth / self.fy
        z3d = depth

        # Stack into homogeneous coordinates
        points_3d = np.stack([x3d, y3d, z3d, np.ones_like(z3d)], axis=-1)

        # Transform to current camera frame
        transformed_points = (T @ points_3d.reshape(-1, 4).T).T
        transformed_points = transformed_points.reshape(h, w, 4)

        # Project to current image
        x_proj = (transformed_points[..., 0] * self.fx) / transformed_points[..., 2] + self.cx
        y_proj = (transformed_points[..., 1] * self.fy) / transformed_points[..., 2] + self.cy

        # Sample current image at projected coordinates
        valid_mask = (x_proj >= 0) & (x_proj < w) & (y_proj >= 0) & (y_proj < h) & (transformed_points[..., 2] > 0)

        # Calculate photometric error
        sampled_img = cv2.remap(img2, x_proj.astype(np.float32), y_proj.astype(np.float32),
                               cv2.INTER_LINEAR)
        error = np.abs(img1.astype(np.float32) - sampled_img.astype(np.float32))

        return np.mean(error[valid_mask]), valid_mask
```

## NVIDIA Isaac VSLAM Implementation

### Isaac ROS Visual SLAM
NVIDIA Isaac ROS provides hardware-accelerated VSLAM:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation as R

class IsaacROSVisualSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_vslam_node')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect',
            self.image_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # VSLAM state
        self.camera_matrix = None
        self.has_imu = False
        self.prev_image = None
        self.current_pose = np.eye(4)
        self.frame_count = 0

        # Parameters
        self.declare_parameter('enable_imu_fusion', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def imu_callback(self, msg):
        """Process IMU data for VSLAM"""
        if self.get_parameter('enable_imu_fusion').value:
            self.has_imu = True
            # Process IMU data for visual-inertial fusion
            self.process_imu_data(msg)

    def image_callback(self, msg):
        """Process incoming camera images for VSLAM"""
        if self.camera_matrix is None:
            return

        # Convert ROS image to OpenCV
        from cv_bridge import CvBridge
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # Process VSLAM
        if self.prev_image is not None:
            # Estimate motion between frames
            motion = self.estimate_motion(self.prev_image, cv_image)

            if motion is not None:
                # Update pose
                self.current_pose = self.current_pose @ motion

                # Publish results
                self.publish_results(msg.header, self.current_pose)

        self.prev_image = cv_image
        self.frame_count += 1

    def estimate_motion(self, prev_img, curr_img):
        """Estimate motion between two frames using Isaac ROS methods"""
        # This would use Isaac ROS optimized functions
        # In practice, this calls hardware-accelerated kernels

        # Feature detection (GPU accelerated)
        keypoints_prev, descriptors_prev = self.detect_features(prev_img)
        keypoints_curr, descriptors_curr = self.detect_features(curr_img)

        # Feature matching
        matches = self.match_features(descriptors_prev, descriptors_curr)

        # Pose estimation
        if len(matches) >= 10:
            pose_change, success = self.estimate_pose(
                keypoints_prev, keypoints_curr, matches
            )

            if success:
                return pose_change

        return None

    def detect_features(self, image):
        """GPU-accelerated feature detection"""
        # In Isaac ROS, this uses CUDA-optimized feature detectors
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Isaac ROS optimized feature detector
        # This is pseudo-code - actual Isaac ROS API would be used
        keypoints, descriptors = self.isaac_ros_feature_detector.detect_and_compute(gray)

        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """GPU-accelerated feature matching"""
        # Use Isaac ROS optimized matcher
        matches = self.isaac_ros_feature_matcher.match(desc1, desc2)
        return matches

    def publish_results(self, header, pose):
        """Publish VSLAM results"""
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = self.get_parameter('odom_frame').value
        odom_msg.child_frame_id = self.get_parameter('base_frame').value

        # Set pose
        odom_msg.pose.pose.position.x = pose[0, 3]
        odom_msg.pose.pose.position.y = pose[1, 3]
        odom_msg.pose.pose.position.z = pose[2, 3]

        # Convert rotation matrix to quaternion
        r = R.from_matrix(pose[:3, :3])
        quat = r.as_quat()
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Publish TF
        t = TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = self.get_parameter('odom_frame').value
        t.child_frame_id = self.get_parameter('base_frame').value

        t.transform.translation.x = pose[0, 3]
        t.transform.translation.y = pose[1, 3]
        t.transform.translation.z = pose[2, 3]

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def process_imu_data(self, imu_msg):
        """Process IMU data for visual-inertial fusion"""
        # This would integrate IMU data with visual data
        # in Isaac ROS VSLAM implementation
        pass

def main(args=None):
    rclpy.init(args=args)
    vslam_node = IsaacROSVisualSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Sim VSLAM Integration

### Simulating VSLAM in Isaac Sim
```python
# Isaac Sim VSLAM simulation example
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np

class IsaacSimVSLAMSimulation:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)

        # Add ground plane and objects
        self.world.scene.add_default_ground_plane()

        # Add robot with camera
        self.setup_robot_with_camera()

        # Initialize VSLAM simulation
        self.gt_poses = []  # Ground truth poses
        self.vslam_poses = []  # VSLAM estimated poses
        self.frame_count = 0

    def setup_robot_with_camera(self):
        """Setup robot and camera for VSLAM simulation"""
        # Add a simple robot (could be a more complex model)
        add_reference_to_stage(
            usd_path="/Isaac/Robots/Carter/carter_vision.usd",
            prim_path="/World/Robot"
        )

        # Add camera to robot
        self.camera = Camera(
            prim_path="/World/Robot/base_link/camera",
            position=[0.3, 0.0, 0.2],
            frequency=30
        )

        # Configure camera for VSLAM
        self.camera.add_render_product()
        self.camera.add_depth_to_render_product()

    def simulate_vslam_trajectory(self, trajectory_points):
        """Simulate VSLAM along a trajectory"""
        robot = self.world.scene.get_object("Robot")

        for i, (pos, quat) in enumerate(trajectory_points):
            # Set robot pose
            robot.set_world_pose(position=pos, orientation=quat)

            # Step simulation
            self.world.step(render=True)

            # Get ground truth pose
            gt_pose = robot.get_world_pose()
            self.gt_poses.append(gt_pose)

            # Simulate VSLAM processing (add noise to simulate errors)
            vslam_pose = self.simulate_vslam_estimation(gt_pose)
            self.vslam_poses.append(vslam_pose)

            # Save frame data for VSLAM processing
            if i % 5 == 0:  # Process every 5th frame
                rgb_data = self.camera.get_rgb()
                depth_data = self.camera.get_depth()

                # Process with VSLAM algorithm
                self.process_vslam_frame(rgb_data, depth_data, vslam_pose)

            self.frame_count += 1

    def simulate_vslam_estimation(self, gt_pose):
        """Simulate VSLAM pose estimation with realistic errors"""
        pos, quat = gt_pose

        # Add realistic VSLAM errors
        pos_noise = np.random.normal(0, 0.02, 3)  # 2cm position noise
        quat_noise = np.random.normal(0, 0.01, 4)  # Small orientation noise

        estimated_pos = pos + pos_noise
        estimated_quat = self.add_quaternion_noise(quat, quat_noise)

        return (estimated_pos, estimated_quat)

    def add_quaternion_noise(self, q, noise):
        """Add noise to quaternion while maintaining unit length"""
        q_noise = q + noise * 0.1  # Scale noise appropriately
        return q_noise / np.linalg.norm(q_noise)

    def process_vslam_frame(self, rgb_data, depth_data, current_pose):
        """Process a frame through VSLAM pipeline"""
        # This would call actual VSLAM processing
        # In Isaac Sim, this could use synthetic data generation
        # to create realistic VSLAM inputs

        # For simulation purposes, we track the estimated vs ground truth
        print(f"Processed frame {self.frame_count}, GT: {current_pose[0]}, Estimated: {current_pose[0]}")

    def evaluate_vslam_performance(self):
        """Evaluate VSLAM performance metrics"""
        if len(self.gt_poses) == 0 or len(self.vslam_poses) == 0:
            return {}

        gt_positions = np.array([pose[0] for pose in self.gt_poses])
        est_positions = np.array([pose[0] for pose in self.vslam_poses])

        # Calculate position errors
        position_errors = np.linalg.norm(gt_positions - est_positions, axis=1)

        metrics = {
            'rmse_position': np.sqrt(np.mean(position_errors**2)),
            'mean_position_error': np.mean(position_errors),
            'max_position_error': np.max(position_errors),
            'total_frames': len(position_errors)
        }

        return metrics
```

## Performance Optimization for VSLAM

### Multi-Resolution Processing
```python
class MultiResolutionVSLAM:
    def __init__(self):
        self.levels = 3  # Number of image pyramid levels
        self.feature_detectors = {}

        # Initialize feature detectors for each level
        for level in range(self.levels):
            # Use different detector parameters for each level
            detector_params = self.get_detector_params_for_level(level)
            self.feature_detectors[level] = cv2.SIFT_create(**detector_params)

    def build_image_pyramid(self, image):
        """Build Gaussian pyramid for multi-resolution processing"""
        pyramid = [image]
        for i in range(1, self.levels):
            h, w = pyramid[-1].shape[:2]
            # Downsample image
            downsampled = cv2.pyrDown(pyramid[-1])
            pyramid.append(downsampled)
        return pyramid

    def process_multiresolution_features(self, pyramid):
        """Process features at multiple resolutions"""
        all_keypoints = []
        all_descriptors = []

        for level, img in enumerate(pyramid):
            # Detect features at this level
            kp, desc = self.feature_detectors[level].detectAndCompute(img, None)

            # Scale keypoints back to original resolution
            scaled_kp = []
            scale_factor = 2 ** level
            for pt in kp:
                new_pt = cv2.KeyPoint(
                    x=pt.pt[0] * scale_factor,
                    y=pt.pt[1] * scale_factor,
                    _size=pt.size * scale_factor,
                    _angle=pt.angle
                )
                scaled_kp.append(new_pt)

            all_keypoints.extend(scaled_kp)
            if desc is not None:
                if all_descriptors == []:
                    all_descriptors = desc
                else:
                    all_descriptors = np.vstack([all_descriptors, desc])

        return all_keypoints, all_descriptors
```

### GPU-Accelerated Feature Matching
```python
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

class GPUFeatureMatcher:
    def __init__(self):
        # CUDA kernel for feature matching
        self.matching_kernel = self.compile_matching_kernel()

    def compile_matching_kernel(self):
        """Compile CUDA kernel for feature matching"""
        kernel_code = """
        __global__ void match_features(float* desc1, float* desc2, int* matches,
                                     int num_desc1, int num_desc2, int desc_dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < num_desc1) {
                float min_dist = 1e10;
                int best_match = -1;

                for (int j = 0; j < num_desc2; j++) {
                    float dist = 0;
                    for (int k = 0; k < desc_dim; k++) {
                        float diff = desc1[idx * desc_dim + k] - desc2[j * desc_dim + k];
                        dist += diff * diff;
                    }

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_match = j;
                    }
                }

                matches[idx] = best_match;
            }
        }
        """

        mod = SourceModule(kernel_code)
        return mod.get_function("match_features")

    def match_features_gpu(self, desc1, desc2):
        """Perform feature matching on GPU"""
        if desc1 is None or desc2 is None:
            return np.array([])

        num_desc1, desc_dim = desc1.shape
        num_desc2 = desc2.shape[0]

        # Allocate GPU memory
        desc1_gpu = cuda.mem_alloc(desc1.nbytes)
        desc2_gpu = cuda.mem_alloc(desc2.nbytes)
        matches_gpu = cuda.mem_alloc(num_desc1 * 4)  # int array

        # Copy data to GPU
        cuda.memcpy_htod(desc1_gpu, desc1.astype(np.float32))
        cuda.memcpy_htod(desc2_gpu, desc2.astype(np.float32))

        # Configure kernel launch parameters
        block_size = 256
        grid_size = (num_desc1 + block_size - 1) // block_size

        # Launch kernel
        self.matching_kernel(
            desc1_gpu, desc2_gpu, matches_gpu,
            np.int32(num_desc1), np.int32(num_desc2), np.int32(desc_dim),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

        # Copy results back
        matches = np.empty(num_desc1, dtype=np.int32)
        cuda.memcpy_dtoh(matches, matches_gpu)

        # Cleanup
        desc1_gpu.free()
        desc2_gpu.free()
        matches_gpu.free()

        return matches
```

## Integration with Navigation Systems

### VSLAM + Nav2 Integration
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from tf2_ros import Buffer, TransformListener
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class VSLAMNavigationIntegrator(Node):
    def __init__(self):
        super().__init__('vslam_navigation_integrator')

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers for navigation
        self.odom_pub = self.create_publisher(Odometry, '/vslam_odom', 10)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timer for TF publishing
        self.timer = self.create_timer(0.05, self.publish_vslam_tf)  # 20 Hz

        # VSLAM to odom transform
        self.vslam_to_odom_transform = np.eye(4)

        # Parameters
        self.declare_parameter('vslam_frame', 'vslam_frame')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')

    def publish_vslam_tf(self):
        """Publish TF from VSLAM frame to robot base"""
        try:
            # Get transform from VSLAM frame to base link
            vslam_to_base = self.tf_buffer.lookup_transform(
                self.get_parameter('vslam_frame').value,
                self.get_parameter('base_frame').value,
                rclpy.time.Time()
            )

            # Calculate VSLAM to odom transform (for Nav2 integration)
            odom_to_vslam = self.calculate_odom_to_vslam(vslam_to_base)

            # Publish the transform
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.get_parameter('odom_frame').value
            t.child_frame_id = self.get_parameter('vslam_frame').value

            t.transform.translation.x = odom_to_vslam[0, 3]
            t.transform.translation.y = odom_to_vslam[1, 3]
            t.transform.translation.z = odom_to_vslam[2, 3]

            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(odom_to_vslam[:3, :3])
            quat = r.as_quat()
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            # Publish transform
            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().warn(f"Could not lookup transform: {e}")

    def calculate_odom_to_vslam(self, vslam_to_base):
        """Calculate odom to VSLAM transform"""
        # This would maintain the relationship between VSLAM and odometry frames
        # In practice, this might use a kalman filter to fuse VSLAM and wheel odometry
        pass

    def send_navigation_goal(self, x, y, theta):
        """Send navigation goal using VSLAM localization"""
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Navigation action server not available")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.get_parameter('map_frame').value
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        from scipy.spatial.transform import Rotation as R
        quat = R.from_euler('z', theta).as_quat()
        goal_msg.pose.pose.orientation.x = quat[0]
        goal_msg.pose.pose.orientation.y = quat[1]
        goal_msg.pose.pose.orientation.z = quat[2]
        goal_msg.pose.pose.orientation.w = quat[3]

        self.nav_client.send_goal_async(goal_msg)
        return True
```

## Performance Evaluation and Tuning

### VSLAM Accuracy Metrics
```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class VSLAMMetrics:
    @staticmethod
    def calculate_ate(gt_poses, est_poses):
        """Calculate Absolute Trajectory Error"""
        if len(gt_poses) != len(est_poses):
            raise ValueError("Ground truth and estimated poses must have same length")

        errors = []
        for gt, est in zip(gt_poses, est_poses):
            gt_pos, gt_rot = gt
            est_pos, est_rot = est

            # Position error
            pos_error = np.linalg.norm(gt_pos - est_pos)
            errors.append(pos_error)

        return {
            'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors),
            'max': np.max(errors)
        }

    @staticmethod
    def calculate_rpe(gt_poses, est_poses, delta_frames=1):
        """Calculate Relative Pose Error"""
        errors = []
        for i in range(len(gt_poses) - delta_frames):
            # Ground truth relative transform
            gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta_frames]
            # Estimated relative transform
            est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i + delta_frames]

            # Calculate error
            error_transform = np.linalg.inv(gt_rel) @ est_rel
            pos_error = np.linalg.norm(error_transform[:3, 3])
            errors.append(pos_error)

        return {
            'rmse': np.sqrt(np.mean(np.array(errors) ** 2)),
            'mean': np.mean(errors),
            'std': np.std(errors)
        }

    @staticmethod
    def calculate_orientation_error(gt_poses, est_poses):
        """Calculate orientation error"""
        errors = []
        for gt, est in zip(gt_poses, est_poses):
            gt_pos, gt_quat = gt
            est_pos, est_quat = est

            # Convert to rotation matrices
            gt_rot = R.from_quat(gt_quat).as_matrix()
            est_rot = R.from_quat(est_quat).as_matrix()

            # Calculate rotation error
            rel_rot = np.linalg.inv(gt_rot) @ est_rot
            angle_error = np.arccos(np.clip((np.trace(rel_rot) - 1) / 2, -1, 1))
            errors.append(angle_error)

        return {
            'rmse_deg': np.degrees(np.sqrt(np.mean(np.array(errors) ** 2))),
            'mean_deg': np.degrees(np.mean(errors))
        }
```

## Troubleshooting VSLAM Issues

### Common Problems and Solutions

#### 1. Drift Issues
- **Symptoms**: Accumulated position errors over time
- **Solutions**:
  - Implement loop closure detection
  - Use visual-inertial fusion
  - Add pose graph optimization

#### 2. Tracking Failure
- **Symptoms**: Lost tracking in textureless areas
- **Solutions**:
  - Use direct methods in addition to feature-based
  - Implement relocalization
  - Use multiple camera viewpoints

#### 3. Performance Issues
- **Symptoms**: Low frame rates, high latency
- **Solutions**:
  - Optimize feature detection parameters
  - Use multi-resolution processing
  - Leverage GPU acceleration

## Best Practices

### 1. Algorithm Selection
- Use visual-inertial SLAM for robustness
- Consider environment characteristics (lighting, texture)
- Balance accuracy vs. computational requirements

### 2. Hardware Optimization
- Leverage GPU acceleration when available
- Optimize memory usage for real-time performance
- Use appropriate image resolutions

### 3. Validation
- Test on diverse environments
- Compare against ground truth when available
- Monitor performance metrics continuously

## Summary

Hardware-accelerated VSLAM provides:
- Real-time localization and mapping using visual sensors
- GPU acceleration for improved performance
- Integration with navigation systems
- Isaac Sim for realistic testing and validation

Successful VSLAM implementation requires careful consideration of algorithm choice, hardware optimization, and system integration.

## Exercises

1. Implement a basic VSLAM system using feature detection and matching
2. Integrate VSLAM with Nav2 for autonomous navigation
3. Evaluate VSLAM performance in Isaac Sim environments
4. Optimize VSLAM for deployment on NVIDIA Jetson hardware