---
sidebar_position: 8
---

# Sim-to-Real Transfer Techniques

## Learning Objectives
- Understand the challenges of transferring policies from simulation to reality
- Learn domain randomization and system identification techniques
- Implement sim-to-real transfer methodologies
- Validate and fine-tune policies for real-world deployment
- Address latency, sensor noise, and actuator differences

## Introduction to Sim-to-Real Transfer

Sim-to-real transfer is the process of taking policies, controllers, or algorithms developed in simulation and successfully deploying them on physical robots. This approach offers significant advantages:

- **Safety**: Test dangerous scenarios in simulation
- **Cost**: Reduce hardware wear and testing costs
- **Speed**: Accelerate development and testing
- **Scalability**: Train on multiple parallel simulation instances

However, sim-to-real transfer faces the "reality gap" - the difference between simulated and real environments that can cause policies to fail when deployed on physical robots.

## The Reality Gap Problem

### Sources of the Reality Gap

#### 1. Modeling Inaccuracies
- **Dynamics**: Inaccurate mass, inertia, friction parameters
- **Actuators**: Idealized motor models vs. real-world limitations
- **Sensors**: Perfect simulation vs. noisy real sensors
- **Environment**: Simplified physics vs. complex real interactions

#### 2. Environmental Differences
- **Lighting**: Controlled simulation vs. variable real lighting
- **Textures**: Perfect textures vs. real-world surface variations
- **Obstacles**: Idealized shapes vs. complex real objects
- **Physics**: Simplified models vs. real-world physics

#### 3. Latency and Timing
- **Control frequency**: Simulation vs. real-time constraints
- **Communication delays**: Network vs. direct hardware communication
- **Processing time**: Different computational capabilities

## Domain Randomization

Domain randomization is a technique that randomizes simulation parameters to make policies robust to variations:

### Basic Domain Randomization Implementation
```python
import numpy as np
import torch

class DomainRandomizer:
    def __init__(self):
        # Define randomization ranges
        self.randomization_ranges = {
            'robot_mass': [0.8, 1.2],  # Multiplier
            'friction_coeff': [0.5, 2.0],
            'actuator_gain': [0.9, 1.1],
            'sensor_noise_std': [0.0, 0.05],
            'latency_range': [0.0, 0.02],  # seconds
            'lighting_intensity': [0.5, 1.5],
            'camera_noise': [0.0, 0.1]
        }

        # Current randomization values
        self.current_params = {}

    def randomize_environment(self, env_id=None):
        """Randomize environment parameters"""
        for param_name, (min_val, max_val) in self.randomization_ranges.items():
            if 'robot' in param_name:
                # Robot-specific randomization
                random_val = np.random.uniform(min_val, max_val)
                self.current_params[param_name] = random_val
                self.apply_robot_randomization(param_name, random_val)
            elif 'sensor' in param_name or 'camera' in param_name:
                # Sensor-specific randomization
                random_val = np.random.uniform(min_val, max_val)
                self.current_params[param_name] = random_val
            elif 'lighting' in param_name:
                # Visual randomization
                random_val = np.random.uniform(min_val, max_val)
                self.current_params[param_name] = random_val
                self.apply_visual_randomization(random_val)

    def apply_robot_randomization(self, param_name, value):
        """Apply robot parameter randomization"""
        if param_name == 'robot_mass':
            # Apply mass multiplier to all robot links
            pass
        elif param_name == 'friction_coeff':
            # Apply friction randomization
            pass
        elif param_name == 'actuator_gain':
            # Apply actuator gain randomization
            pass

    def apply_visual_randomization(self, intensity_multiplier):
        """Apply visual appearance randomization"""
        # Randomize lighting, textures, colors, etc.
        pass

    def get_randomized_observation(self, base_observation):
        """Add randomization effects to observations"""
        obs = base_observation.copy()

        # Add sensor noise
        noise_std = self.current_params.get('sensor_noise_std', 0.0)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, obs.shape)
            obs += noise

        # Apply other randomization effects
        return obs

    def get_randomized_action(self, base_action):
        """Apply randomization to actions (e.g., actuator delays)"""
        # Simulate actuator delays and limitations
        action = base_action.copy()

        # Add latency simulation
        latency = self.current_params.get('latency_range', [0.0, 0.02])
        actual_latency = np.random.uniform(latency[0], latency[1])

        # Apply actuator limitations
        gain = self.current_params.get('actuator_gain', 1.0)
        action = action * gain

        return action
```

### Advanced Domain Randomization with Isaac Sim
```python
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import carb
import numpy as np

class IsaacSimDomainRandomizer:
    def __init__(self, world):
        self.world = world
        self.randomization_params = {
            'mass_range': [0.8, 1.2],
            'friction_range': [0.5, 1.5],
            'restitution_range': [0.0, 0.2],
            'damping_range': [0.0, 0.1],
            'stiffness_range': [0.0, 10.0]
        }

    def randomize_robot_dynamics(self, robot_prim_path, env_ids):
        """Randomize robot dynamics parameters"""
        for env_id in env_ids:
            # Get robot prim
            robot_prim = get_prim_at_path(f"{robot_prim_path}_{env_id}")

            if robot_prim:
                # Randomize mass for each link
                self.randomize_mass(robot_prim)

                # Randomize joint properties
                self.randomize_joint_properties(robot_prim)

    def randomize_mass(self, robot_prim):
        """Randomize mass properties of robot links"""
        # Get all children prims (links)
        for child in robot_prim.GetChildren():
            if child.GetTypeName() == "Xform":
                link_prim = child
                # Get current mass
                mass_api = link_prim.GetAttribute("physics:mass")
                if mass_api:
                    current_mass = mass_api.Get()
                    # Apply randomization
                    mass_multiplier = np.random.uniform(
                        self.randomization_params['mass_range'][0],
                        self.randomization_params['mass_range'][1]
                    )
                    new_mass = current_mass * mass_multiplier
                    mass_api.Set(new_mass)

    def randomize_joint_properties(self, robot_prim):
        """Randomize joint friction and damping"""
        for child in robot_prim.GetChildren():
            if "joint" in child.GetTypeName().lower():
                joint_prim = child
                # Randomize joint friction
                friction_attr = joint_prim.GetAttribute("physics:jointFriction")
                if friction_attr:
                    base_friction = friction_attr.Get() or 0.0
                    friction_multiplier = np.random.uniform(
                        self.randomization_params['friction_range'][0],
                        self.randomization_params['friction_range'][1]
                    )
                    new_friction = base_friction * friction_multiplier
                    friction_attr.Set(new_friction)

    def randomize_visual_appearance(self, env_ids):
        """Randomize visual appearance for domain randomization"""
        for env_id in env_ids:
            # Randomize lighting
            light_prims = [p for p in self.world.scene.stage.GetPrims()
                          if "light" in p.GetTypeName().lower()]

            for light_prim in light_prims:
                # Randomize light intensity
                intensity_attr = light_prim.GetAttribute("intensity")
                if intensity_attr:
                    base_intensity = intensity_attr.Get()
                    intensity_multiplier = np.random.uniform(0.5, 1.5)
                    new_intensity = base_intensity * intensity_multiplier
                    intensity_attr.Set(new_intensity)

    def randomize_material_properties(self, env_ids):
        """Randomize material properties for textures"""
        # This would randomize surface properties like roughness, metallic, etc.
        pass
```

## System Identification

System identification involves modeling the real robot's dynamics to bridge the simulation-reality gap:

### Dynamics Model Identification
```python
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn

class SystemIdentifier:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.identified_params = {}
        self.dynamics_model = None

    def collect_excitation_data(self, robot, trajectory_generator):
        """Collect data for system identification"""
        data_points = []

        # Execute various trajectories to excite different dynamics
        for trajectory in trajectory_generator.generate_excitation_trajectories():
            # Execute trajectory on robot
            positions, velocities, accelerations, torques = self.execute_trajectory(
                robot, trajectory
            )

            # Store data point
            for i in range(len(positions)):
                data_points.append({
                    'q': positions[i],
                    'q_dot': velocities[i],
                    'q_ddot': accelerations[i],
                    'tau': torques[i]
                })

        return data_points

    def identify_rigid_body_parameters(self, data_points):
        """Identify rigid body dynamics parameters"""
        # Robot dynamics: M(q)q_ddot + C(q, q_dot)q_dot + g(q) + F(q_dot) = τ
        # We can rewrite this as: Y(θ) * φ = τ
        # Where Y is the regressor matrix and φ are the parameters to identify

        Y_matrix = []
        tau_vector = []

        for data_point in data_points:
            q = data_point['q']
            q_dot = data_point['q_dot']
            q_ddot = data_point['q_ddot']
            tau = data_point['tau']

            # Construct regressor matrix Y
            Y_row = self.construct_regressor_matrix(q, q_dot, q_ddot)
            Y_matrix.append(Y_row)
            tau_vector.append(tau)

        Y_matrix = np.array(Y_matrix)
        tau_vector = np.array(tau_vector)

        # Solve for parameters using least squares
        # Each joint has its own parameters
        n_joints = len(tau_vector[0])
        identified_params = {}

        for joint_idx in range(n_joints):
            tau_j = tau_vector[:, joint_idx]

            # Solve: Y * φ = τ for φ
            params, residuals, rank, s = np.linalg.lstsq(
                Y_matrix, tau_j, rcond=None
            )

            identified_params[f'joint_{joint_idx}'] = params

        self.identified_params = identified_params
        return identified_params

    def construct_regressor_matrix(self, q, q_dot, q_ddot):
        """Construct the regressor matrix for rigid body dynamics"""
        # This is a simplified example - real implementation would be more complex
        # The regressor matrix contains terms that are linear in the dynamic parameters

        n_dof = len(q)
        Y = np.zeros(n_dof * 10)  # Placeholder size

        # Example terms (in practice, these would be computed based on robot structure)
        for i in range(n_dof):
            # Mass matrix terms
            Y[i] = q_ddot[i]

            # Coriolis and centrifugal terms
            Y[n_dof + i] = q_dot[i] ** 2

            # Gravity terms
            Y[2 * n_dof + i] = np.sin(q[i])

            # Friction terms
            Y[3 * n_dof + i] = np.sign(q_dot[i])
            Y[4 * n_dof + i] = q_dot[i]

        return Y

    def build_dynamics_model(self):
        """Build a learned dynamics model"""
        # Create a neural network model to learn the dynamics
        class DynamicsModel(nn.Module):
            def __init__(self, n_dof):
                super().__init__()
                self.n_dof = n_dof
                self.network = nn.Sequential(
                    nn.Linear(3 * n_dof, 256),  # q, q_dot, q_ddot as input
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, n_dof)  # output: torques
                )

            def forward(self, q, q_dot, q_ddot):
                input_tensor = torch.cat([q, q_dot, q_ddot], dim=-1)
                return self.network(input_tensor)

        self.dynamics_model = DynamicsModel(len(self.identified_params))
        return self.dynamics_model

    def validate_identified_model(self, test_data):
        """Validate the identified model against test data"""
        errors = []

        for data_point in test_data:
            q = torch.FloatTensor(data_point['q']).unsqueeze(0)
            q_dot = torch.FloatTensor(data_point['q_dot']).unsqueeze(0)
            q_ddot = torch.FloatTensor(data_point['q_ddot']).unsqueeze(0)

            # Predict torques using identified model
            predicted_tau = self.dynamics_model(q, q_dot, q_ddot)
            actual_tau = torch.FloatTensor(data_point['tau']).unsqueeze(0)

            # Calculate error
            error = torch.mean((predicted_tau - actual_tau) ** 2)
            errors.append(error.item())

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors)
        }
```

### Neural Network Dynamics Model
```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralDynamicsModel(nn.Module):
    def __init__(self, n_dof, hidden_size=256):
        super().__init__()
        self.n_dof = n_dof

        # Forward dynamics: given q, q_dot, τ -> q_ddot
        self.forward_net = nn.Sequential(
            nn.Linear(2 * n_dof + n_dof, hidden_size),  # q, q_dot, τ
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_dof)  # q_ddot
        )

        # Inverse dynamics: given q, q_dot, q_ddot -> τ
        self.inverse_net = nn.Sequential(
            nn.Linear(3 * n_dof, hidden_size),  # q, q_dot, q_ddot
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_dof)  # τ
        )

    def forward_dynamics(self, q, q_dot, tau):
        """Predict acceleration given position, velocity, and torque"""
        input_tensor = torch.cat([q, q_dot, tau], dim=-1)
        return self.forward_net(input_tensor)

    def inverse_dynamics(self, q, q_dot, q_ddot):
        """Predict required torque given desired acceleration"""
        input_tensor = torch.cat([q, q_dot, q_ddot], dim=-1)
        return self.inverse_net(input_tensor)

class NeuralSystemID:
    def __init__(self, n_dof):
        self.model = NeuralDynamicsModel(n_dof)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train_model(self, dataset, epochs=1000):
        """Train the neural dynamics model"""
        for epoch in range(epochs):
            total_loss = 0

            for batch in dataset:
                q = batch['q']
                q_dot = batch['q_dot']
                q_ddot = batch['q_ddot']
                tau = batch['tau']

                # Forward pass - inverse dynamics
                predicted_tau = self.model.inverse_dynamics(q, q_dot, q_ddot)
                loss_inv = self.criterion(predicted_tau, tau)

                # Forward pass - forward dynamics
                predicted_q_ddot = self.model.forward_dynamics(q, q_dot, tau)
                loss_fwd = self.criterion(predicted_q_ddot, q_ddot)

                # Combined loss
                loss = loss_inv + loss_fwd

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataset):.6f}")
```

## Latency and Timing Considerations

### Latency Modeling and Compensation
```python
import numpy as np
from collections import deque
import time

class LatencyCompensator:
    def __init__(self, max_latency=0.1, control_freq=50):
        self.max_latency = max_latency
        self.control_period = 1.0 / control_freq

        # History buffers for prediction
        self.state_history = deque(maxlen=int(max_latency * control_freq * 2))
        self.action_history = deque(maxlen=int(max_latency * control_freq * 2))

        # Measured latencies
        self.latency_buffer = deque(maxlen=100)

    def measure_latency(self):
        """Measure system latency"""
        start_time = time.time()
        # Send test signal
        test_start = time.time()
        # Simulate round-trip
        time.sleep(0.01)  # Simulated processing
        test_end = time.time()
        measured_latency = test_end - test_start

        self.latency_buffer.append(measured_latency)
        return measured_latency

    def predict_future_state(self, current_state, time_ahead):
        """Predict robot state time_ahead seconds into the future"""
        if len(self.state_history) < 2:
            return current_state

        # Simple linear prediction based on recent history
        recent_states = list(self.state_history)
        if len(recent_states) >= 2:
            # Estimate velocity
            dt = self.control_period
            velocity = (recent_states[-1] - recent_states[-2]) / dt

            # Predict future state
            predicted_state = current_state + velocity * time_ahead
            return predicted_state

        return current_state

    def compensate_for_latency(self, desired_action, current_state):
        """Compensate action for expected latency"""
        # Estimate current latency
        if self.latency_buffer:
            avg_latency = np.mean(self.latency_buffer)
        else:
            avg_latency = self.max_latency / 2

        # Predict state at action application time
        predicted_state = self.predict_future_state(current_state, avg_latency)

        # Compute action based on predicted state
        compensated_action = self.compute_action(predicted_state, desired_action)

        return compensated_action

    def compute_action(self, predicted_state, desired_action):
        """Compute action considering predicted state"""
        # This would implement your specific control logic
        # taking into account the predicted future state
        return desired_action
```

## Sensor Noise and Uncertainty Modeling

### Sensor Noise Modeling
```python
import numpy as np
from scipy import ndimage
import cv2

class SensorNoiseModel:
    def __init__(self):
        self.noise_params = {
            'camera': {
                'gaussian_noise_std': 0.01,
                'salt_pepper_prob': 0.001,
                'blur_kernel_size': 1
            },
            'lidar': {
                'range_noise_std': 0.02,  # meters
                'angular_noise_std': 0.001,  # radians
                'dropout_rate': 0.01
            },
            'imu': {
                'accel_noise_std': 0.01,
                'gyro_noise_std': 0.001,
                'bias_walk_std': 0.0001
            }
        }

    def add_camera_noise(self, image, sensor_type='camera'):
        """Add realistic noise to camera images"""
        params = self.noise_params[sensor_type]

        # Gaussian noise
        gaussian_noise = np.random.normal(0, params['gaussian_noise_std'], image.shape)
        noisy_image = image + gaussian_noise

        # Salt and pepper noise
        if np.random.random() < params['salt_pepper_prob']:
            # Randomly set some pixels to min/max
            salt_pepper_mask = np.random.random(image.shape[:2]) < params['salt_pepper_prob']/2
            noisy_image[salt_pepper_mask] = 0  # Pepper
            salt_mask = np.random.random(image.shape[:2]) < params['salt_pepper_prob']/2
            noisy_image[salt_mask] = 255  # Salt

        # Blur (simulating motion blur or focus issues)
        if params['blur_kernel_size'] > 0:
            kernel = np.ones((params['blur_kernel_size'], params['blur_kernel_size'])) / (params['blur_kernel_size'] ** 2)
            noisy_image = cv2.filter2D(noisy_image, -1, kernel)

        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def add_lidar_noise(self, ranges, angles, sensor_type='lidar'):
        """Add realistic noise to LiDAR data"""
        params = self.noise_params[sensor_type]

        # Range noise
        range_noise = np.random.normal(0, params['range_noise_std'], ranges.shape)
        noisy_ranges = ranges + range_noise

        # Angular noise (small errors in angle measurements)
        angular_noise = np.random.normal(0, params['angular_noise_std'], angles.shape)
        noisy_angles = angles + angular_noise

        # Dropout (missing readings)
        dropout_mask = np.random.random(ranges.shape) < params['dropout_rate']
        noisy_ranges[dropout_mask] = 0  # Invalid readings

        return noisy_ranges, noisy_angles

    def add_imu_noise(self, accel, gyro, dt, sensor_type='imu'):
        """Add realistic noise to IMU data"""
        params = self.noise_params[sensor_type]

        # Add noise to accelerometer
        accel_noise = np.random.normal(0, params['accel_noise_std'], accel.shape)
        noisy_accel = accel + accel_noise

        # Add noise to gyroscope
        gyro_noise = np.random.normal(0, params['gyro_noise_std'], gyro.shape)
        noisy_gyro = gyro + gyro_noise

        # Simulate bias walk over time
        if not hasattr(self, 'accel_bias'):
            self.accel_bias = np.zeros(3)
            self.gyro_bias = np.zeros(3)

        # Random walk bias
        accel_bias_drift = np.random.normal(0, params['bias_walk_std'] * dt, 3)
        gyro_bias_drift = np.random.normal(0, params['bias_walk_std'] * dt, 3)

        self.accel_bias += accel_bias_drift
        self.gyro_bias += gyro_bias_drift

        noisy_accel += self.accel_bias
        noisy_gyro += self.gyro_bias

        return noisy_accel, noisy_gyro
```

## Actuator Modeling and Compensation

### Actuator Dynamics Modeling
```python
import numpy as np
from scipy import signal

class ActuatorModel:
    def __init__(self, motor_constants):
        """
        motor_constants: dict with keys like:
        - 'torque_constant': Nm/A
        - 'resistance': Ohms
        - 'inductance': Henries
        - 'gear_ratio':
        - 'max_current': Amps
        - 'friction_coeff': Nm/(rad/s)
        """
        self.constants = motor_constants
        self.current_state = 0.0  # Current position
        self.current_velocity = 0.0  # Current velocity

        # Create first-order approximation of motor dynamics
        # Transfer function: G(s) = K / (tau*s + 1)
        self.gain = self.estimate_gain()
        self.time_constant = self.estimate_time_constant()

    def estimate_gain(self):
        """Estimate steady-state gain of actuator"""
        # Simplified estimation
        return self.constants.get('torque_constant', 1.0) / self.constants.get('friction_coeff', 0.1)

    def estimate_time_constant(self):
        """Estimate time constant of actuator"""
        # tau = L/R for electrical time constant
        # Or use mechanical time constant based on inertia
        elec_tau = self.constants.get('inductance', 0.001) / self.constants.get('resistance', 1.0)
        mech_tau = 0.1  # Mechanical time constant (estimated)
        return max(elec_tau, mech_tau)

    def simulate_response(self, commanded_position, dt):
        """Simulate actuator response with dynamics"""
        # First-order system response
        error = commanded_position - self.current_state
        rate = error / self.time_constant

        # Apply saturation limits
        max_rate = self.constants.get('max_velocity', 1.0)  # rad/s
        rate = np.clip(rate, -max_rate, max_rate)

        # Update state
        self.current_velocity = rate
        self.current_state += rate * dt

        return self.current_state

    def add_actuator_noise(self, command):
        """Add realistic actuator noise and limitations"""
        # Current noise
        current_noise_std = 0.01  # Amperes
        noisy_command = command + np.random.normal(0, current_noise_std)

        # Current limits
        max_current = self.constants.get('max_current', 10.0)
        noisy_command = np.clip(noisy_command, -max_current, max_current)

        # Torque limits
        max_torque = self.constants.get('torque_constant', 1.0) * max_current
        noisy_command = np.clip(noisy_command, -max_torque, max_torque)

        return noisy_command

class ActuatorCompensator:
    def __init__(self, actuator_model):
        self.model = actuator_model
        self.command_history = []
        self.feedback_history = []

    def compensate_command(self, desired_position, current_position):
        """Compensate command for actuator dynamics"""
        # Predict what the actuator will actually achieve
        predicted_position = self.model.simulate_response(desired_position, dt=0.02)  # 50Hz control

        # Calculate compensation based on error
        error = desired_position - predicted_position
        compensated_command = desired_position + error * 1.2  # Feedforward gain

        return compensated_command
```

## Fine-Tuning and Adaptation

### Online Adaptation
```python
import torch
import torch.nn as nn
import numpy as np

class OnlineAdaptation:
    def __init__(self, policy_network, adaptation_lr=1e-5):
        self.policy_network = policy_network
        self.adaptation_lr = adaptation_lr
        self.optimizer = torch.optim.Adam(policy_network.parameters(), lr=adaptation_lr)

        # Buffer for recent experiences
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': []
        }
        self.buffer_size = 1000

    def update_policy(self, state, action, reward, next_state):
        """Update policy based on real-world experience"""
        # Add to buffer
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)

        # Trim buffer if too large
        if len(self.buffer['states']) > self.buffer_size:
            for key in self.buffer:
                self.buffer[key] = self.buffer[key][-self.buffer_size:]

        # Perform adaptation step if enough data
        if len(self.buffer['states']) > 100:
            self._adaptation_step()

    def _adaptation_step(self):
        """Perform a single adaptation step"""
        # Sample from buffer
        indices = np.random.choice(len(self.buffer['states']), size=32, replace=False)

        states = torch.FloatTensor([self.buffer['states'][i] for i in indices])
        actions = torch.FloatTensor([self.buffer['actions'][i] for i in indices])
        rewards = torch.FloatTensor([self.buffer['rewards'][i] for i in indices])
        next_states = torch.FloatTensor([self.buffer['next_states'][i] for i in indices])

        # Compute loss (example: policy gradient with real rewards)
        predicted_actions = self.policy_network(states)
        action_diff = (predicted_actions - actions) ** 2
        adaptation_loss = torch.mean(action_diff * rewards.unsqueeze(1))

        # Update network
        self.optimizer.zero_grad()
        adaptation_loss.backward()
        self.optimizer.step()

class MetaLearningAdapter:
    def __init__(self, base_policy, meta_lr=1e-4):
        self.base_policy = base_policy
        self.meta_lr = meta_lr
        self.task_embedding_dim = 64

        # Task-specific embedding network
        self.task_encoder = nn.Sequential(
            nn.Linear(100, 128),  # Input: environment features
            nn.ReLU(),
            nn.Linear(128, self.task_embedding_dim)
        )

        # Policy adaptation network
        self.adaptation_network = nn.Sequential(
            nn.Linear(self.task_embedding_dim + base_policy.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, base_policy.output_dim)
        )

    def adapt_to_task(self, task_features, num_adaptation_steps=5):
        """Adapt policy to new task using meta-learning"""
        task_embedding = self.task_encoder(torch.FloatTensor(task_features))

        # Adapt for several steps
        for _ in range(num_adaptation_steps):
            # Sample task-specific data
            # Update adaptation network
            pass

        return task_embedding
```

## Validation and Testing Framework

### Comprehensive Testing Pipeline
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class SimToRealValidator:
    def __init__(self, sim_env, real_robot):
        self.sim_env = sim_env
        self.real_robot = real_robot
        self.metrics = {}

    def run_validation_tests(self, test_trajectories):
        """Run comprehensive validation tests"""
        results = {
            'tracking_accuracy': [],
            'stability_metrics': [],
            'safety_violations': [],
            'performance_comparison': []
        }

        for trajectory in test_trajectories:
            # Test in simulation
            sim_result = self.test_in_simulation(trajectory)

            # Test in reality (if safe)
            real_result = self.test_in_reality(trajectory)

            # Compare results
            comparison = self.compare_sim_real(sim_result, real_result)
            results['performance_comparison'].append(comparison)

            # Check safety
            safety_violations = self.check_safety_violations(real_result)
            results['safety_violations'].append(safety_violations)

        self.metrics = self.calculate_validation_metrics(results)
        return self.metrics

    def test_in_simulation(self, trajectory):
        """Test trajectory in simulation"""
        # Execute trajectory in sim
        # Record states, actions, rewards
        pass

    def test_in_reality(self, trajectory):
        """Test trajectory in real robot (with safety)"""
        # Implement safe execution with emergency stops
        # Record actual performance
        pass

    def compare_sim_real(self, sim_result, real_result):
        """Compare simulation vs real performance"""
        # Calculate similarity metrics
        tracking_error = np.mean(np.abs(sim_result['positions'] - real_result['positions']))

        # Statistical similarity test
        ks_statistic, p_value = stats.ks_2samp(
            sim_result['velocities'], real_result['velocities']
        )

        return {
            'tracking_error': tracking_error,
            'ks_p_value': p_value,
            'similarity_score': p_value if p_value > 0.05 else 0.0  # If p > 0.05, distributions are similar
        }

    def check_safety_violations(self, real_result):
        """Check for safety violations in real test"""
        violations = []

        # Check joint limits
        if np.any(real_result['positions'] > self.real_robot.joint_limits[1]):
            violations.append("Joint position limit exceeded")

        # Check velocities
        if np.any(np.abs(real_result['velocities']) > self.real_robot.max_velocities):
            violations.append("Velocity limit exceeded")

        # Check forces/torques
        if np.any(np.abs(real_result['torques']) > self.real_robot.max_torques):
            violations.append("Torque limit exceeded")

        return violations

    def calculate_validation_metrics(self, results):
        """Calculate overall validation metrics"""
        metrics = {}

        # Performance similarity
        similarity_scores = [r['similarity_score'] for r in results['performance_comparison']]
        metrics['average_similarity'] = np.mean(similarity_scores)
        metrics['similarity_std'] = np.std(similarity_scores)

        # Safety metrics
        total_violations = sum(len(v) for v in results['safety_violations'])
        metrics['safety_score'] = 1.0 - (total_violations / len(results['safety_violations']))

        # Performance metrics
        tracking_errors = [r['tracking_error'] for r in results['performance_comparison']]
        metrics['average_tracking_error'] = np.mean(tracking_errors)

        return metrics
```

## Best Practices for Sim-to-Real Transfer

### 1. Progressive Domain Randomization
- Start with narrow randomization ranges
- Gradually increase ranges during training
- Monitor performance to avoid excessive randomization

### 2. Systematic Validation
- Test on increasingly complex scenarios
- Validate safety-critical behaviors first
- Use statistical tests to verify distribution similarity

### 3. Conservative Deployment
- Start with reduced performance parameters
- Gradually increase capabilities
- Maintain human oversight during early deployment

### 4. Continuous Monitoring
- Monitor performance degradation over time
- Retrain policies as needed
- Update system identification models

## Troubleshooting Common Issues

### 1. Performance Degradation
- **Symptoms**: Policy performs well in sim but poorly in reality
- **Solutions**:
  - Increase domain randomization coverage
  - Collect real-world data for fine-tuning
  - Revisit system identification

### 2. Safety Violations
- **Symptoms**: Robot violates safety constraints in reality
- **Solutions**:
  - Add safety constraints to simulation
  - Implement robust safety controllers
  - Reduce performance parameters conservatively

### 3. Unmodeled Dynamics
- **Symptoms**: Unexpected behaviors in reality
- **Solutions**:
  - Conduct thorough system identification
  - Add unmodeled dynamics to simulation
  - Use robust control techniques

## Summary

Sim-to-real transfer techniques bridge the gap between simulation and reality:
- Domain randomization for robust policy learning
- System identification for accurate modeling
- Latency and sensor noise compensation
- Online adaptation for real-world conditions
- Comprehensive validation frameworks

Successful sim-to-real transfer requires careful attention to modeling accuracy, safety, and systematic validation.

## Exercises

1. Implement domain randomization for a simple robot simulation
2. Conduct system identification on a physical robot
3. Validate sim-to-real transfer with safety constraints
4. Develop an online adaptation system for changing conditions