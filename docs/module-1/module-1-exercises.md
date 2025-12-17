---
sidebar_position: 7
---

# Module 1: Exercises and Review

## Learning Objectives Review

By completing Module 1, you should be able to:
- Understand ROS 2 architecture and DDS middleware
- Create and connect ROS 2 nodes using rclpy
- Implement publishers, subscribers, services, and actions
- Configure nodes using launch files and parameters
- Describe robot models using URDF for humanoid robots

## Conceptual Questions

### 1. ROS 2 Architecture
1. What are the key differences between ROS 1 and ROS 2 architecture?
2. Explain the role of DDS in ROS 2 communication.
3. What are Quality of Service (QoS) profiles and why are they important?
4. Describe the four main communication patterns in ROS 2 and when to use each.

### 2. Node Development
1. What is the basic structure of a ROS 2 Python node?
2. How do you properly initialize and clean up a ROS 2 node?
3. Explain the difference between synchronous and asynchronous service calls.
4. When would you use an Action instead of a Service?

### 3. URDF and Robot Modeling
1. What are the three main components of a URDF link?
2. Explain the difference between visual and collision properties in URDF.
3. What are the different joint types available in URDF?
4. Why is it important to specify accurate inertial properties in URDF?

## Practical Exercises

### Exercise 1: Basic Publisher and Subscriber
Create a publisher node that publishes temperature data and a subscriber node that processes this data to determine if heating/cooling is needed.

**Requirements:**
- Use a custom message type for temperature data (temperature value and timestamp)
- Implement proper error handling
- Add parameters to configure the temperature threshold
- Use appropriate QoS settings

### Exercise 2: Service Implementation
Implement a robot configuration service that allows remote configuration of robot parameters.

**Requirements:**
- Create a custom service definition for robot configuration
- Implement server that can set multiple parameters
- Implement client that sends configuration requests
- Add validation to prevent invalid configurations

### Exercise 3: Action Server
Create an action server for a robot arm that performs a pick-and-place operation.

**Requirements:**
- Define an action for pick-and-place operations
- Implement the action server with proper feedback
- Handle preemption and cancellation requests
- Include error handling for failed operations

### Exercise 4: Launch File Configuration
Create a modular launch system for a humanoid robot with different configurations.

**Requirements:**
- Create separate launch files for sensors, actuators, and control
- Implement launch arguments for different robot configurations
- Use conditional launching based on arguments
- Include parameter files for different scenarios

### Exercise 5: URDF Model
Design a URDF model for a simple humanoid robot with at least 10 joints.

**Requirements:**
- Include base, torso, head, and two arms
- Use Xacro for parameterization and reuse
- Define proper inertial properties
- Include both visual and collision elements

## Programming Challenges

### Challenge 1: Robot Health Monitor
Create a node that monitors the health of other nodes in the system and provides status reports.

**Advanced Requirements:**
- Monitor node lifecycles and connection status
- Implement heartbeat mechanism
- Provide service interface for health queries
- Publish health status to a topic

### Challenge 2: Dynamic Parameter Server
Implement a parameter server that allows runtime reconfiguration of multiple nodes.

**Advanced Requirements:**
- Centralized parameter management
- Configuration validation and constraints
- Parameter backup and restore functionality
- Web interface for parameter management

## Solutions and Hints

### Exercise 1 Solution Outline
```python
# Temperature publisher
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64  # or custom message

class TemperaturePublisher(Node):
    def __init__(self):
        super().__init__('temperature_publisher')
        self.publisher = self.create_publisher(Float64, 'temperature', 10)
        self.timer = self.create_timer(1.0, self.publish_temp)

    def publish_temp(self):
        # Generate simulated temperature data
        temp_msg = Float64()
        temp_msg.data = 20.0 + (self.get_clock().now().nanoseconds % 10)
        self.publisher.publish(temp_msg)

# Temperature subscriber
class TemperatureSubscriber(Node):
    def __init__(self):
        super().__init__('temperature_subscriber')
        self.subscription = self.create_subscription(
            Float64, 'temperature', self.temp_callback, 10)
        self.declare_parameter('threshold', 25.0)

    def temp_callback(self, msg):
        threshold = self.get_parameter('threshold').value
        if msg.data > threshold:
            self.get_logger().info(f'Temperature {msg.data} exceeds threshold!')
        else:
            self.get_logger().info(f'Temperature {msg.data} is normal')
```

### Exercise 5 Solution Outline
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.25 0.2 0.5"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.25 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="base_torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- More joints and links for arms and legs would follow... -->
</robot>
```

## Summary

Module 1 covered the foundational concepts of ROS 2 development:
- ROS 2 architecture and DDS middleware
- Node development with rclpy
- Communication patterns (topics, services, actions)
- Launch files and parameter management
- URDF for robot modeling

These concepts form the basis for all subsequent modules and provide the tools needed to build complex robotic systems.

## Key Takeaways

1. ROS 2's DDS-based architecture provides robust, scalable communication
2. Proper use of communication patterns is essential for system design
3. Launch files and parameters enable flexible system configuration
4. Accurate URDF models are crucial for simulation and control
5. Following best practices ensures maintainable and efficient code