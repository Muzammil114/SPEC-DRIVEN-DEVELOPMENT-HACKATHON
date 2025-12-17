---
sidebar_position: 6
---

# URDF (Unified Robot Description Format) for Humanoid Robots

## Learning Objectives
- Understand the structure and components of URDF files
- Learn to create robot models for humanoid robots
- Master the creation of complex kinematic chains and joint structures
- Implement visual and collision properties for realistic simulation

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used to describe robots in ROS. It defines the physical structure of a robot including links, joints, visual elements, collision properties, and inertial properties.

### URDF Components

A URDF file typically contains:
- **Links**: Rigid bodies of the robot
- **Joints**: Connections between links
- **Visual elements**: How the robot appears in simulation
- **Collision elements**: Collision properties for physics simulation
- **Inertial properties**: Mass, center of mass, and moments of inertia

## Basic URDF Structure

Here's a minimal URDF example:

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- A child link connected by a joint -->
  <link name="upper_body">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.3"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="body_joint" type="fixed">
    <parent link="base_link"/>
    <child link="upper_body"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>
</robot>
```

## Links in URDF

Links represent rigid bodies in the robot. Each link can have:

- **Visual**: How the link appears in simulation
- **Collision**: Properties used for collision detection
- **Inertial**: Mass and inertial properties for physics simulation

### Visual Properties

```xml
<link name="example_link">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Supported geometry types -->
      <box size="1 1 1"/>                 <!-- Box: width, depth, height -->
      <!-- <cylinder radius="0.1" length="0.5"/> -->  <!-- Cylinder -->
      <!-- <sphere radius="0.1"/> -->                 <!-- Sphere -->
      <!-- <mesh filename="package://path/to/mesh.stl"/> -->  <!-- 3D mesh -->
    </geometry>
    <material name="example_material">
      <color rgba="0.8 0.2 0.2 1.0"/>
    </material>
  </visual>
</link>
```

### Collision Properties

```xml
<link name="example_link">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Often simpler geometry than visual for performance -->
      <box size="1 1 1"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

```xml
<link name="example_link">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0"
             iyy="0.1" iyz="0.0"
             izz="0.1"/>
  </inertial>
</link>
```

## Joints in URDF

Joints connect links and define their relative motion. Common joint types include:

- **Fixed**: No motion allowed
- **Revolute**: Rotational motion around a single axis
- **Continuous**: Like revolute but unlimited rotation
- **Prismatic**: Linear motion along a single axis
- **Planar**: Motion on a plane
- **Floating**: Free motion in 6DOF

### Joint Examples

```xml
<!-- Revolute joint (like an elbow) -->
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="lower_arm"/>
  <origin xyz="0 0 -0.2" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Rotation axis -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>

<!-- Fixed joint (no movement) -->
<joint name="sensor_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.1 0 0.2" rpy="0 0 0"/>
</joint>

<!-- Continuous joint (unlimited rotation) -->
<joint name="continuous_joint" type="continuous">
  <parent link="base"/>
  <child link="wheel"/>
  <axis xyz="0 0 1"/>
</joint>
```

## Humanoid Robot URDF Example

Here's a simplified example of a humanoid robot structure:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.15 0.3"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.15 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
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
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="50" velocity="1"/>
  </joint>

  <!-- Left arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="dark_grey">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.1 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>
</robot>
```

## Using Xacro for Complex URDFs

Xacro is a macro system for URDF that allows for parameterization, reuse, and cleaner definitions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="mass_shoulder" value="1.0" />
  <xacro:property name="mass_upper_arm" value="1.0" />
  <xacro:property name="mass_lower_arm" value="0.5" />

  <!-- Macro for creating arm links -->
  <xacro:macro name="arm" params="side reflect">
    <!-- Shoulder -->
    <link name="${side}_shoulder">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <material name="dark_grey">
          <color rgba="0.3 0.3 0.3 1"/>
        </material>
      </visual>
      <inertial>
        <mass value="${mass_shoulder}"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_pitch" type="revolute">
      <parent link="torso"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${reflect * 0.1} 0.1 0.1" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-2 * M_PI}" upper="${2 * M_PI}" effort="100" velocity="1"/>
    </joint>

    <!-- Upper arm -->
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="0.04" length="0.25"/>
        </geometry>
        <origin xyz="0 0 -0.125" rpy="0 0 0"/>
        <material name="skin">
          <color rgba="0.8 0.6 0.4 1"/>
        </material>
      </visual>
      <inertial>
        <mass value="${mass_upper_arm}"/>
        <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_yaw" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 -0.05" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100" velocity="1"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro to create both arms -->
  <xacro:arm side="left" reflect="1" />
  <xacro:arm side="right" reflect="-1" />

  <!-- Other robot components would go here -->

</robot>
```

## Working with URDF in ROS 2

### Loading URDF in a Node

```python
import rclpy
from rclpy.node import Node
from urdf_parser_py.urdf import URDF

class URDFAnalyzer(Node):
    def __init__(self):
        super().__init__('urdf_analyzer')

        # Load URDF from parameter
        self.declare_parameter('robot_description', '')
        robot_desc = self.get_parameter('robot_description').value

        if robot_desc:
            # Parse the URDF
            self.robot = URDF.from_xml_string(robot_desc)
            self.get_logger().info(f'Loaded robot model: {self.robot.name}')

            # Analyze the robot structure
            self.get_logger().info(f'Number of links: {len(self.robot.links)}')
            self.get_logger().info(f'Number of joints: {len(self.robot.joints)}')
```

### Publishing Robot State

To visualize the robot in RViz, you need to publish the joint states:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Timer to publish joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Sample joint names (these should match your URDF)
        self.joint_names = [
            'left_shoulder_pitch', 'left_shoulder_yaw',
            'right_shoulder_pitch', 'right_shoulder_yaw'
        ]

        # Current joint positions
        self.joint_positions = [0.0] * len(self.joint_names)

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## URDF Best Practices

### 1. Consistent Naming Convention
```xml
<!-- Good: Descriptive names -->
<link name="left_arm_upper_link"/>
<joint name="left_arm_shoulder_pitch_joint"/>

<!-- Avoid: Ambiguous names -->
<link name="l1"/>
<joint name="j1"/>
```

### 2. Proper Inertial Properties
- Use realistic mass values
- Calculate inertias for basic shapes:
  - Box: Ixx = 1/12 * m * (h² + d²)
  - Cylinder: Izz = 1/2 * m * r², Ixx = Iyy = 1/12 * m * (3*r² + h²)
  - Sphere: Ixx = Iyy = Izz = 2/5 * m * r²

### 3. Separate Visual and Collision Geometries
- Use detailed meshes for visual elements
- Use simpler shapes for collision detection for better performance

### 4. Use Xacro for Complex Models
- Parameterize common values
- Create reusable macros for symmetrical parts
- Use mathematical expressions for complex calculations

## Validating URDF

To check your URDF file:
```bash
# Check syntax
check_urdf /path/to/robot.urdf

# Visualize the robot structure
urdf_to_graphiz /path/to/robot.urdf
```

## Summary

URDF is fundamental for describing robot models in ROS 2. Key concepts include:

- **Links**: Rigid bodies with visual, collision, and inertial properties
- **Joints**: Connections between links with specific motion constraints
- **Xacro**: Macro system for creating reusable and parameterized URDFs
- **Best Practices**: Consistent naming, proper inertial values, and performance considerations

## Exercises

1. Create a URDF for a simple wheeled robot with sensors
2. Design a humanoid arm using Xacro macros
3. Implement joint state publishing for your robot model
4. Create both visual and collision meshes for a complex link