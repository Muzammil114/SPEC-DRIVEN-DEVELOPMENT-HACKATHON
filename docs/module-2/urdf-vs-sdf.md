---
sidebar_position: 4
---

# URDF vs SDF: Simulation Format Comparison

## Learning Objectives
- Understand the differences between URDF and SDF formats
- Learn when to use each format for robotics applications
- Master the conversion between URDF and SDF when needed
- Implement best practices for both formats

## Introduction to URDF and SDF

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are both XML-based formats used in robotics, but they serve different primary purposes:

- **URDF**: Primarily used for robot description in ROS/ROS 2, focusing on kinematic and geometric properties
- **SDF**: Used by Gazebo for simulation, including physics, sensors, and plugins

## URDF Overview

URDF is the standard format for describing robots in ROS/ROS 2. It focuses on the physical structure of robots:

```xml
<?xml version="1.0"?>
<robot name="urdf_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Joints define connections between links -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="arm_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.005"/>
    </inertial>
  </link>
</robot>
```

### URDF Strengths
- **ROS Integration**: Native support in ROS/ROS 2 ecosystem
- **Simplicity**: Focused on robot structure, easier to understand
- **Tooling**: Extensive tooling for visualization, kinematics, etc.
- **Xacro**: Powerful macro system for complex models

### URDF Limitations
- **Simulation-specific**: Lacks simulation-specific features
- **Sensors**: No native sensor definitions
- **Plugins**: Cannot define simulation plugins
- **Worlds**: Cannot describe complete simulation environments

## SDF Overview

SDF is designed specifically for simulation environments like Gazebo:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="sdf_world">
    <!-- Physics configuration -->
    <physics name="ode" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Models in the world -->
    <model name="sdf_robot">
      <pose>0 0 0.1 0 0 0</pose>

      <!-- Links -->
      <link name="base_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.2</izz>
          </inertia>
        </inertial>

        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
          </material>
        </visual>

        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>0.1</length>
            </cylinder>
          </geometry>
        </collision>

        <!-- SDF-specific: Sensors -->
        <sensor name="camera" type="camera">
          <camera>
            <horizontal_fov>1.047</horizontal_fov>
            <image>
              <width>640</width>
              <height>480</height>
            </image>
            <clip>
              <near>0.1</near>
              <far>10</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <visualize>true</visualize>
        </sensor>

        <!-- SDF-specific: Plugins -->
        <plugin name="ros_control" filename="libgazebo_ros2_control.so">
          <parameters>$(find my_robot_description)/config/robot_controllers.yaml</parameters>
        </plugin>
      </link>

      <!-- Joints -->
      <joint name="joint1" type="revolute">
        <parent>base_link</parent>
        <child>arm_link</child>
        <pose>0 0 0.05 0 0 0</pose>
        <axis>
          <xyz>0 0 1</xyz>
          <limit>
            <lower>-1.57</lower>
            <upper>1.57</upper>
            <effort>100</effort>
            <velocity>1</velocity>
          </limit>
        </axis>
      </joint>

      <link name="arm_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.01</iyy>
            <iyz>0</iyz>
            <izz>0.005</izz>
          </inertia>
        </inertial>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.3</size>
            </box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.3</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Environment objects -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### SDF Strengths
- **Complete simulation**: Supports entire simulation environments
- **Sensors**: Native support for various sensor types
- **Plugins**: Extensible through plugins
- **Physics**: Detailed physics configuration
- **Worlds**: Can describe complete simulation scenarios

### SDF Limitations
- **ROS Integration**: Less native integration with ROS tooling
- **Complexity**: More complex for simple robot descriptions
- **Learning curve**: More concepts to understand

## Converting Between URDF and SDF

### Using the robot_state_publisher and Gazebo
In practice, you rarely need to manually convert URDF to SDF. Gazebo can load URDF models using plugins:

```xml
<!-- In your URDF, add Gazebo-specific tags -->
<robot name="urdf_with_gazebo_extensions">
  <!-- Standard URDF content -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Gazebo-specific extensions -->
  <gazebo reference="base_link">
    <!-- Material definition -->
    <material>Gazebo/Blue</material>

    <!-- Sensor definition -->
    <sensor name="camera" type="camera">
      <pose>0.1 0 0.1 0 0 0</pose>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

  <!-- Controller plugin -->
  <gazebo>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find my_robot_description)/config/robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

### Manual Conversion Example

**URDF:**
```xml
<joint name="wheel_joint" type="continuous">
  <parent link="base_link"/>
  <child link="wheel_link"/>
  <origin xyz="0 0.2 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
</joint>
```

**Equivalent SDF:**
```xml
<joint name="wheel_joint" type="revolute">
  <parent>base_link</parent>
  <child>wheel_link</child>
  <pose>0 0.2 0 0 0 0</pose>
  <axis>
    <xyz>0 1 0</xyz>
  </axis>
</joint>
```

## Best Practices for Each Format

### URDF Best Practices
1. **Use Xacro**: Parameterize models for reusability
2. **Accurate inertials**: Calculate proper mass and inertia values
3. **Separate visual/collision**: Use simpler collision models
4. **Consistent naming**: Use descriptive, consistent names
5. **Documentation**: Comment complex sections

```xml
<!-- Good: Using Xacro for parameterization -->
<xacro:macro name="wheel" params="prefix reflect">
  <link name="${prefix}_wheel">
    <visual>
      <geometry>
        <cylinder radius="$(wheel_radius)" length="$(wheel_width)"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="$(wheel_radius)" length="$(wheel_width)"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="$(wheel_mass)"/>
      <inertia ixx="..." ixy="0" ixz="0" iyy="..." iyz="0" izz="..."/>
    </inertial>
  </link>
</xacro:macro>
```

### SDF Best Practices
1. **Modular worlds**: Break complex worlds into includes
2. **Physics tuning**: Optimize for your specific use case
3. **Sensor configuration**: Configure for realistic data
4. **Plugin management**: Use appropriate plugins for your needs
5. **Performance**: Balance quality with simulation speed

```xml
<!-- Good: Modular world design -->
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics -->
    <physics name="my_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Robot -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <!-- Objects -->
    <include>
      <uri>model://table</uri>
      <pose>2 0 0 0 0 0</pose>
    </include>
  </world>
</sdf>
```

## When to Use Each Format

### Use URDF When:
- Describing robot kinematic structure
- Working primarily in ROS/ROS 2 ecosystem
- Creating reusable robot models
- Integrating with ROS tools (RViz, MoveIt, etc.)
- Focusing on robot hardware description

### Use SDF When:
- Creating complete simulation environments
- Defining sensor models and properties
- Working with simulation-specific plugins
- Describing complex world scenarios
- Need advanced physics configuration

### Use Both When:
- Simulating robots in ROS/ROS 2
- Need both robot description and simulation features
- Implementing sim-to-real workflows

## Practical Example: Complete Setup

Here's how to use both formats together for a complete robot simulation:

**URDF file (robot.urdf.xacro):**
```xml
<?xml version="1.0"?>
<robot name="complete_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <xacro:macro name="wheel" params="prefix x_pos y_pos">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x_pos} ${y_pos} 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <xacro:wheel prefix="left" x_pos="0" y_pos="0.2"/>
  <xacro:wheel prefix="right" x_pos="0" y_pos="-0.2"/>

  <!-- Gazebo-specific extensions -->
  <gazebo>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find robot_description)/config/diff_drive.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

**Launch file to use the robot in simulation:**
```python
# launch/robot_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'empty_world.launch.py'
                ])
            ])
        ),

        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='both',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_description': PathJoinSubstitution([
                    FindPackageShare('robot_description'),
                    'urdf',
                    'robot.urdf.xacro'
                ])}
            ]
        ),

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'robot',
                '-x', '0', '-y', '0', '-z', '0.1'
            ],
            output='screen'
        )
    ])
```

## Summary

URDF and SDF serve complementary purposes in robotics:
- **URDF**: Robot description for ROS/ROS 2, focused on kinematic structure
- **SDF**: Simulation description for Gazebo, focused on complete simulation environments
- **Integration**: Use both formats together for comprehensive robot simulation
- **Best practices**: Choose the right format for your specific needs

Understanding both formats is essential for effective robotics development, especially when working with simulation.

## Exercises

1. Convert a simple URDF model to SDF format manually
2. Create a URDF model with Gazebo extensions for simulation
3. Design a complete simulation world with multiple robots and objects
4. Implement a sensor configuration in both URDF and SDF formats