---
sidebar_position: 2
---

# Gazebo Simulation Setup

## Learning Objectives
- Install and configure Gazebo for robotics simulation
- Understand Gazebo's architecture and core concepts
- Set up a basic simulation environment
- Integrate ROS 2 with Gazebo for robot simulation

## Introduction to Gazebo

Gazebo is a 3D simulation environment that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics research and development for testing algorithms, robot designs, and control systems without the need for physical hardware.

### Key Features of Gazebo
- **Realistic Physics**: Accurate simulation of rigid body dynamics, collisions, and contacts
- **Sensor Simulation**: Support for various sensors including cameras, LiDAR, IMUs, and more
- **Flexible Models**: Support for complex robot and environment models
- **Plugin Architecture**: Extensible through plugins for custom functionality
- **ROS Integration**: Seamless integration with ROS and ROS 2

## Installing Gazebo

For ROS 2 Humble Hawksbill (Ubuntu 22.04):
```bash
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-dev
```

### Alternative: Installing Gazebo Garden
Gazebo Garden is the latest standalone version:
```bash
# Add Gazebo repository
sudo curl -sSL http://get.gazebosim.org | sh
sudo apt install gz-garden
```

## Basic Gazebo Concepts

### Worlds
World files define the simulation environment including:
- Physics properties
- Models and their initial positions
- Lighting and environment settings
- Plugins

Example world file:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics engine -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Include a model from Gazebo's model database -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Models
Models represent objects in the simulation including robots, obstacles, and environment elements. They contain:
- **Links**: Rigid bodies with visual, collision, and inertial properties
- **Joints**: Connections between links
- **Sensors**: Simulated sensor devices
- **Plugins**: Custom functionality

## Starting Gazebo

### Launch Gazebo GUI
```bash
# Using gazebo_ros package
ros2 launch gazebo_ros empty_world.launch.py

# Or with a specific world file
ros2 launch gazebo_ros empty_world.launch.py world_name:=path/to/world.sdf
```

### Launch Gazebo Server (headless)
```bash
# For automated testing or CI/CD
gz sim -s -r empty.sdf
# or with ROS 2
ros2 launch gazebo_ros empty_world.launch.py gui_required:=false
```

## ROS 2 Integration

### Gazebo ROS Packages
The `gazebo_ros_pkgs` provide the bridge between ROS 2 and Gazebo:

- **gazebo_ros**: Core ROS 2 integration
- **gazebo_plugins**: Common plugins for sensors and actuators
- **gazebo_msgs**: ROS 2 messages for Gazebo interaction

### Launching with ROS 2
```xml
<launch>
  <!-- Start Gazebo -->
  <include file="$(find-pkg-share gazebo_ros)/launch/empty_world.launch.py">
    <arg name="world" value="$(find-pkg-share my_robot_gazebo)/worlds/my_world.sdf"/>
    <arg name="gui" value="true"/>
  </include>

  <!-- Spawn robot in Gazebo -->
  <node pkg="gazebo_ros" exec="spawn_entity.py"
        args="-topic robot_description -entity my_robot -x 0 -y 0 -z 1">
  </node>

  <!-- Launch robot state publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher"
        name="robot_state_publisher">
    <param name="robot_description" value="$(var robot_description)"/>
  </node>
</launch>
```

## Creating a Simple Robot Model

### URDF to SDF Conversion
While Gazebo uses SDF, you can use URDF models by converting them or using the `libgazebo_ros_control.so` plugin.

Example URDF for a simple differential drive robot:
```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
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
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_robot_description)/config/diff_drive_controller.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

### Controller Configuration
Create a controller configuration file:

```yaml
# config/diff_drive_controller.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

diff_drive_controller:
  type: diff_drive_controller/DiffDriveController

diff_drive_controller:
  ros__parameters:
    left_wheel_names: ["left_wheel_joint"]
    right_wheel_names: ["right_wheel_joint"]

    wheel_separation: 0.4
    wheel_radius: 0.1

    # Publish all kinds of topics
    publish_rate: 50.0
    odom_frame_id: odom
    base_frame_id: base_link
    pose_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
    twist_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]

    # Velocity and acceleration limits
    linear.x.has_velocity_limits: true
    linear.x.has_acceleration_limits: true
    linear.x.has_jerk_limits: false
    linear.x.max_velocity: 1.0
    linear.x.min_velocity: -1.0
    linear.x.max_acceleration: 2.0
    linear.x.min_acceleration: -2.0

    angular.z.has_velocity_limits: true
    angular.z.has_acceleration_limits: true
    angular.z.has_jerk_limits: false
    angular.z.max_velocity: 1.0
    angular.z.min_velocity: -1.0
    angular.z.max_acceleration: 2.0
    angular.z.min_acceleration: -2.0
```

## Launching Your Robot in Gazebo

Create a launch file to bring everything together:

```python
# launch/robot_gazebo.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'empty_world.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'my_world.sdf'
            ])
        }.items()
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'robot.urdf.xacro'
            ])}
        ]
    )

    # Spawn entity
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot'
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

## Troubleshooting Common Issues

### 1. Robot Not Spawning
- Check that the URDF is properly loaded
- Verify the robot name matches between spawn command and URDF
- Check for parsing errors in the URDF

### 2. Controllers Not Working
- Ensure the controller configuration is properly loaded
- Check that joint names match between URDF and controller config
- Verify the controller manager is running

### 3. Performance Issues
- Reduce physics update rate for better performance
- Simplify collision geometries
- Use simpler visual models

## Best Practices

1. **Model Organization**: Keep robot models in separate packages
2. **Parameter Configuration**: Use launch arguments for flexibility
3. **Simulation vs Real**: Design with sim-to-real transfer in mind
4. **Performance**: Balance accuracy with simulation speed
5. **Testing**: Validate controllers in simulation before real hardware

## Summary

Gazebo provides a powerful simulation environment for robotics development. Key concepts include:
- World files define the simulation environment
- Robot models can be defined in URDF and used in Gazebo
- ROS 2 integration allows for seamless simulation workflows
- Controllers connect the simulation to ROS 2 topics
- Launch files orchestrate the complete simulation setup

## Exercises

1. Create a simple world file with multiple objects
2. Build a URDF model for a custom robot
3. Set up a complete simulation launch file
4. Implement a simple controller for your simulated robot