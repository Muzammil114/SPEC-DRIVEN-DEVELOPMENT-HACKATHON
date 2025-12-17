---
sidebar_position: 5
---

# Launch Files and Parameter Management

## Learning Objectives
- Understand the purpose and structure of ROS 2 launch files
- Learn to create complex launch files with multiple nodes and configurations
- Master parameter management and dynamic reconfiguration
- Implement best practices for launch file organization

## Introduction to Launch Files

Launch files in ROS 2 allow you to start multiple nodes with specific configurations from a single command. They replace the ROS 1 launch system with a more powerful and flexible Python-based approach.

### Why Use Launch Files?

- **Convenience**: Start multiple nodes with one command
- **Configuration**: Set parameters and remappings for nodes
- **Flexibility**: Conditional execution and parameter substitution
- **Reusability**: Share configurations across different environments
- **Organization**: Group related nodes into logical units

## Launch File Structure

ROS 2 launch files are Python scripts that define a `generate_launch_description()` function:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    # Create launch description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    ))

    # Add nodes to launch
    ld.add_action(Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='sim',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    ))

    return ld
```

## Launch Arguments and Substitutions

### Launch Arguments

Launch arguments allow users to customize launch behavior:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Simple argument with default
        DeclareLaunchArgument(
            'robot_name',
            default_value='robot1',
            description='Name of the robot'
        ),

        # Argument with choices
        DeclareLaunchArgument(
            'robot_model',
            default_value='turtlebot3_waffle',
            choices=['turtlebot3_waffle', 'turtlebot3_burger', 'custom_robot'],
            description='Robot model to use'
        ),

        # Boolean argument
        DeclareLaunchArgument(
            'use_camera',
            default_value='true',
            description='Whether to start camera node'
        ),

        # Node that uses launch arguments
        Node(
            package='my_robot_package',
            executable='robot_driver',
            name=[LaunchConfiguration('robot_name'), '_driver'],
            parameters=[
                {'robot_model': LaunchConfiguration('robot_model')},
                {'use_camera': LaunchConfiguration('use_camera')}
            ]
        )
    ])
```

### Common Substitutions

Launch files support various substitutions for dynamic values:

```python
from launch.substitutions import (
    LaunchConfiguration,
    TextSubstitution,
    PathJoinSubstitution,
    PythonExpression
)
from launch_ros.substitutions import FindPackageShare

# Text substitution
TextSubstitution(text='my_value')

# Path joining (useful for file paths)
PathJoinSubstitution([
    FindPackageShare('my_package'),
    'config',
    'my_config.yaml'
])

# Python expressions
PythonExpression(['"', LaunchConfiguration('param1'), '" == "value"'])
```

## Advanced Launch Concepts

### Conditional Launch Actions

```python
from launch import LaunchDescription, LaunchCondition
from launch.actions import IncludeLaunchDescription, SetLaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    declare_use_camera = DeclareLaunchArgument(
        'use_camera',
        default_value='true',
        description='Whether to start camera node'
    )

    # Conditionally include another launch file
    include_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('camera_package'),
            '/launch/camera.launch.py'
        ]),
        condition=IfCondition(LaunchConfiguration('use_camera'))
    )

    return LaunchDescription([
        declare_use_camera,
        include_camera_launch
    ])
```

### Launch File Composition

Include other launch files to create modular configurations:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Include robot hardware launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('my_robot_description'),
                    'launch',
                    'robot_hardware.launch.py'
                ])
            ])
        ),

        # Include navigation stack
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('nav2_bringup'),
                    'launch',
                    'navigation_launch.py'
                ])
            ]),
            launch_arguments={
                'use_sim_time': 'false',
                'params_file': PathJoinSubstitution([
                    FindPackageShare('my_robot_navigation'),
                    'config',
                    'nav2_params.yaml'
                ])
            }.items()
        )
    ])
```

## Parameter Management

### Setting Parameters in Launch Files

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='navigation_node',
            name='nav2_bringup',
            parameters=[
                # Direct value
                {'use_sim_time': False},

                # From launch configuration
                {'robot_base_frame': LaunchConfiguration('base_frame')},

                # From YAML file
                PathJoinSubstitution([
                    FindPackageShare('my_robot_config'),
                    'config',
                    'navigation.yaml'
                ]),

                # Multiple parameters in dict
                {
                    'planner_server.plugin_names': ['GridBased'],
                    'planner_server.GridBased.type': 'nav2_navfn_planner/NavfnPlanner',
                    'planner_server.GridBased.expected_planner_frequency': 20.0
                }
            ]
        )
    ])
```

### Parameter Files (YAML)

Create separate YAML files for complex parameter configurations:

```yaml
# config/navigation.yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: false
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

controller_server:
  ros__parameters:
    controller_frequency: 20.0
    use_sim_time: false
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller/RegulatedPurePursuitController"
      desired_linear_vel: 0.5
      lookahead_dist: 0.6
```

## Complex Launch Example

Here's a comprehensive launch file for a humanoid robot:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import ReplaceString

def generate_launch_description():
    # Declare launch arguments
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam = LaunchConfiguration('slam')
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    default_bt_xml_filename = LaunchConfiguration('default_bt_xml_filename')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')

    # Lifecycle manager for navigation
    lifecycle_nodes = [
        'controller_server',
        'planner_server',
        'recoveries_server',
        'bt_navigator',
        'waypoint_follower'
    ]

    # Launch navigation lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': autostart},
                    {'node_names': lifecycle_nodes}]
    )

    # Group actions if namespace is used
    bringup_cmd_group = GroupAction(
        condition=IfCondition(use_namespace),
        actions=[
            PushRosNamespace(
                namespace),
            lifecycle_manager,
        ]
    )

    # Navigation server nodes
    controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel_nav')],
        respawn=use_respawn,
        respawn_delay=2.0,
        arguments=['--ros-args', '--log-level', log_level]
    )

    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        respawn=use_respawn,
        respawn_delay=2.0,
        arguments=['--ros-args', '--log-level', log_level]
    )

    recoveries_server_node = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        respawn=use_respawn,
        respawn_delay=2.0,
        arguments=['--ros-args', '--log-level', log_level]
    )

    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        respawn=use_respawn,
        respawn_delay=2.0,
        arguments=['--ros-args', '--log-level', log_level]
    )

    waypoint_follower_node = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        respawn=use_respawn,
        respawn_delay=2.0,
        arguments=['--ros-args', '--log-level', log_level]
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Top-level namespace'),
        DeclareLaunchArgument(
            'use_namespace', default_value='false',
            description='Whether to apply a namespace to the navigation stack'),
        DeclareLaunchArgument(
            'slam', default_value='False',
            choices=['True', 'false'],
            description='Enable SLAM'),
        DeclareLaunchArgument(
            'map',
            default_value=PathJoinSubstitution([
                FindPackageShare('my_robot_config'),
                'maps',
                'turtlebot3_world.yaml'
            ]),
            description='Full path to map file to load'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('my_robot_config'),
                'config',
                'nav2_params.yaml'
            ]),
            description='Full path to the ROS2 parameters file to use for all launched nodes'),
        DeclareLaunchArgument(
            'default_bt_xml_filename',
            default_value=PathJoinSubstitution([
                FindPackageShare('nav2_bt_navigator'),
                'behavior_trees',
                'navigate_w_replanning_and_recovery.xml'
            ]),
            description='Full path to the behavior tree xml file to use'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            choices=['true', 'false'],
            description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument(
            'autostart', default_value='true',
            choices=['true', 'false'],
            description='Automatically startup the nav2 stack'),
        DeclareLaunchArgument(
            'use_composition', default_value='False',
            choices=['True', 'false'],
            description='Whether to use composed bringup'),
        DeclareLaunchArgument(
            'use_respawn', default_value='False',
            choices=['True', 'false'],
            description='Whether to respawn if a node crashes'),
        DeclareLaunchArgument(
            'log_level', default_value='info',
            description='Logging level'),

        # Launch nodes
        controller_server_node,
        planner_server_node,
        recoveries_server_node,
        bt_navigator_node,
        waypoint_follower_node,
        lifecycle_manager,
        bringup_cmd_group,
    ])
```

## Launch File Best Practices

### 1. Organize Launch Files Logically

```
my_robot_package/
├── launch/
│   ├── robot.launch.py          # Main robot launch
│   ├── navigation.launch.py     # Navigation-specific launch
│   ├── sensors.launch.py        # Sensor-specific launch
│   └── simulation.launch.py     # Simulation-specific launch
```

### 2. Use Descriptive Names and Documentation

```python
# Good: Clear and descriptive
DeclareLaunchArgument(
    'robot_base_frame',
    default_value='base_link',
    description='The name of the robot base frame for tf transformations'
)

# Avoid: Unclear names
DeclareLaunchArgument(
    'frame',
    default_value='base_link',
    description='Frame name'
)
```

### 3. Handle Dependencies Properly

```python
# Launch nodes in correct order with dependencies
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart

# Launch robot hardware first
robot_driver = Node(
    package='my_robot_hardware',
    executable='driver',
    name='robot_driver'
)

# Launch controller after driver is running
controller = Node(
    package='my_robot_control',
    executable='controller',
    name='controller'
)

# Register event handler to start controller after driver
delay_controller = RegisterEventHandler(
    OnProcessStart(
        target_action=robot_driver,
        on_start=[controller]
    )
)
```

### 4. Use Parameter Files for Complex Configurations

For complex parameter sets, use YAML files instead of embedding everything in Python:

```python
# Good: Use YAML files for complex parameters
Node(
    package='my_robot_package',
    executable='complex_node',
    parameters=[PathJoinSubstitution([
        FindPackageShare('my_robot_config'),
        'config',
        'complex_params.yaml'
    ])]
)

# Avoid: Large parameter dictionaries in launch files
Node(
    package='my_robot_package',
    executable='complex_node',
    parameters=[{
        'param1': 'value1',
        'param2': 'value2',
        # ... many more parameters
    }]
)
```

## Summary

Launch files are essential for ROS 2 robot system deployment. Key concepts include:

- **Structure**: Python-based launch files with `generate_launch_description()`
- **Launch Arguments**: Customizable parameters for launch files
- **Substitutions**: Dynamic value insertion using various substitution types
- **Conditional Execution**: If/else logic in launch files
- **Composition**: Including other launch files for modularity
- **Parameter Management**: Setting parameters from various sources
- **Best Practices**: Organization, documentation, and dependency handling

## Exercises

1. Create a launch file that starts a robot simulation with multiple sensors
2. Implement a launch file with conditional node launching based on arguments
3. Create parameter files for different robot configurations (indoor, outdoor, etc.)
4. Design a modular launch system that can be reused across different robot platforms