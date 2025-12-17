---
sidebar_position: 6
---

# Unity for High-Fidelity Visualization and Human-Robot Interaction

## Learning Objectives
- Understand Unity's role in robotics simulation and visualization
- Learn to set up Unity for robotics applications
- Implement robot control and visualization in Unity
- Integrate Unity with ROS 2 for bidirectional communication

## Introduction to Unity in Robotics

Unity is a powerful 3D game engine that has found significant applications in robotics for high-fidelity visualization, human-robot interaction studies, and virtual reality applications. Unlike Gazebo, which focuses on physics-based simulation, Unity excels in visual quality, real-time rendering, and user interaction.

### Unity vs Gazebo for Robotics

| Aspect | Unity | Gazebo |
|--------|-------|---------|
| Visual Quality | High-fidelity graphics | Good, but simpler visuals |
| Physics Simulation | Basic (NVIDIA PhysX) | Advanced (ODE, Bullet, etc.) |
| User Interaction | Excellent (VR, AR, UI) | Basic GUI |
| Real-time Performance | Optimized for real-time | Can be computationally heavy |
| Learning Curve | Game development focused | Robotics focused |

## Setting Up Unity for Robotics

### Prerequisites
- Unity Hub and Unity Editor (2021.3 LTS or newer recommended)
- Visual Studio or another C# IDE
- ROS 2 installation (Humble Hawksbill)
- Robot Operating System (ROS) bridge tools

### Unity ROS 2 Bridge
The Unity Robotics Hub provides tools for ROS 2 integration:

1. **Unity Robotics Hub**: Collection of tools, samples, and documentation
2. **ROS TCP Connector**: Enables communication between Unity and ROS 2
3. **URDF Importer**: Imports robot models from URDF files

### Installation Process

1. **Install Unity Hub** from unity.com
2. **Install Unity Editor** (2021.3.22f1 LTS recommended)
3. **Create a new 3D project**
4. **Import Unity Robotics packages** via Package Manager

## Unity ROS 2 Communication

### ROS TCP Connector
The ROS TCP Connector enables communication between Unity and ROS 2:

```csharp
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotName = "my_robot";

    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisteredGazeboTopics.Add("/cmd_vel");
    }

    void Update()
    {
        // Send robot commands to ROS
        var cmdVel = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.Twist();
        cmdVel.linear.x = Input.GetAxis("Vertical");  // Forward/backward
        cmdVel.angular.z = Input.GetAxis("Horizontal");  // Rotation

        ros.Publish("/cmd_vel", cmdVel);
    }
}
```

### Message Types
Unity ROS 2 bridge supports standard ROS message types:

```csharp
using Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class MessageHandler : MonoBehaviour
{
    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.LaserScanMsg>(
            "/scan", LaserScanCallback);
    }

    void LaserScanCallback(Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.LaserScanMsg scan)
    {
        Debug.Log($"Received scan with {scan.ranges.Length} points");
        // Process laser scan data
    }
}
```

## Importing Robot Models with URDF Importer

### Basic URDF Import
The URDF Importer plugin allows direct import of ROS robot models:

1. **Install URDF Importer** from Unity Package Manager
2. **Prepare URDF files** with proper mesh references
3. **Import robot** using the URDF Importer window

### Example URDF for Unity Import
```xml
<?xml version="1.0"?>
<robot name="unity_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://my_robot_description/meshes/base.dae"/>
      </geometry>
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

  <!-- Wheel links -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <mesh filename="package://my_robot_description/meshes/wheel.dae"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

### Import Process
```csharp
using Unity.Robotics.URDFImport;
using UnityEngine;

public class RobotImporter : MonoBehaviour
{
    public string urdfPath = "package://my_robot_description/urdf/robot.urdf";
    public ImportSettings settings;

    void Start()
    {
        // Import robot from URDF
        var robot = URDFRobotExtensions.LoadRobotFromPath(urdfPath, settings);

        if (robot != null)
        {
            robot.transform.SetParent(transform);
            Debug.Log("Robot imported successfully!");
        }
    }
}
```

## Creating Interactive Robot Visualization

### Robot Control Script
```csharp
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;
using UnityEngine;

public class UnityRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float maxLinearSpeed = 1.0f;
    public float maxAngularSpeed = 1.0f;

    [Header("Visualization")]
    public Transform baseLink;
    public Transform[] wheels;

    ROSConnection ros;
    Rigidbody rb;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        rb = GetComponent<Rigidbody>();

        // Subscribe to ROS topics
        ros.Subscribe<Twist>(GetTopicName("/cmd_vel"), CmdVelCallback);

        // Publish joint states
        InvokeRepeating("PublishJointStates", 0.1f, 0.1f);
    }

    void CmdVelCallback(Twist cmd)
    {
        // Convert ROS Twist to Unity movement
        float linearX = Mathf.Clamp((float)cmd.linear.x, -maxLinearSpeed, maxLinearSpeed);
        float angularZ = Mathf.Clamp((float)cmd.angular.z, -maxAngularSpeed, maxAngularSpeed);

        // Apply movement (this is a simplified example)
        Vector3 movement = transform.forward * linearX;
        transform.Translate(movement * Time.deltaTime);
        transform.Rotate(Vector3.up, angularZ * Time.deltaTime);

        // Update wheel rotations for visualization
        UpdateWheelVisuals(linearX, angularZ);
    }

    void UpdateWheelVisuals(float linearSpeed, float angularSpeed)
    {
        foreach (Transform wheel in wheels)
        {
            // Rotate wheels based on movement
            wheel.Rotate(Vector3.right, linearSpeed * 60 * Time.deltaTime);
        }
    }

    void PublishJointStates()
    {
        // Publish joint states back to ROS
        var jointState = new Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor_msgs.JointState();
        jointState.name = new string[] { "left_wheel_joint", "right_wheel_joint" };
        jointState.position = new double[] { 0, 0 }; // Update with actual joint positions
        jointState.header.stamp = new TimeStamp(0, 0);

        ros.Publish(GetTopicName("/joint_states"), jointState);
    }

    string GetTopicName(string topic) => "unity/" + topic;
}
```

## Human-Robot Interaction in Unity

### VR Integration
Unity excels at VR applications for human-robot interaction:

```csharp
using UnityEngine;
using UnityEngine.XR;

public class VRInteraction : MonoBehaviour
{
    [Header("VR Controllers")]
    public GameObject leftController;
    public GameObject rightController;

    [Header("Robot Interaction")]
    public GameObject robot;
    public float interactionDistance = 2.0f;

    void Update()
    {
        // Handle VR controller input
        if (IsControllerActive(leftController))
        {
            HandleLeftControllerInput();
        }

        if (IsControllerActive(rightController))
        {
            HandleRightControllerInput();
        }
    }

    bool IsControllerActive(GameObject controller)
    {
        var inputDevice = GetControllerDevice(controller);
        return inputDevice.isValid;
    }

    void HandleLeftControllerInput()
    {
        // Example: Robot manipulation
        if (Input.GetKeyDown(KeyCode.JoystickButton2)) // A button
        {
            // Send command to robot
            SendRobotCommand("manipulate_object");
        }
    }

    void HandleRightControllerInput()
    {
        // Example: Robot navigation
        if (Input.GetKeyDown(KeyCode.JoystickButton0)) // X button
        {
            // Send navigation command
            SendRobotCommand("move_to_waypoint");
        }
    }

    void SendRobotCommand(string command)
    {
        // Send command via ROS
        var ros = ROSConnection.GetOrCreateInstance();
        var msg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Std_msgs.StringMsg();
        msg.data = command;
        ros.Publish("/robot_command", msg);
    }
}
```

### UI for Robot Control
```csharp
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs;

public class RobotControlUI : MonoBehaviour
{
    [Header("UI Elements")]
    public Slider linearSpeedSlider;
    public Slider angularSpeedSlider;
    public Button moveButton;
    public Text statusText;

    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Setup UI event handlers
        moveButton.onClick.AddListener(MoveRobot);
        linearSpeedSlider.onValueChanged.AddListener(OnLinearSpeedChanged);
        angularSpeedSlider.onValueChanged.AddListener(OnAngularSpeedChanged);
    }

    void MoveRobot()
    {
        var cmd = new Twist();
        cmd.linear.x = linearSpeedSlider.value;
        cmd.angular.z = angularSpeedSlider.value;

        ros.Publish("/cmd_vel", cmd);
        statusText.text = $"Moving: Linear={cmd.linear.x:F2}, Angular={cmd.angular.z:F2}";
    }

    void OnLinearSpeedChanged(float value)
    {
        statusText.text = $"Linear Speed: {value:F2} m/s";
    }

    void OnAngularSpeedChanged(float value)
    {
        statusText.text = $"Angular Speed: {value:F2} rad/s";
    }
}
```

## Performance Optimization for Robotics

### Rendering Optimization
```csharp
using UnityEngine;

public class RobotVisualizationOptimization : MonoBehaviour
{
    [Header("LOD Settings")]
    public float lodDistance = 10.0f;
    public Renderer[] highDetailRenderers;
    public Renderer[] lowDetailRenderers;

    [Header("Culling")]
    public float maxVisualizationDistance = 50.0f;

    Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;
    }

    void Update()
    {
        float distanceToCamera = Vector3.Distance(transform.position, mainCamera.transform.position);

        // LOD switching
        if (distanceToCamera > lodDistance)
        {
            SetLOD(false); // Use low detail
        }
        else
        {
            SetLOD(true);  // Use high detail
        }

        // Culling
        if (distanceToCamera > maxVisualizationDistance)
        {
            foreach (var renderer in highDetailRenderers)
                renderer.enabled = false;
            foreach (var renderer in lowDetailRenderers)
                renderer.enabled = false;
        }
        else
        {
            // Ensure renderers are enabled
            foreach (var renderer in highDetailRenderers)
                renderer.enabled = true;
            foreach (var renderer in lowDetailRenderers)
                renderer.enabled = true;
        }
    }

    void SetLOD(bool highDetail)
    {
        foreach (var renderer in highDetailRenderers)
            renderer.enabled = highDetail;
        foreach (var renderer in lowDetailRenderers)
            renderer.enabled = !highDetail;
    }
}
```

### Network Optimization
```csharp
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class NetworkOptimization : MonoBehaviour
{
    [Header("Update Rates")]
    public float sensorUpdateRate = 10.0f;  // Hz
    public float controlUpdateRate = 50.0f; // Hz

    ROSConnection ros;
    float lastSensorUpdate;
    float lastControlUpdate;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
    }

    void Update()
    {
        float currentTime = Time.time;

        // Throttle sensor updates
        if (currentTime - lastSensorUpdate > 1.0f / sensorUpdateRate)
        {
            SendSensorData();
            lastSensorUpdate = currentTime;
        }

        // Throttle control updates
        if (currentTime - lastControlUpdate > 1.0f / controlUpdateRate)
        {
            SendControlData();
            lastControlUpdate = currentTime;
        }
    }

    void SendSensorData()
    {
        // Send sensor data with reduced frequency
        // Only send critical data at high frequency
    }

    void SendControlData()
    {
        // Send control commands at higher frequency for responsiveness
    }
}
```

## Integration with ROS 2 Ecosystem

### Launching Unity with ROS 2
Create a launch file to coordinate Unity and ROS 2:

```python
# launch/unity_robot_simulation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    unity_executable = LaunchConfiguration('unity_executable')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),

        DeclareLaunchArgument(
            'unity_executable',
            default_value='/path/to/unity_robot_sim',
            description='Path to Unity executable'
        ),

        # Start Unity simulation
        ExecuteProcess(
            cmd=[unity_executable],
            output='screen'
        ),

        # Robot state publisher for Unity visualization
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='unity_robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_description': PathJoinSubstitution([
                    FindPackageShare('my_robot_description'),
                    'urdf',
                    'robot.urdf.xacro'
                ])}
            ]
        ),

        # Unity-ROS bridge node
        Node(
            package='unity_robot_bridge',
            executable='unity_bridge_node',
            name='unity_bridge',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'unity_ip': '127.0.0.1'},
                {'unity_port': 10000}
            ]
        )
    ])
```

## Best Practices for Unity Robotics

### 1. Coordinate System Consistency
```csharp
// Unity uses left-handed coordinate system (X-right, Y-up, Z-forward)
// ROS uses right-handed coordinate system (X-forward, Y-left, Z-up)
// Convert as needed:

Vector3 RosToUnity(Vector3 rosVector)
{
    return new Vector3(rosVector.z, rosVector.x, rosVector.y);
}

Vector3 UnityToRos(Vector3 unityVector)
{
    return new Vector3(unityVector.y, unityVector.z, unityVector.x);
}
```

### 2. Frame Transformations
```csharp
// Maintain proper TF tree relationships
public class UnityTFManager : MonoBehaviour
{
    public Transform rosWorld;
    public Transform robotBase;
    public Transform[] sensorFrames;

    void Update()
    {
        // Publish TF transforms to ROS
        PublishTransforms();
    }

    void PublishTransforms()
    {
        // Send transforms for each frame
        SendTransform(rosWorld, robotBase, "map", "base_link");
        foreach (var sensor in sensorFrames)
        {
            SendTransform(robotBase, sensor, "base_link", sensor.name);
        }
    }

    void SendTransform(Transform parent, Transform child, string parentFrame, string childFrame)
    {
        var ros = ROSConnection.GetOrCreateInstance();
        var tfMsg = new Unity.Robotics.ROSTCPConnector.MessageTypes.Geometry_msgs.TransformStamped();

        // Fill in transform data
        tfMsg.header.frame_id = parentFrame;
        tfMsg.child_frame_id = childFrame;
        tfMsg.header.stamp = new TimeStamp(0, 0);

        ros.Publish("/tf", tfMsg);
    }
}
```

### 3. Resource Management
- Use object pooling for frequently created/destroyed objects
- Implement proper cleanup of ROS connections
- Optimize mesh and texture loading for robot models

## Troubleshooting Common Issues

### 1. Connection Issues
- Verify Unity and ROS are on the same network
- Check firewall settings for the TCP port
- Ensure ROS TCP Connector settings match

### 2. Coordinate System Mismatches
- Verify coordinate system conversions
- Check TF frame relationships
- Validate sensor data orientations

### 3. Performance Issues
- Reduce visual quality settings during testing
- Limit update rates for non-critical data
- Use LOD systems for distant objects

## Summary

Unity provides high-fidelity visualization and human-robot interaction capabilities:

- **ROS 2 Integration**: TCP-based communication with standard message types
- **URDF Import**: Direct import of ROS robot models
- **Human Interaction**: Excellent support for VR, AR, and UI
- **Visualization**: High-quality rendering and graphics
- **Performance**: Real-time optimization techniques

## Exercises

1. Import a URDF robot model into Unity and visualize it
2. Create a VR interface for teleoperating a robot in Unity
3. Implement a sensor visualization system in Unity
4. Develop a Unity-based robot programming interface