---
sidebar_position: 5
---

# Sensor Simulation in Gazebo

## Learning Objectives
- Understand how to simulate various sensor types in Gazebo
- Learn to configure sensor properties for realistic data
- Implement sensor integration with ROS 2
- Troubleshoot common sensor simulation issues

## Introduction to Sensor Simulation

Sensor simulation in Gazebo allows you to test perception algorithms, sensor fusion techniques, and robot behaviors without physical hardware. Gazebo provides realistic simulation of various sensor types including cameras, LiDAR, IMUs, GPS, force/torque sensors, and more.

## Camera Simulation

### Basic Camera Sensor
Camera sensors simulate RGB cameras and can publish image data to ROS 2 topics:

```xml
<sensor name="camera" type="camera">
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <visualize>true</visualize>
</sensor>
```

### Depth Camera
For depth information, use a depth camera sensor:

```xml
<sensor name="depth_camera" type="depth">
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <camera name="depth_head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <depth_camera>
      <output>depths</output>
    </depth_camera>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <visualize>true</visualize>
</sensor>
```

### Camera with ROS 2 Integration
To integrate with ROS 2, use the appropriate plugin:

```xml
<sensor name="camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>/camera</namespace>
      <remapping>~/image_raw:=image</remapping>
      <remapping>~/camera_info:=camera_info</remapping>
    </ros>
    <camera_name>camera</camera_name>
    <frame_name>camera_link</frame_name>
    <hack_baseline>0.07</hack_baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
  </plugin>
</sensor>
```

## LiDAR and Range Sensors

### 2D LiDAR (Ray Sensor)
For 2D LiDAR simulation:

```xml
<sensor name="laser_scan" type="ray">
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>    <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>
    </noise>
  </ray>
  <plugin name="laser_scan" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>/laser</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
    <frame_name>laser_link</frame_name>
  </plugin>
</sensor>
```

### 3D LiDAR (Multi-ray)
For 3D LiDAR like Velodyne:

```xml
<sensor name="velodyne" type="ray">
  <pose>0 0 0.3 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>800</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>32</samples>
        <resolution>1</resolution>
        <min_angle>-0.261799</min_angle>  <!-- -15 degrees -->
        <max_angle>0.261799</max_angle>   <!-- 15 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="velodyne_driver" filename="libgazebo_ros_velodyne_gpu_laser.so">
    <ros>
      <namespace>/velodyne</namespace>
      <remapping>~/out:=points</remapping>
    </ros>
    <topic_name>points</topic_name>
    <frame_name>velodyne</frame_name>
    <min_range>0.9</min_range>
    <max_range>130.0</max_range>
    <gaussian_noise>0.008</gaussian_noise>
  </plugin>
</sensor>
```

## IMU Simulation

IMU sensors provide acceleration, angular velocity, and orientation data:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.0003</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.0003</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.0003</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/imu</namespace>
      <remapping>~/out:=data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
    <body_name>imu_body</body_name>
    <update_rate>100</update_rate>
  </plugin>
</sensor>
```

## GPS Simulation

GPS sensors provide location data:

```xml
<sensor name="gps" type="gps">
  <always_on>1</always_on>
  <update_rate>1</update_rate>
  <plugin name="gps_plugin" filename="libgazebo_ros_gps.so">
    <ros>
      <namespace>/gps</namespace>
      <remapping>~/out:=fix</remapping>
    </ros>
    <frame_name>gps_link</frame_name>
    <topic_name>fix</topic_name>
    <update_rate>1.0</update_rate>
    <gaussian_noise>0.05</gaussian_noise>
  </plugin>
</sensor>
```

## Force/Torque Sensors

For measuring forces and torques at joints:

```xml
<sensor name="ft_sensor" type="force_torque">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin name="ft_sensor_plugin" filename="libgazebo_ros_ft_sensor.so">
    <ros>
      <namespace>/ft_sensor</namespace>
      <remapping>~/wrench:=wrench</remapping>
    </ros>
    <frame_name>ft_sensor_link</frame_name>
    <topic_name>wrench</topic_name>
    <update_rate>100</update_rate>
  </plugin>
</sensor>
```

## Custom Sensor Plugins

For specialized sensors, you can create custom plugins:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <sensor_msgs/Range.h>

namespace gazebo
{
  class CustomRangeSensor : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Get the range sensor
      this->parentSensor =
        std::dynamic_pointer_cast<sensors::RaySensor>(_sensor);

      if (!this->parentSensor)
      {
        gzerr << "CustomRangeSensor requires a RaySensor.\n";
        return;
      }

      // Initialize ROS
      if (!ros::isInitialized())
      {
        gzerr << "Not loading plugin since ROS hasn't been initialized.\n";
        return;
      }

      this->rosNode.reset(new ros::NodeHandle("gazebo_client"));
      this->pub = this->rosNode->advertise<sensor_msgs::Range>("range", 1);

      // Connect to sensor update event
      this->updateConnection = this->parentSensor->Topic()->Connect(
        std::bind(&CustomRangeSensor::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      sensor_msgs::Range msg;
      msg.header.stamp = ros::Time::now();
      msg.header.frame_id = "range_sensor_link";

      // Get range data from the sensor
      double range = this->parentSensor->Range(0);

      msg.radiation_type = sensor_msgs::Range::INFRARED;
      msg.field_of_view = 0.1;  // Approximate
      msg.min_range = 0.01;
      msg.max_range = 5.0;
      msg.range = range;

      this->pub.publish(msg);
    }

    private: sensors::RaySensorPtr parentSensor;
    private: ros::Publisher pub;
    private: std::unique_ptr<ros::NodeHandle> rosNode;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomRangeSensor)
}
```

## Sensor Integration with ROS 2

### Using ROS 2 Sensor Messages
When sensors are properly configured with ROS 2 plugins, they publish standard ROS 2 messages:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, CameraInfo
from cv_bridge import CvBridge
import numpy as np

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Initialize CV bridge for image processing
        self.bridge = CvBridge()

        # Subscribe to different sensor topics
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/laser/scan',
            self.scan_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

    def scan_callback(self, msg):
        # Process LiDAR data
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f'Min obstacle distance: {min_distance:.2f}m')

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Process image (e.g., object detection)
            height, width, _ = cv_image.shape
            self.get_logger().info(f'Received image: {width}x{height}')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def imu_callback(self, msg):
        # Process IMU data
        linear_accel = msg.linear_acceleration
        angular_vel = msg.angular_velocity
        orientation = msg.orientation

        self.get_logger().info(
            f'Linear Acc: ({linear_accel.x:.2f}, {linear_accel.y:.2f}, {linear_accel.z:.2f})'
        )

def main(args=None):
    rclpy.init(args=args)
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Configuration Best Practices

### 1. Noise Modeling
Real sensors have noise, so include realistic noise models:

```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.01</stddev>  <!-- Adjust based on real sensor specs -->
  <bias_mean>0.0</bias_mean>
  <bias_stddev>0.001</bias_stddev>
</noise>
```

### 2. Frame Conventions
Use proper frame naming and conventions:

```xml
<plugin name="camera_plugin" filename="libgazebo_ros_camera.so">
  <frame_name>camera_optical_frame</frame_name>  <!-- Include _optical suffix for cameras -->
</plugin>
```

### 3. Update Rates
Set appropriate update rates based on sensor type and application:

- IMU: 100-1000 Hz
- LiDAR: 5-20 Hz
- Cameras: 15-30 Hz
- GPS: 1-10 Hz

### 4. Computational Performance
Balance sensor quality with simulation performance:

```xml
<!-- For performance: lower resolution -->
<image>
  <width>320</width>
  <height>240</height>
</image>

<!-- For accuracy: higher resolution -->
<!-- <image> -->
<!--   <width>1280</width> -->
<!--   <height>960</height> -->
<!-- </image> -->
```

## Troubleshooting Common Issues

### 1. Sensor Not Publishing Data
- Check that the sensor is properly attached to a link
- Verify plugin is correctly configured
- Check ROS namespace and topic names
- Ensure Gazebo simulation is running

### 2. Incorrect Sensor Data
- Verify sensor pose and orientation
- Check noise parameters
- Validate sensor range and field of view settings

### 3. Performance Issues
- Reduce sensor resolution or update rate
- Simplify sensor noise models
- Use simpler collision geometries for sensor links

### 4. TF Issues
- Ensure proper frame names match URDF
- Check that robot_state_publisher is running
- Verify sensor links are part of the robot model

## Simulation-Specific Considerations

### Realism vs Performance
- Add realistic noise models for perception algorithm testing
- Use appropriate physics settings for sensor accuracy
- Balance simulation quality with real-time performance

### Sim-to-Real Transfer
- Use sensor parameters that match real hardware when possible
- Include realistic noise and distortion models
- Validate sensor simulation against real sensor data

## Summary

Sensor simulation in Gazebo provides realistic data for testing robotics algorithms:

- **Camera sensors**: RGB, depth, and stereo vision simulation
- **LiDAR sensors**: 2D and 3D laser scanning simulation
- **IMU sensors**: Acceleration, angular velocity, and orientation
- **GPS sensors**: Location and navigation data
- **Custom sensors**: Specialized sensor types through plugins
- **ROS 2 integration**: Standard message types and topics

## Exercises

1. Configure a multi-sensor robot with camera, LiDAR, and IMU
2. Implement a sensor fusion algorithm using simulated data
3. Create a custom sensor plugin for a specialized application
4. Compare simulated sensor data with real sensor specifications