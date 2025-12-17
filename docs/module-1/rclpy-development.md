---
sidebar_position: 4
---

# Python-based ROS 2 Development using rclpy

## Learning Objectives
- Master the rclpy client library for Python-based ROS 2 development
- Learn to create nodes, publishers, subscribers, services, and actions in Python
- Understand ROS 2 parameter management and node configuration
- Implement best practices for Python-based robot software development

## Introduction to rclpy

rclpy is the Python client library for ROS 2. It provides a Python API to interact with the ROS 2 middleware, allowing you to create nodes, publish and subscribe to topics, provide and call services, and manage actions.

### Installation and Setup

rclpy is typically installed as part of the ROS 2 Python development packages:

```bash
pip install rclpy
# or through ROS 2 installation
sudo apt install python3-rclpy
```

### Basic Node Structure

Every ROS 2 Python node follows a similar structure:

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)  # Initialize ROS communications

    node = MyNode()  # Create node instance

    try:
        rclpy.spin(node)  # Keep node alive
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # Clean up
        rclpy.shutdown()  # Shutdown ROS communications

if __name__ == '__main__':
    main()
```

## Creating Publishers and Subscribers

### Publisher Example with Custom Message

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import math

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publisher
        self.publisher_ = self.create_publisher(
            LaserScan,
            'scan',
            10  # QoS queue size
        )

        # Create timer to publish data periodically
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Sample data
        self.i = 0

    def timer_callback(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'

        # Fill in laser scan data
        msg.angle_min = -math.pi / 2
        msg.angle_max = math.pi / 2
        msg.angle_increment = math.pi / 180  # 1 degree
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.1
        msg.range_max = 10.0

        # Generate sample ranges
        num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1
        msg.ranges = [1.0 + (self.i % 10) * 0.1] * num_readings

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing laser scan: {len(msg.ranges)} readings')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber with Message Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        # Create subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Create publisher for movement commands
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Parameters
        self.declare_parameter('safe_distance', 0.5)
        self.safe_distance = self.get_parameter('safe_distance').value

    def scan_callback(self, msg):
        # Process laser scan to detect obstacles
        min_range = float('inf')

        for range_val in msg.ranges:
            if not (math.isnan(range_val) or math.isinf(range_val)):
                if range_val < min_range:
                    min_range = range_val

        # Create movement command based on obstacle detection
        move_cmd = Twist()

        if min_range < self.safe_distance:
            # Obstacle detected, stop and turn
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.5  # Turn right
            self.get_logger().warn(f'Obstacle detected at {min_range:.2f}m, turning!')
        else:
            # Clear path, move forward
            move_cmd.linear.x = 0.2
            move_cmd.angular.z = 0.0
            self.get_logger().info(f'Path clear, distance: {min_range:.2f}m')

        self.publisher.publish(move_cmd)

def main(args=None):
    rclpy.init(args=args)

    obstacle_detector = ObstacleDetector()

    try:
        rclpy.spin(obstacle_detector)
    except KeyboardInterrupt:
        pass
    finally:
        obstacle_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services in rclpy

### Service Server Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class CalculatorService(Node):
    def __init__(self):
        super().__init__('calculator_service')

        # Create service
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )

        # Additional services for other operations
        self.mul_srv = self.create_service(
            AddTwoInts,  # Using same message type for simplicity
            'multiply_two_ints',
            self.multiply_callback
        )

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response

    def multiply_callback(self, request, response):
        response.sum = request.a * request.b
        self.get_logger().info(f'{request.a} * {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)

    calc_service = CalculatorService()

    try:
        rclpy.spin(calc_service)
    except KeyboardInterrupt:
        pass
    finally:
        calc_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class CalculatorClient(Node):
    def __init__(self):
        super().__init__('calculator_client')

        # Create clients
        self.add_client = self.create_client(AddTwoInts, 'add_two_ints')
        self.mul_client = self.create_client(AddTwoInts, 'multiply_two_ints')

        # Wait for services to be available
        while not self.add_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Add service not available, waiting again...')

        while not self.mul_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Multiply service not available, waiting again...')

    def send_add_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.add_client.call_async(request)
        return future

    def send_multiply_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        future = self.mul_client.call_async(request)
        return future

def main(args=None):
    rclpy.init(args=args)

    calc_client = CalculatorClient()

    # Example: Send a request and wait for response
    future = calc_client.send_add_request(2, 3)

    try:
        rclpy.spin_until_future_complete(calc_client, future)
        response = future.result()
        if response:
            calc_client.get_logger().info(f'Result: {response.sum}')
    except Exception as e:
        calc_client.get_logger().error(f'Service call failed: {e}')
    finally:
        calc_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameters in rclpy

ROS 2 provides a parameter system for configuring nodes at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # Set parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info(
            f'Initialized with: name={self.robot_name}, '
            f'velocity={self.max_velocity}, distance={self.safety_distance}'
        )

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                if param.value > 2.0:
                    return SetParametersResult(successful=False, reason='Max velocity too high')

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)

    param_node = ParameterNode()

    try:
        rclpy.spin(param_node)
    except KeyboardInterrupt:
        pass
    finally:
        param_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Launch Files

Launch files in ROS 2 allow you to start multiple nodes with specific configurations:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Launch the sensor publisher node
        Node(
            package='my_robot_package',
            executable='sensor_publisher',
            name='lidar_publisher',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'sensor_range': 10.0}
            ],
            remappings=[
                ('/scan', '/robot1/scan')
            ]
        ),

        # Launch the obstacle detector node
        Node(
            package='my_robot_package',
            executable='obstacle_detector',
            name='obstacle_detector',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'safe_distance': 0.5}
            ]
        )
    ])
```

## Best Practices for rclpy Development

### 1. Error Handling
```python
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    node.get_logger().info('Interrupted by user')
except Exception as e:
    node.get_logger().error(f'Unexpected error: {e}')
finally:
    node.destroy_node()
    rclpy.shutdown()
```

### 2. Resource Management
```python
class ResourceManagedNode(Node):
    def __init__(self):
        super().__init__('resource_node')
        self.timers = []
        self.publishers = []
        self.subscribers = []

    def destroy_node(self):
        # Clean up resources
        for timer in self.timers:
            timer.destroy()
        # Clean up other resources
        super().destroy_node()
```

### 3. Logging
```python
# Use appropriate log levels
self.get_logger().debug('Detailed debug information')
self.get_logger().info('General information')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
self.get_logger().fatal('Fatal error message')
```

### 4. Threading Considerations
```python
import threading

class ThreadedNode(Node):
    def __init__(self):
        super().__init__('threaded_node')

        # Use threading for CPU-intensive tasks
        self.data_lock = threading.Lock()
        self.shared_data = []

    def process_data_in_thread(self, data):
        def worker():
            with self.data_lock:
                # Process data safely
                self.shared_data.append(data)

        thread = threading.Thread(target=worker)
        thread.start()
```

## Summary

rclpy provides a comprehensive Python API for ROS 2 development. Key concepts include:

- **Node structure**: Proper initialization, spinning, and cleanup
- **Communication patterns**: Publishers, subscribers, services, and actions
- **Parameters**: Runtime configuration of nodes
- **Launch files**: Starting multiple nodes with configurations
- **Best practices**: Error handling, resource management, and threading

Mastering rclpy is essential for developing Python-based robot applications in ROS 2.

## Exercises

1. Create a node that publishes temperature sensor data with proper error handling
2. Implement a service that calculates robot navigation paths
3. Create a launch file that starts multiple sensor nodes with different parameters
4. Develop a parameter callback that validates parameter values before accepting them