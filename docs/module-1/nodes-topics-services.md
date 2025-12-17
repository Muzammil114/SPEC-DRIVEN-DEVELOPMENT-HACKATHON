---
sidebar_position: 3
---

# Nodes, Topics, Services, and Actions

## Learning Objectives
- Understand the fundamental communication patterns in ROS 2
- Learn to implement nodes with different communication methods
- Practice creating publishers, subscribers, services, and actions
- Understand when to use each communication pattern

## Nodes in ROS 2

A node is the fundamental unit of computation in ROS 2. It represents a single process that performs computation. Multiple nodes are combined together to form a complete robot application.

### Creating a Node

To create a node in Python, inherit from `rclpy.node.Node`:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Initialize node-specific components here
```

### Node Lifecycle

ROS 2 nodes can have different lifecycle states:
- **Unconfigured**: Node is created but not yet configured
- **Inactive**: Node is configured but not active
- **Active**: Node is running and processing
- **Finalized**: Node is shutting down

## Topics and Message Passing

Topics implement a publish-subscribe communication pattern where publishers send messages to topics without knowing which subscribers (if any) there are.

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
```

## Services

Services provide a request-response communication pattern. A service client sends a request and waits for a response from the service server.

### Service Definition

Services are defined in `.srv` files with the format:

```
# Request
string name
int32 age
---
# Response
bool success
string message
```

### Service Server Example

```python
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response
```

### Service Client Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Actions

Actions are designed for long-running tasks that provide feedback during execution. They consist of a goal, feedback, and result.

### Action Definition

Actions are defined in `.action` files:

```
# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] sequence
```

### Action Server Example

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

## When to Use Each Communication Pattern

### Use Topics when:
- Multiple subscribers need the same data
- Real-time streaming of data is required
- The publisher doesn't need to know about subscribers
- Data is being continuously generated

### Use Services when:
- Request-response pattern is needed
- The call is relatively short-lived
- Error handling and return codes are important
- Synchronous communication is preferred

### Use Actions when:
- Long-running tasks are involved
- Feedback during execution is needed
- Tasks might be preempted or canceled
- Asynchronous communication with status updates is required

## Best Practices

1. **Naming Conventions**:
   - Use descriptive names for topics, services, and actions
   - Follow ROS naming conventions (lowercase with underscores)

2. **Error Handling**:
   - Always handle potential connection failures
   - Implement timeouts for service calls
   - Check for service availability before calling

3. **Resource Management**:
   - Clean up publishers, subscribers, and other resources
   - Use appropriate queue sizes for messages

4. **Thread Safety**:
   - Be aware of threading in ROS 2
   - Use appropriate synchronization for shared data

## Summary

ROS 2 provides multiple communication patterns to suit different use cases:
- Topics for publish-subscribe communication
- Services for request-response communication
- Actions for long-running tasks with feedback

Understanding when to use each pattern is crucial for effective robot software architecture.

## Exercises

1. Create a publisher node that publishes sensor data
2. Create a subscriber node that processes the sensor data
3. Implement a service to configure sensor parameters
4. Design an action for performing a complex manipulation task