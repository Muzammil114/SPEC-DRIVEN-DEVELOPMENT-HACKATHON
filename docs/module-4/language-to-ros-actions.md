---
sidebar_position: 6
---

# Translating Language Goals into ROS 2 Action Graphs

## Learning Objectives
- Understand the structure and execution of ROS 2 action graphs
- Learn to map natural language commands to ROS 2 action sequences
- Implement action graph construction from parsed language commands
- Create action execution monitoring and feedback systems
- Develop error handling and recovery mechanisms for action graphs

## Introduction to ROS 2 Action Graphs

ROS 2 action graphs represent sequences of coordinated robot behaviors that can be executed in response to high-level goals. These graphs bridge the gap between natural language commands and low-level robot actions, enabling complex multi-step tasks to be performed reliably.

### ROS 2 Actions vs Services vs Topics

| Communication Type | Purpose | Characteristics | Use Case |
|-------------------|---------|-----------------|----------|
| Topics | Continuous data streaming | Unidirectional, many-to-many | Sensor data, state publishing |
| Services | Request-response communication | Synchronous, one-to-one | Configuration, single operations |
| Actions | Goal-oriented tasks | Asynchronous, with feedback | Navigation, manipulation, complex tasks |

### Action Graph Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Language      │    │   Action        │    │   ROS 2         │
│   Command       │───▶│   Graph         │───▶│   Execution     │
│   "Clean the    │    │   Construction  │    │   Framework     │
│   kitchen"      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Graph         │
                    │   Validation    │
                    │   & Optimization│
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Execution     │
                    │   Monitoring    │
                    └─────────────────┘
```

## ROS 2 Actions Fundamentals

### Action Message Structure

ROS 2 actions have three message types:

1. **Goal**: Defines the desired outcome
2. **Result**: Contains the final outcome
3. **Feedback**: Provides intermediate progress information

```python
# Example action definition (action/NavigateToPose.action)
# Goal definition
geometry_msgs/PoseStamped pose
string behavior_tree

# Result definition
bool reached_goal
builtin_interfaces/Time processing_time
geometry_msgs/PoseStamped final_pose

# Feedback definition
geometry_msgs/PoseStamped current_pose
float32 distance_remaining
builtin_interfaces/Duration navigation_time
```

### Creating Action Clients and Servers

```python
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import time

class ActionExampleNode(Node):
    def __init__(self):
        super().__init__('action_example_node')

        # Action client for navigation
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Publishers for status updates
        self.status_pub = self.create_publisher(String, 'action_status', 10)

        # Wait for action server to be available
        self.nav_client.wait_for_server()

    def send_navigation_goal(self, x, y, theta):
        """Send a navigation goal to the action server"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        from math import sin, cos
        goal_msg.pose.pose.orientation.z = sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = cos(theta / 2.0)

        goal_msg.pose.header.frame_id = 'map'

        # Send goal with feedback callback
        self._send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle action result"""
        result = future.result().result
        self.get_logger().info(f'Result: {result.reached_goal}')

    def feedback_callback(self, feedback_msg):
        """Handle action feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Distance remaining: {feedback.distance_remaining:.2f}'
        )
```

## Language-to-Action Mapping

### Semantic Action Mapping System

```python
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class ActionMapping:
    """Defines mapping from language concepts to ROS 2 actions"""
    language_pattern: str
    ros_action: str
    parameters: Dict[str, str]  # Maps language entities to action parameters
    required_entities: List[str]
    action_package: str
    action_interface: str

class LanguageActionMapper:
    def __init__(self):
        self.action_mappings = self._initialize_action_mappings()
        self.parameter_extractors = self._initialize_parameter_extractors()

    def _initialize_action_mappings(self) -> Dict[str, ActionMapping]:
        """Initialize mappings from language patterns to ROS 2 actions"""
        return {
            'navigation': ActionMapping(
                language_pattern=r'(go to|navigate to|move to|travel to) (?P<location>\w+)',
                ros_action='NavigateToPose',
                parameters={
                    'location': 'pose.header.frame_id'
                },
                required_entities=['location'],
                action_package='nav2_msgs',
                action_interface='NavigateToPose'
            ),
            'manipulation_pick': ActionMapping(
                language_pattern=r'(pick up|get|grab|take) (the )?(?P<color>\w+ )?(?P<object>\w+)',
                ros_action='ManipulationAction',
                parameters={
                    'object': 'object_type',
                    'color': 'object_color'
                },
                required_entities=['object'],
                action_package='custom_msgs',
                action_interface='ManipulationAction'
            ),
            'manipulation_place': ActionMapping(
                language_pattern=r'(place|put|set) (the )?(?P<object>\w+) (on|in|at) (the )?(?P<location>\w+)',
                ros_action='ManipulationAction',
                parameters={
                    'object': 'object_type',
                    'location': 'target_location'
                },
                required_entities=['object', 'location'],
                action_package='custom_msgs',
                action_interface='ManipulationAction'
            ),
            'detection': ActionMapping(
                language_pattern=r'(find|locate|search for) (the )?(?P<object>\w+)',
                ros_action='FindObject',
                parameters={
                    'object': 'object_type'
                },
                required_entities=['object'],
                action_package='custom_msgs',
                action_interface='FindObject'
            )
        }

    def _initialize_parameter_extractors(self) -> Dict[str, callable]:
        """Initialize functions to extract parameters from language entities"""
        return {
            'location': self._extract_location,
            'object': self._extract_object,
            'color': self._extract_color,
            'quantity': self._extract_quantity
        }

    def map_language_to_action(self, parsed_command: Dict) -> Optional[Dict]:
        """Map a parsed language command to a ROS 2 action"""
        action_type = parsed_command.get('action')
        entities = parsed_command.get('entities', {})

        # Find appropriate mapping
        for mapping_name, mapping in self.action_mappings.items():
            if self._matches_mapping(mapping, action_type, entities):
                action_config = {
                    'action_name': mapping.ros_action,
                    'action_package': mapping.action_package,
                    'action_interface': mapping.action_interface,
                    'parameters': self._extract_parameters(mapping, entities),
                    'required_entities': mapping.required_entities
                }
                return action_config

        return None

    def _matches_mapping(self, mapping: ActionMapping, action_type: str, entities: Dict) -> bool:
        """Check if command matches a particular mapping"""
        # Check if required entities are present
        entity_types = {entity['type'] for entity in entities}
        required_present = all(req in entity_types for req in mapping.required_entities)

        # Check if action type matches
        action_match = action_type.lower() in mapping.language_pattern.lower()

        return required_present and action_match

    def _extract_parameters(self, mapping: ActionMapping, entities: Dict) -> Dict:
        """Extract action parameters from language entities"""
        parameters = {}

        for entity in entities:
            entity_type = entity['type']
            entity_value = entity['value']

            if entity_type in mapping.parameters:
                param_name = mapping.parameters[entity_type]
                parameters[param_name] = entity_value

        return parameters

    def _extract_location(self, entity_value: str) -> str:
        """Extract and normalize location parameter"""
        # Map natural location names to ROS frame names
        location_map = {
            'kitchen': 'kitchen_frame',
            'bedroom': 'bedroom_frame',
            'office': 'office_frame',
            'living room': 'living_room_frame',
            'table': 'table_frame',
            'couch': 'couch_frame'
        }
        return location_map.get(entity_value.lower(), entity_value)

    def _extract_object(self, entity_value: str) -> str:
        """Extract and normalize object parameter"""
        # Standardize object names
        object_map = {
            'cup': 'cup',
            'glass': 'glass',
            'bottle': 'bottle',
            'book': 'book',
            'phone': 'phone',
            'laptop': 'laptop'
        }
        return object_map.get(entity_value.lower(), entity_value)

    def _extract_color(self, entity_value: str) -> str:
        """Extract color parameter"""
        color_map = {
            'red': 'red',
            'blue': 'blue',
            'green': 'green',
            'yellow': 'yellow',
            'black': 'black',
            'white': 'white'
        }
        return color_map.get(entity_value.lower(), entity_value)

    def _extract_quantity(self, entity_value: str) -> int:
        """Extract quantity parameter"""
        try:
            return int(entity_value)
        except ValueError:
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            return word_to_num.get(entity_value.lower(), 1)
```

## Action Graph Construction

### Graph-Based Action Planning

```python
from typing import Set, Any
import networkx as nx
from dataclasses import dataclass

@dataclass
class ActionNode:
    """Represents a single action in an action graph"""
    id: str
    action_type: str
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of actions that must complete first
    timeout: float = 30.0
    retry_count: int = 3
    success_threshold: float = 0.9

class ActionGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for dependencies
        self.nodes: Dict[str, ActionNode] = {}

    def add_action(self, node: ActionNode):
        """Add an action node to the graph"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, node_data=node)

        # Add dependency edges
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self.graph.add_edge(dep_id, node.id)

    def get_execution_order(self) -> List[str]:
        """Get the order in which actions should be executed"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Action graph has circular dependencies")

    def get_ready_actions(self, completed_actions: Set[str]) -> List[str]:
        """Get actions that are ready to execute based on completed actions"""
        ready = []
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]['node_data']
            # Check if all dependencies are completed
            dependencies_met = all(dep in completed_actions for dep in node.dependencies)
            not_completed = node_id not in completed_actions
            not_running = node_id not in self.get_running_actions()

            if dependencies_met and not_completed and not_running:
                ready.append(node_id)

        return ready

    def get_running_actions(self) -> Set[str]:
        """Get actions that are currently running"""
        # This would track actual execution status
        # For now, return empty set
        return set()

    def validate_graph(self) -> Dict[str, List[str]]:
        """Validate the action graph for common issues"""
        issues = {
            'circular_dependencies': [],
            'missing_dependencies': [],
            'isolated_nodes': []
        }

        # Check for circular dependencies
        try:
            list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            issues['circular_dependencies'].append("Graph has circular dependencies")

        # Check for missing dependencies
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]['node_data']
            for dep_id in node.dependencies:
                if dep_id not in self.graph.nodes:
                    issues['missing_dependencies'].append(
                        f"Action '{node_id}' depends on non-existent action '{dep_id}'"
                    )

        # Check for isolated nodes (nodes with no connections)
        for node_id in self.graph.nodes:
            if self.graph.degree(node_id) == 0:
                issues['isolated_nodes'].append(f"Action '{node_id}' is isolated")

        return issues

    def visualize_graph(self, filename: str = "action_graph.png"):
        """Visualize the action graph"""
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph)
            plt.figure(figsize=(12, 8))
            nx.draw(self.graph, pos, with_labels=True, node_color='lightblue',
                   node_size=3000, font_size=10, font_weight='bold',
                   arrows=True, arrowsize=20)
            plt.title("Action Graph")
            plt.savefig(filename)
            plt.close()
        except ImportError:
            print("Matplotlib not available for graph visualization")

class LanguageToActionGraph:
    def __init__(self, language_action_mapper: LanguageActionMapper):
        self.mapper = language_action_mapper

    def create_action_graph(self, commands: List[Dict]) -> ActionGraph:
        """Create an action graph from a list of parsed commands"""
        graph = ActionGraph()

        for i, command in enumerate(commands):
            # Map language command to ROS action
            action_config = self.mapper.map_language_to_action(command)

            if action_config:
                # Create action node
                node = ActionNode(
                    id=f"action_{i}",
                    action_type=action_config['action_name'],
                    parameters=action_config['parameters'],
                    dependencies=self._determine_dependencies(i, commands),
                    timeout=action_config.get('timeout', 30.0),
                    retry_count=action_config.get('retry_count', 3)
                )

                graph.add_action(node)

        return graph

    def _determine_dependencies(self, current_idx: int, commands: List[Dict]) -> List[str]:
        """Determine dependencies for an action based on command sequence"""
        dependencies = []

        # Simple dependency: previous actions in sequence
        if current_idx > 0:
            dependencies.append(f"action_{current_idx - 1}")

        # More sophisticated dependency analysis would go here
        # For example: "pick up object" should depend on "navigate to object location"

        return dependencies
```

## Action Execution Framework

### Action Execution Manager

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class ActionResult:
    """Result of an action execution"""
    success: bool
    result_data: Dict[str, Any]
    error_message: str = ""
    execution_time: float = 0.0
    retries: int = 0

class ActionExecutionManager:
    def __init__(self, node_interface):
        self.node_interface = node_interface
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.active_executions = {}
        self.execution_callbacks = {}

    def execute_action_graph(self, graph: ActionGraph,
                           completion_callback: Callable = None) -> str:
        """Execute an action graph with dependency management"""
        execution_id = f"exec_{int(time.time())}_{len(self.active_executions)}"

        execution_context = {
            'graph': graph,
            'completed_actions': set(),
            'failed_actions': set(),
            'execution_order': graph.get_execution_order(),
            'start_time': time.time(),
            'status': 'running'
        }

        self.active_executions[execution_id] = execution_context

        # Start execution in a separate thread
        execution_thread = threading.Thread(
            target=self._execute_graph_thread,
            args=(execution_id, completion_callback)
        )
        execution_thread.start()

        return execution_id

    def _execute_graph_thread(self, execution_id: str,
                            completion_callback: Callable = None):
        """Execute the action graph in a thread"""
        context = self.active_executions[execution_id]

        try:
            while context['status'] == 'running':
                # Get ready actions
                ready_actions = context['graph'].get_ready_actions(
                    context['completed_actions']
                )

                if not ready_actions:
                    # Check if execution is complete
                    all_actions = set(context['graph'].nodes.keys())
                    completed = context['completed_actions']
                    failed = context['failed_actions']

                    if completed.union(failed) == all_actions:
                        break

                    # Wait a bit before checking again
                    time.sleep(0.1)
                    continue

                # Execute ready actions (could be multiple in parallel)
                for action_id in ready_actions:
                    if action_id not in context['completed_actions'] and \
                       action_id not in context['failed_actions']:
                        self._execute_single_action(execution_id, action_id)

        except Exception as e:
            self.node_interface.get_logger().error(f"Execution error: {e}")
            context['status'] = 'failed'

        finally:
            context['status'] = 'completed'
            if completion_callback:
                completion_callback(execution_id, context)

    def _execute_single_action(self, execution_id: str, action_id: str):
        """Execute a single action"""
        context = self.active_executions[execution_id]
        action_node = context['graph'].nodes[action_id]

        self.node_interface.get_logger().info(f"Executing action: {action_id}")

        # Execute the action using the node interface
        result = self._execute_ros_action(action_node)

        if result.success:
            context['completed_actions'].add(action_id)
            self.node_interface.get_logger().info(f"Action {action_id} completed successfully")
        else:
            context['failed_actions'].add(action_id)
            self.node_interface.get_logger().error(f"Action {action_id} failed: {result.error_message}")

    def _execute_ros_action(self, action_node: ActionNode) -> ActionResult:
        """Execute a ROS 2 action"""
        start_time = time.time()

        try:
            # Based on action type, call appropriate ROS interface
            if action_node.action_type == 'NavigateToPose':
                result = self._execute_navigation_action(action_node)
            elif action_node.action_type == 'ManipulationAction':
                result = self._execute_manipulation_action(action_node)
            else:
                result = self._execute_generic_action(action_node)

            execution_time = time.time() - start_time
            return ActionResult(
                success=result.get('success', False),
                result_data=result.get('result_data', {}),
                error_message=result.get('error_message', ''),
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ActionResult(
                success=False,
                result_data={},
                error_message=str(e),
                execution_time=execution_time
            )

    def _execute_navigation_action(self, action_node: ActionNode) -> Dict:
        """Execute navigation action"""
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()

        # Set pose from parameters
        if 'pose' in action_node.parameters:
            goal_msg.pose = action_node.parameters['pose']
        else:
            # Create pose from location parameter
            location = action_node.parameters.get('location', 'unknown')
            pose = self._get_pose_for_location(location)
            if pose:
                goal_msg.pose = pose
            else:
                return {'success': False, 'error_message': f'Unknown location: {location}'}

        # Send goal and wait for result
        try:
            goal_handle = self.node_interface.nav_client.send_goal(goal_msg)
            result = goal_handle.get_result()

            return {
                'success': result.result.reached_goal,
                'result_data': {'final_pose': result.result.final_pose}
            }
        except Exception as e:
            return {'success': False, 'error_message': str(e)}

    def _execute_manipulation_action(self, action_node: ActionNode) -> Dict:
        """Execute manipulation action"""
        # This would interface with manipulation action server
        # For now, return mock result
        return {'success': True, 'result_data': {}}

    def _execute_generic_action(self, action_node: ActionNode) -> Dict:
        """Execute generic action based on action type"""
        # Generic action execution logic
        return {'success': True, 'result_data': {}}

    def _get_pose_for_location(self, location: str) -> Optional[PoseStamped]:
        """Get pose for a named location"""
        # In a real system, this would look up location poses from a map
        # For now, return mock poses for known locations
        location_poses = {
            'kitchen': self._create_mock_pose(1.0, 2.0, 0.0),
            'bedroom': self._create_mock_pose(-1.0, 1.0, 1.57),
            'office': self._create_mock_pose(0.0, -2.0, 3.14)
        }
        return location_poses.get(location)

    def _create_mock_pose(self, x: float, y: float, theta: float) -> PoseStamped:
        """Create a mock pose for testing"""
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        # Convert theta to quaternion
        from math import sin, cos
        pose.pose.orientation.z = sin(theta / 2.0)
        pose.pose.orientation.w = cos(theta / 2.0)

        return pose

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an ongoing execution"""
        if execution_id in self.active_executions:
            self.active_executions[execution_id]['status'] = 'cancelled'
            return True
        return False

    def get_execution_status(self, execution_id: str) -> Dict:
        """Get status of an execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            return {
                'execution_id': execution_id,
                'status': context['status'],
                'completed_actions': len(context['completed_actions']),
                'failed_actions': len(context['failed_actions']),
                'total_actions': len(context['graph'].nodes),
                'progress': len(context['completed_actions']) / len(context['graph'].nodes) if context['graph'].nodes else 0,
                'execution_time': time.time() - context['start_time']
            }
        return {'error': 'Execution not found'}
```

## Integration with Language Understanding

### Complete Language-to-Action Pipeline

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import json

class LanguageToActionNode(Node):
    def __init__(self):
        super().__init__('language_to_action_node')

        # Initialize components
        self.language_action_mapper = LanguageActionMapper()
        self.action_graph_builder = LanguageToActionGraph(self.language_action_mapper)
        self.action_executor = ActionExecutionManager(self)

        # ROS 2 interfaces
        self.command_sub = self.create_subscription(
            String,
            'language_command',
            self.command_callback,
            10
        )

        self.action_status_pub = self.create_publisher(
            String,
            'action_status',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'action_feedback',
            10
        )

        # Action clients for robot capabilities
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manip_client = ActionClient(self, ManipulationAction, 'manipulation')

        # Store execution IDs
        self.active_executions = {}

        self.get_logger().info("Language-to-Action Node initialized")

    def command_callback(self, msg: String):
        """Process language commands and convert to action graphs"""
        command_text = msg.data
        self.get_logger().info(f"Received command: {command_text}")

        try:
            # Parse the command (in a real system, this would come from NLU)
            parsed_command = self._parse_command(command_text)

            if not parsed_command:
                self.get_logger().warn(f"Could not parse command: {command_text}")
                self._send_feedback("I didn't understand that command")
                return

            # Create action graph
            commands = [parsed_command]  # For single command, create list
            action_graph = self.action_graph_builder.create_action_graph(commands)

            # Validate graph
            validation_issues = action_graph.validate_graph()
            if any(issues for issues in validation_issues.values()):
                self.get_logger().error(f"Action graph validation failed: {validation_issues}")
                self._send_feedback("Could not create a valid action plan")
                return

            self.get_logger().info(f"Created action graph with {len(action_graph.nodes)} actions")

            # Execute the action graph
            execution_id = self.action_executor.execute_action_graph(
                action_graph,
                completion_callback=self._execution_completed_callback
            )

            self.active_executions[execution_id] = action_graph

            self._send_feedback(f"Started executing: {command_text}")

        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            self._send_feedback(f"Error processing command: {str(e)}")

    def _parse_command(self, command_text: str) -> Dict:
        """Parse command text into structured format"""
        # This is a simplified parser - in reality, this would come from NLU system
        command_text_lower = command_text.lower()

        # Determine action type based on keywords
        if any(keyword in command_text_lower for keyword in ['go', 'navigate', 'move', 'travel']):
            action = 'navigation'
            # Extract location (simplified)
            for location in ['kitchen', 'bedroom', 'office', 'living room']:
                if location in command_text_lower:
                    return {
                        'action': action,
                        'entities': [
                            {'type': 'location', 'value': location}
                        ]
                    }
        elif any(keyword in command_text_lower for keyword in ['pick', 'get', 'grab', 'take']):
            action = 'manipulation_pick'
            # Extract object
            for obj in ['cup', 'bottle', 'book', 'phone']:
                if obj in command_text_lower:
                    return {
                        'action': action,
                        'entities': [
                            {'type': 'object', 'value': obj}
                        ]
                    }

        return None

    def _execution_completed_callback(self, execution_id: str, context: Dict):
        """Callback when action execution completes"""
        status = self.action_executor.get_execution_status(execution_id)

        completion_msg = String()
        completion_msg.data = json.dumps({
            'execution_id': execution_id,
            'status': status['status'],
            'completed_actions': status['completed_actions'],
            'failed_actions': status['failed_actions'],
            'total_actions': status['total_actions'],
            'execution_time': status['execution_time']
        })

        self.action_status_pub.publish(completion_msg)

        # Log completion
        success_rate = (status['completed_actions'] / status['total_actions']) if status['total_actions'] > 0 else 0
        self.get_logger().info(f"Execution {execution_id} completed with {success_rate:.1%} success rate")

        # Clean up
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]

    def _send_feedback(self, message: str):
        """Send feedback to user"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

    def get_active_execution_status(self):
        """Get status of all active executions"""
        statuses = {}
        for exec_id in list(self.active_executions.keys()):
            status = self.action_executor.get_execution_status(exec_id)
            if 'error' not in status:
                statuses[exec_id] = status
            else:
                # Remove invalid execution
                if exec_id in self.active_executions:
                    del self.active_executions[exec_id]

        return statuses

def main(args=None):
    rclpy.init(args=args)
    node = LanguageToActionNode()

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

## Error Handling and Recovery

### Robust Action Execution with Recovery

```python
from enum import Enum
import random

class RecoveryStrategy(Enum):
    RETRY = "retry"
    SKIP = "skip"
    REPLAN = "replan"
    ABORT = "abort"

class RobustActionExecutor:
    def __init__(self, base_executor: ActionExecutionManager):
        self.base_executor = base_executor
        self.recovery_strategies = self._initialize_recovery_strategies()

    def _initialize_recovery_strategies(self) -> Dict[str, RecoveryStrategy]:
        """Initialize error-to-recovery mappings"""
        return {
            'navigation_timeout': RecoveryStrategy.RETRY,
            'obstacle_detected': RecoveryStrategy.REPLAN,
            'object_not_found': RecoveryStrategy.SKIP,
            'manipulation_failure': RecoveryStrategy.RETRY,
            'connection_lost': RecoveryStrategy.ABORT
        }

    def execute_action_graph_with_recovery(self, graph: ActionGraph,
                                         max_retries: int = 3) -> Dict:
        """Execute action graph with automatic recovery"""
        execution_context = {
            'graph': graph,
            'completed_actions': set(),
            'failed_actions': set(),
            'skipped_actions': set(),
            'retry_counts': {node_id: 0 for node_id in graph.nodes.keys()},
            'max_retries': max_retries,
            'status': 'running'
        }

        while execution_context['status'] == 'running':
            # Get ready actions
            ready_actions = self._get_ready_actions_with_retry(execution_context)

            if not ready_actions:
                # Check if execution is complete
                all_actions = set(graph.nodes.keys())
                processed = (execution_context['completed_actions'] |
                           execution_context['failed_actions'] |
                           execution_context['skipped_actions'])

                if processed == all_actions:
                    execution_context['status'] = 'completed'
                else:
                    # Wait and continue
                    time.sleep(0.1)
                    continue

            # Execute ready actions
            for action_id in ready_actions:
                if action_id not in execution_context['completed_actions'] and \
                   action_id not in execution_context['failed_actions'] and \
                   action_id not in execution_context['skipped_actions']:
                    result = self._execute_action_with_recovery(
                        action_id, execution_context
                    )

                    if result.success:
                        execution_context['completed_actions'].add(action_id)
                    else:
                        # Handle failure
                        self._handle_action_failure(action_id, result, execution_context)

        return {
            'success': len(execution_context['failed_actions']) == 0,
            'completed': execution_context['completed_actions'],
            'failed': execution_context['failed_actions'],
            'skipped': execution_context['skipped_actions'],
            'total_actions': len(graph.nodes)
        }

    def _get_ready_actions_with_retry(self, context: Dict) -> List[str]:
        """Get ready actions, considering retry status"""
        ready = []
        graph = context['graph']
        completed = context['completed_actions']
        failed = context['failed_actions']
        skipped = context['skipped_actions']

        for node_id in graph.graph.nodes:
            node = graph.graph.nodes[node_id]['node_data']

            # Check dependencies
            dependencies_met = all(dep in completed for dep in node.dependencies)
            not_processed = node_id not in completed and node_id not in failed and node_id not in skipped

            # Check retry status
            retry_allowed = context['retry_counts'][node_id] < context['max_retries']

            if dependencies_met and not_processed and retry_allowed:
                ready.append(node_id)

        return ready

    def _execute_action_with_recovery(self, action_id: str, context: Dict) -> ActionResult:
        """Execute action with potential recovery"""
        action_node = context['graph'].nodes[action_id]

        try:
            result = self.base_executor._execute_ros_action(action_node)

            if not result.success:
                # Increment retry count
                context['retry_counts'][action_id] += 1

            return result

        except Exception as e:
            context['retry_counts'][action_id] += 1
            return ActionResult(
                success=False,
                result_data={},
                error_message=str(e),
                execution_time=0.0
            )

    def _handle_action_failure(self, action_id: str, result: ActionResult, context: Dict):
        """Handle action failure with appropriate recovery"""
        error_type = self._classify_error(result.error_message)
        strategy = self.recovery_strategies.get(error_type, RecoveryStrategy.ABORT)

        if strategy == RecoveryStrategy.RETRY:
            self._handle_retry(action_id, context)
        elif strategy == RecoveryStrategy.SKIP:
            self._handle_skip(action_id, context)
        elif strategy == RecoveryStrategy.REPLAN:
            self._handle_replan(action_id, context)
        elif strategy == RecoveryStrategy.ABORT:
            self._handle_abort(context)

    def _classify_error(self, error_message: str) -> str:
        """Classify error type from error message"""
        error_lower = error_message.lower()

        if 'timeout' in error_lower or 'navigation' in error_lower:
            return 'navigation_timeout'
        elif 'obstacle' in error_lower or 'blocked' in error_lower:
            return 'obstacle_detected'
        elif 'not found' in error_lower or 'missing' in error_lower:
            return 'object_not_found'
        elif 'manipulation' in error_lower or 'grasp' in error_lower:
            return 'manipulation_failure'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'connection_lost'
        else:
            return 'unknown_error'

    def _handle_retry(self, action_id: str, context: Dict):
        """Handle retry for failed action"""
        retry_count = context['retry_counts'][action_id]
        if retry_count <= context['max_retries']:
            # Action will be retried automatically in next cycle
            pass
        else:
            # Max retries exceeded
            context['failed_actions'].add(action_id)

    def _handle_skip(self, action_id: str, context: Dict):
        """Handle skipping of failed action"""
        context['skipped_actions'].add(action_id)
        # Update dependencies of subsequent actions
        self._update_dependencies_after_skip(action_id, context)

    def _handle_replan(self, action_id: str, context: Dict):
        """Handle replanning after failure"""
        # In a real system, this would trigger replanning
        # For now, skip the failed action and continue
        context['skipped_actions'].add(action_id)
        self._update_dependencies_after_skip(action_id, context)

    def _handle_abort(self, context: Dict):
        """Handle aborting execution"""
        context['status'] = 'failed'

    def _update_dependencies_after_skip(self, skipped_action_id: str, context: Dict):
        """Update dependencies when an action is skipped"""
        # Mark skipped action as completed for dependency purposes
        # This allows dependent actions to proceed
        graph = context['graph']

        # Find all actions that depend on the skipped action
        for node_id in graph.nodes:
            node = graph.nodes[node_id]
            if skipped_action_id in node.dependencies:
                # Remove the dependency since the action was skipped
                # In a real system, you might need to replan these actions
                pass
```

## Performance Optimization

### Efficient Action Graph Processing

```python
import asyncio
from typing import List, Set
import heapq

class OptimizedActionExecutor:
    def __init__(self, node_interface):
        self.node_interface = node_interface
        self.action_queue = asyncio.PriorityQueue()
        self.concurrent_limit = 3  # Limit concurrent actions
        self.action_priorities = self._initialize_priorities()

    def _initialize_priorities(self) -> Dict[str, int]:
        """Initialize action priorities for scheduling"""
        return {
            'NavigateToPose': 1,  # Navigation first
            'FindObject': 2,      # Then find objects
            'ManipulationAction': 3,  # Manipulation last
            'default': 5
        }

    async def execute_action_graph_optimized(self, graph: ActionGraph) -> Dict:
        """Execute action graph with optimized scheduling"""
        execution_context = {
            'graph': graph,
            'completed_actions': set(),
            'failed_actions': set(),
            'ready_queue': [],
            'in_progress': set(),
            'results': {}
        }

        # Initialize ready queue
        self._update_ready_queue(execution_context)

        # Execute with limited concurrency
        active_tasks = set()

        while (execution_context['completed_actions'].union(
               execution_context['failed_actions']) != set(graph.nodes.keys()) or
               active_tasks):

            # Start new tasks up to concurrency limit
            while (len(active_tasks) < self.concurrent_limit and
                   execution_context['ready_queue']):

                action_id = heapq.heappop(execution_context['ready_queue'])
                if action_id not in execution_context['in_progress']:
                    task = asyncio.create_task(
                        self._execute_action_async(action_id, execution_context)
                    )
                    active_tasks.add(task)

            # Wait for at least one task to complete
            if active_tasks:
                done, active_tasks = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed tasks
                for task in done:
                    result = await task
                    action_id = result['action_id']

                    if result['success']:
                        execution_context['completed_actions'].add(action_id)
                    else:
                        execution_context['failed_actions'].add(action_id)

                    execution_context['results'][action_id] = result
                    execution_context['in_progress'].discard(action_id)

                    # Update ready queue based on new completions
                    self._update_ready_queue(execution_context)

        return {
            'completed': len(execution_context['completed_actions']),
            'failed': len(execution_context['failed_actions']),
            'total': len(graph.nodes),
            'results': execution_context['results']
        }

    def _update_ready_queue(self, context: Dict):
        """Update the ready queue with newly available actions"""
        ready_actions = context['graph'].get_ready_actions(
            context['completed_actions']
        )

        for action_id in ready_actions:
            if (action_id not in context['in_progress'] and
                action_id not in context['completed_actions'] and
                action_id not in context['failed_actions']):

                node = context['graph'].nodes[action_id]
                priority = self.action_priorities.get(
                    node.action_type,
                    self.action_priorities['default']
                )

                heapq.heappush(context['ready_queue'], (priority, action_id))

    async def _execute_action_async(self, action_id: str, context: Dict) -> Dict:
        """Execute a single action asynchronously"""
        context['in_progress'].add(action_id)

        node = context['graph'].nodes[action_id]
        start_time = time.time()

        try:
            # Simulate action execution
            result = await self._execute_ros_action_async(node)

            return {
                'action_id': action_id,
                'success': result.success,
                'result_data': result.result_data,
                'error_message': result.error_message,
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'action_id': action_id,
                'success': False,
                'result_data': {},
                'error_message': str(e),
                'execution_time': time.time() - start_time
            }

    async def _execute_ros_action_async(self, action_node: ActionNode) -> ActionResult:
        """Execute ROS action asynchronously"""
        # Simulate async execution
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate action time

        # Simulate success/failure
        success = random.random() > 0.1  # 90% success rate

        return ActionResult(
            success=success,
            result_data={'simulated': True} if success else {},
            error_message="" if success else "Simulated failure",
            execution_time=random.uniform(0.1, 0.5)
        )

class ActionGraphOptimizer:
    """Optimize action graphs for better execution performance"""
    def __init__(self):
        self.optimization_rules = self._initialize_optimization_rules()

    def _initialize_optimization_rules(self) -> Dict[str, callable]:
        """Initialize optimization rules"""
        return {
            'merge_similar_actions': self._merge_similar_actions,
            'reorder_for_efficiency': self._reorder_for_efficiency,
            'parallelize_independent': self._parallelize_independent_actions
        }

    def optimize_graph(self, graph: ActionGraph) -> ActionGraph:
        """Apply optimizations to action graph"""
        optimized_graph = graph  # Start with original graph

        # Apply optimization rules
        for rule_name, rule_func in self.optimization_rules.items():
            optimized_graph = rule_func(optimized_graph)

        return optimized_graph

    def _merge_similar_actions(self, graph: ActionGraph) -> ActionGraph:
        """Merge similar consecutive actions"""
        # Example: merge multiple navigation actions to nearby locations
        # This is a simplified example
        return graph

    def _reorder_for_efficiency(self, graph: ActionGraph) -> ActionGraph:
        """Reorder actions for better efficiency"""
        # Example: group navigation actions together to minimize travel
        # This is a simplified example
        return graph

    def _parallelize_independent_actions(self, graph: ActionGraph) -> ActionGraph:
        """Identify and mark independent actions for parallel execution"""
        # Analyze dependency graph to find independent action sets
        # This is a simplified example
        return graph
```

## Summary

Translating language goals into ROS 2 action graphs involves:

- **Action Mapping**: Converting natural language concepts to ROS 2 actions
- **Graph Construction**: Building dependency-aware action sequences
- **Execution Management**: Coordinating action execution with monitoring
- **Error Handling**: Managing failures and recovery strategies
- **Optimization**: Improving performance through scheduling and parallelization

This pipeline enables robots to understand high-level commands and execute them as coordinated sequences of low-level actions.

## Exercises

1. Implement a language-to-action mapper for your robot's capabilities
2. Create an action graph executor with dependency management
3. Develop error recovery strategies for action failures
4. Build an optimized action scheduling system for better performance