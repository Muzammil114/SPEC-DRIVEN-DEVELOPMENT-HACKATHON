---
sidebar_position: 5
---

# LLM-Based Cognitive Planning for Robotics

## Learning Objectives
- Understand how large language models can be used for robotic planning
- Learn to integrate LLMs with robotic action systems
- Implement hierarchical task planning using LLMs
- Create plan validation and execution monitoring systems
- Develop techniques for handling plan failures and replanning

## Introduction to LLM-Based Cognitive Planning

Large Language Models (LLMs) have emerged as powerful tools for cognitive planning in robotics. Unlike traditional planning algorithms that rely on symbolic representations and predefined rules, LLMs can understand natural language commands, reason about complex tasks, and generate detailed execution plans that bridge high-level goals with low-level robot actions.

### Traditional vs LLM-Based Planning

| Aspect | Traditional Planning | LLM-Based Planning |
|--------|---------------------|-------------------|
| Input Format | Structured symbolic representations | Natural language commands |
| Knowledge Source | Hand-coded rules and ontologies | Pre-trained world knowledge |
| Reasoning Type | Logical, symbolic reasoning | Semantic, contextual reasoning |
| Flexibility | Limited to predefined actions | Can handle novel situations |
| Adaptability | Requires manual updates | Can generalize to new tasks |

### Cognitive Planning Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Human User    │    │   LLM Planner   │    │  Robot Actions  │
│                 │    │                 │    │                 │
│ "Clean the     │───▶│ 1. Goal         │───▶│ 4. Action       │
│   kitchen"      │    │    Analysis     │    │    Execution    │
│                 │    │                 │    │                 │
└─────────────────┘    │ 2. Task         │    └─────────────────┘
                       │    Decomposition │
                       │                 │
                       │ 3. Action       │
                       │    Sequencing    │
                       └─────────────────┘
```

## LLM Integration for Task Planning

### Planning Interface Design

```python
import openai
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskStep:
    """Represents a single step in a robot task plan"""
    id: str
    description: str
    action: str  # ROS action name or custom action
    parameters: Dict[str, Any]
    prerequisites: List[str]  # IDs of tasks that must complete first
    expected_duration: float  # Estimated time in seconds
    confidence: float = 1.0

@dataclass
class TaskPlan:
    """Represents a complete task plan"""
    id: str
    goal: str
    steps: List[TaskStep]
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = 0.0
    estimated_duration: float = 0.0

class LLMPlanner:
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.robot_capabilities = self._get_robot_capabilities()
        self.environment_knowledge = {}

    def _get_robot_capabilities(self) -> Dict[str, Any]:
        """Define robot's capabilities for LLM to reference"""
        return {
            "navigation": {
                "actions": ["navigate_to", "move_to", "go_to"],
                "parameters": ["target_location", "speed", "avoid_obstacles"]
            },
            "manipulation": {
                "actions": ["pick_up", "place", "grasp", "release"],
                "parameters": ["object_type", "object_id", "target_location"]
            },
            "perception": {
                "actions": ["detect_object", "find_location", "scan_environment"],
                "parameters": ["object_type", "location", "search_area"]
            },
            "communication": {
                "actions": ["speak", "report_status"],
                "parameters": ["message", "recipient"]
            }
        }

    def create_task_plan(self, goal: str, context: Dict[str, Any] = None) -> Optional[TaskPlan]:
        """Create a task plan using LLM for a given goal"""
        prompt = self._create_planning_prompt(goal, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            plan_data = json.loads(response.choices[0].message.content)
            return self._parse_plan_response(plan_data, goal)

        except Exception as e:
            print(f"Error creating task plan: {e}")
            return None

    def _create_planning_prompt(self, goal: str, context: Dict[str, Any] = None) -> str:
        """Create prompt for task planning"""
        context_str = json.dumps(context) if context else "{}"

        prompt = f"""
        Create a detailed task plan to achieve the following goal: "{goal}"

        Context: {context_str}

        Robot Capabilities:
        {json.dumps(self.robot_capabilities, indent=2)}

        Available Locations (if relevant):
        - kitchen, bedroom, office, living room, bathroom, hallway

        Available Objects (if relevant):
        - cup, bottle, book, phone, laptop, keys, food, water

        Requirements:
        1. Break down the goal into sequential steps
        2. Each step should be executable by the robot
        3. Include relevant parameters for each action
        4. Consider dependencies between steps
        5. Estimate duration for each step

        Respond in JSON format:
        {{
            "goal": "original goal",
            "steps": [
                {{
                    "id": "step_1",
                    "description": "What the robot should do",
                    "action": "action_name",
                    "parameters": {{"param1": "value1"}},
                    "prerequisites": ["step_0"],  // IDs of steps that must complete first
                    "expected_duration": 10.0  // in seconds
                }}
            ],
            "estimated_duration": 120.0  // total estimated duration in seconds
        }}
        """
        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for planning"""
        return """
        You are an expert robot task planner. Create detailed, executable plans for robots
        based on high-level goals. Each plan should consist of concrete, sequential steps
        that the robot can execute. Consider the robot's capabilities, environmental constraints,
        and logical dependencies between actions. Be specific about parameters and realistic
        about timing estimates.
        """

    def _parse_plan_response(self, plan_data: Dict, original_goal: str) -> TaskPlan:
        """Parse LLM response into TaskPlan object"""
        steps = []
        for step_data in plan_data.get('steps', []):
            step = TaskStep(
                id=step_data['id'],
                description=step_data['description'],
                action=step_data['action'],
                parameters=step_data.get('parameters', {}),
                prerequisites=step_data.get('prerequisites', []),
                expected_duration=step_data.get('expected_duration', 10.0),
                confidence=step_data.get('confidence', 0.9)
            )
            steps.append(step)

        plan = TaskPlan(
            id=f"plan_{int(time.time())}",
            goal=original_goal,
            steps=steps,
            estimated_duration=plan_data.get('estimated_duration', sum(s.expected_duration for s in steps))
        )

        return plan

    def validate_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        """Validate the task plan for feasibility"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'suggestions': []
        }

        # Check if all actions are supported
        for step in plan.steps:
            if not self._is_action_supported(step.action):
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f"Action '{step.action}' is not supported by the robot"
                )

        # Check for circular dependencies
        dependency_graph = {step.id: step.prerequisites for step in plan.steps}
        if self._has_circular_dependencies(dependency_graph):
            validation_results['is_valid'] = False
            validation_results['issues'].append("Plan has circular dependencies")

        # Check if prerequisites exist
        all_step_ids = {step.id for step in plan.steps}
        for step in plan.steps:
            invalid_prereqs = set(step.prerequisites) - all_step_ids
            if invalid_prereqs:
                validation_results['issues'].append(
                    f"Step '{step.id}' has invalid prerequisites: {invalid_prereqs}"
                )
                validation_results['is_valid'] = False

        return validation_results

    def _is_action_supported(self, action: str) -> bool:
        """Check if action is supported by robot"""
        for capability_category in self.robot_capabilities.values():
            if action in capability_category['actions']:
                return True
        return False

    def _has_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> bool:
        """Check for circular dependencies in task graph"""
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for prereq in dependency_graph.get(node, []):
                if prereq not in visited:
                    if has_cycle(prereq):
                        return True
                elif prereq in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in dependency_graph:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False
```

## Hierarchical Task Planning

### Multi-Level Planning Architecture

```python
from typing import Union

class HierarchicalPlanner:
    def __init__(self, llm_planner: LLMPlanner):
        self.llm_planner = llm_planner
        self.task_hierarchy = {}

    def create_hierarchical_plan(self, high_level_goal: str,
                                context: Dict[str, Any] = None) -> Optional[TaskPlan]:
        """Create a hierarchical plan with multiple levels of abstraction"""
        # First, let LLM break down the high-level goal
        high_level_plan = self.llm_planner.create_task_plan(high_level_goal, context)

        if not high_level_plan:
            return None

        # For complex goals, further decompose high-level steps
        refined_plan = self._refine_plan(high_level_plan, context)
        return refined_plan

    def _refine_plan(self, plan: TaskPlan, context: Dict[str, Any]) -> TaskPlan:
        """Refine a plan by decomposing complex steps"""
        refined_steps = []

        for step in plan.steps:
            # Check if this step is too complex and needs decomposition
            if self._is_complex_step(step):
                sub_plan = self._decompose_step(step, context)
                if sub_plan:
                    refined_steps.extend(sub_plan.steps)
                else:
                    refined_steps.append(step)  # Keep original if decomposition fails
            else:
                refined_steps.append(step)

        plan.steps = refined_steps
        plan.estimated_duration = sum(s.expected_duration for s in refined_steps)
        return plan

    def _is_complex_step(self, step: TaskStep) -> bool:
        """Determine if a step needs further decomposition"""
        # Complex steps might be high-level abstractions
        complex_keywords = ['clean', 'organize', 'arrange', 'prepare', 'setup']
        step_desc = step.description.lower()

        # If description contains complex keywords or is very general
        return any(keyword in step_desc for keyword in complex_keywords) or len(step_desc.split()) > 5

    def _decompose_step(self, step: TaskStep, context: Dict[str, Any]) -> Optional[TaskPlan]:
        """Decompose a complex step into sub-steps"""
        sub_goal = f"Perform the following task: {step.description}"

        # Create context that includes the parent task
        sub_context = context.copy() if context else {}
        sub_context['parent_task'] = step.description
        sub_context['parent_parameters'] = step.parameters

        return self.llm_planner.create_task_plan(sub_goal, sub_context)

    def execute_hierarchical_plan(self, plan: TaskPlan, robot_interface) -> Dict[str, Any]:
        """Execute a hierarchical plan with monitoring and replanning"""
        execution_results = {
            'plan_id': plan.id,
            'status': 'in_progress',
            'completed_steps': [],
            'failed_steps': [],
            'execution_log': []
        }

        for step in plan.steps:
            self._log_execution(execution_results, f"Starting step: {step.description}")

            try:
                # Execute the step
                step_result = robot_interface.execute_action(step.action, step.parameters)

                if step_result['success']:
                    execution_results['completed_steps'].append(step.id)
                    self._log_execution(execution_results, f"Completed step: {step.description}")
                else:
                    execution_results['failed_steps'].append(step.id)
                    self._log_execution(execution_results, f"Failed step: {step.description}")

                    # Decide whether to continue, retry, or replan
                    if self._should_replan(step, step_result):
                        self._log_execution(execution_results, "Initiating replanning...")
                        # Replanning logic would go here
                        break
                    elif self._should_retry(step, step_result):
                        # Retry logic
                        pass
                    else:
                        # Continue to next step
                        pass

            except Exception as e:
                execution_results['failed_steps'].append(step.id)
                self._log_execution(execution_results, f"Error executing step {step.id}: {str(e)}")

        execution_results['status'] = 'completed' if not execution_results['failed_steps'] else 'partial_success'
        return execution_results

    def _should_replan(self, step: TaskStep, result: Dict) -> bool:
        """Determine if replanning is needed after step failure"""
        # Replan if failure is due to environmental changes or missing information
        failure_reason = result.get('error', '')
        replan_triggers = ['not found', 'obstacle', 'unreachable', 'unknown']
        return any(trigger in failure_reason.lower() for trigger in replan_triggers)

    def _should_retry(self, step: TaskStep, result: Dict) -> bool:
        """Determine if step should be retried"""
        # Retry for temporary failures like network issues
        failure_reason = result.get('error', '')
        retry_triggers = ['timeout', 'connection', 'temporary']
        return any(trigger in failure_reason.lower() for trigger in retry_triggers)

    def _log_execution(self, results: Dict, message: str):
        """Log execution progress"""
        timestamp = time.time()
        results['execution_log'].append({
            'timestamp': timestamp,
            'message': message
        })
```

## Plan Validation and Monitoring

### Execution Monitoring System

```python
import asyncio
import threading
from datetime import datetime, timedelta

class PlanExecutionMonitor:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.active_plans = {}
        self.execution_threads = {}
        self.monitoring = True

    def start_plan_execution(self, plan: TaskPlan) -> str:
        """Start execution of a plan with monitoring"""
        plan_id = plan.id
        self.active_plans[plan_id] = {
            'plan': plan,
            'status': TaskStatus.IN_PROGRESS,
            'start_time': datetime.now(),
            'completed_steps': set(),
            'current_step': 0,
            'execution_log': []
        }

        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitor_plan_execution,
            args=(plan_id,)
        )
        self.execution_threads[plan_id] = monitor_thread
        monitor_thread.start()

        return plan_id

    def _monitor_plan_execution(self, plan_id: str):
        """Monitor plan execution in a separate thread"""
        plan_info = self.active_plans[plan_id]
        plan = plan_info['plan']

        for i, step in enumerate(plan.steps):
            if not self.monitoring:
                break

            # Wait for step prerequisites
            if not self._check_prerequisites(plan_info, step.prerequisites):
                self._log_to_plan(plan_id, f"Waiting for prerequisites of step {step.id}")
                time.sleep(1)
                continue

            # Execute the step
            self._log_to_plan(plan_id, f"Executing step {i+1}/{len(plan.steps)}: {step.description}")
            plan_info['current_step'] = i

            step_result = self._execute_step_with_monitoring(plan_id, step)

            if step_result['success']:
                plan_info['completed_steps'].add(step.id)
                self._log_to_plan(plan_id, f"Step completed: {step.description}")
            else:
                self._log_to_plan(plan_id, f"Step failed: {step.description}")
                # Handle failure
                self._handle_step_failure(plan_id, step, step_result)
                break

            # Check if plan should be cancelled
            if plan_info['status'] == TaskStatus.CANCELLED:
                break

        # Update final status
        if plan_info['status'] != TaskStatus.CANCELLED:
            completed_count = len(plan_info['completed_steps'])
            total_count = len(plan.steps)
            if completed_count == total_count:
                plan_info['status'] = TaskStatus.COMPLETED
            else:
                plan_info['status'] = TaskStatus.FAILED

    def _check_prerequisites(self, plan_info: Dict, prerequisites: List[str]) -> bool:
        """Check if all prerequisites for a step are completed"""
        completed = plan_info['completed_steps']
        return all(prereq in completed for prereq in prerequisites)

    def _execute_step_with_monitoring(self, plan_id: str, step: TaskStep) -> Dict:
        """Execute a step with timeout and monitoring"""
        start_time = time.time()
        timeout = step.expected_duration * 2  # 2x expected duration as timeout

        try:
            # Execute the robot action
            result = self.robot_interface.execute_action(step.action, step.parameters)

            # Monitor execution if needed
            execution_time = time.time() - start_time
            if execution_time > step.expected_duration * 1.5:
                self._log_to_plan(plan_id, f"Step taking longer than expected: {execution_time:.1f}s vs {step.expected_duration:.1f}s")

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _handle_step_failure(self, plan_id: str, step: TaskStep, result: Dict):
        """Handle step execution failure"""
        plan_info = self.active_plans[plan_id]

        # Log the failure
        error_msg = result.get('error', 'Unknown error')
        self._log_to_plan(plan_id, f"Step failed with error: {error_msg}")

        # Determine next action based on failure type
        if self._is_recoverable_failure(result):
            self._attempt_recovery(plan_id, step, result)
        else:
            # Mark plan as failed
            plan_info['status'] = TaskStatus.FAILED

    def _is_recoverable_failure(self, result: Dict) -> bool:
        """Determine if a failure is recoverable"""
        error = result.get('error', '').lower()
        recoverable_errors = [
            'timeout', 'temporary', 'connection', 'retry', 'blocked'
        ]
        return any(err in error for err in recoverable_errors)

    def _attempt_recovery(self, plan_id: str, step: TaskStep, result: Dict):
        """Attempt to recover from step failure"""
        plan_info = self.active_plans[plan_id]
        self._log_to_plan(plan_id, f"Attempting recovery for failed step: {step.description}")

        # Simple recovery: retry the step
        retry_result = self._execute_step_with_monitoring(plan_id, step)

        if retry_result['success']:
            plan_info['completed_steps'].add(step.id)
            self._log_to_plan(plan_id, "Recovery successful")
        else:
            plan_info['status'] = TaskStatus.FAILED
            self._log_to_plan(plan_id, "Recovery failed, plan terminated")

    def _log_to_plan(self, plan_id: str, message: str):
        """Add log entry to plan execution"""
        if plan_id in self.active_plans:
            timestamp = datetime.now()
            log_entry = {
                'timestamp': timestamp,
                'message': message
            }
            self.active_plans[plan_id]['execution_log'].append(log_entry)

    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel an active plan"""
        if plan_id in self.active_plans:
            self.active_plans[plan_id]['status'] = TaskStatus.CANCELLED
            return True
        return False

    def get_plan_status(self, plan_id: str) -> Dict:
        """Get current status of a plan"""
        if plan_id in self.active_plans:
            plan_info = self.active_plans[plan_id]
            return {
                'plan_id': plan_id,
                'status': plan_info['status'].value,
                'completed_steps': len(plan_info['completed_steps']),
                'total_steps': len(plan_info['plan'].steps),
                'progress': len(plan_info['completed_steps']) / len(plan_info['plan'].steps) if plan_info['plan'].steps else 0,
                'execution_time': (datetime.now() - plan_info['start_time']).total_seconds(),
                'current_step': plan_info['current_step'],
                'log': plan_info['execution_log'][-10:]  # Last 10 log entries
            }
        return {'error': 'Plan not found'}

    def stop_monitoring(self):
        """Stop all monitoring"""
        self.monitoring = False
        for thread in self.execution_threads.values():
            thread.join()
```

## Integration with Robotics Systems

### ROS 2 Integration Example

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
import json
import asyncio

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize LLM planner
        api_key = self.declare_parameter('openai_api_key', '').value
        if not api_key:
            self.get_logger().error("OpenAI API key not provided")
            return

        self.llm_planner = LLMPlanner(api_key=api_key)
        self.hierarchical_planner = HierarchicalPlanner(self.llm_planner)
        self.monitor = PlanExecutionMonitor(self)

        # ROS 2 interfaces
        self.goal_sub = self.create_subscription(
            String,
            'high_level_goal',
            self.goal_callback,
            10
        )

        self.plan_pub = self.create_publisher(
            String,
            'generated_plan',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            'plan_status',
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'planning_feedback',
            10
        )

        # Action clients for robot capabilities
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manip_client = ActionClient(self, ManipulationAction, 'manipulation')

        # Store active plan IDs
        self.active_plan_ids = set()

        self.get_logger().info("LLM Planning Node initialized")

    def goal_callback(self, msg: String):
        """Process high-level goals and create plans"""
        goal_text = msg.data
        self.get_logger().info(f"Received goal: {goal_text}")

        try:
            # Create plan using LLM
            plan = self.hierarchical_planner.create_hierarchical_plan(goal_text)

            if not plan:
                self.get_logger().error("Failed to create plan")
                self._send_feedback("Could not create a plan for that goal")
                return

            # Validate the plan
            validation = self.llm_planner.validate_plan(plan)
            if not validation['is_valid']:
                self.get_logger().error(f"Plan validation failed: {validation['issues']}")
                self._send_feedback(f"Plan validation failed: {', '.join(validation['issues'])}")
                return

            self.get_logger().info(f"Created plan with {len(plan.steps)} steps")

            # Publish the plan
            plan_msg = String()
            plan_msg.data = json.dumps({
                'plan_id': plan.id,
                'goal': plan.goal,
                'steps': [
                    {
                        'id': step.id,
                        'description': step.description,
                        'action': step.action,
                        'parameters': step.parameters,
                        'prerequisites': step.prerequisites,
                        'expected_duration': step.expected_duration
                    }
                    for step in plan.steps
                ],
                'estimated_duration': plan.estimated_duration
            })
            self.plan_pub.publish(plan_msg)

            # Start execution monitoring
            plan_id = self.monitor.start_plan_execution(plan)
            self.active_plan_ids.add(plan_id)

            self._send_feedback(f"Plan created and execution started: {plan.goal}")

        except Exception as e:
            self.get_logger().error(f"Error processing goal: {e}")
            self._send_feedback(f"Error processing goal: {str(e)}")

    def execute_action(self, action_name: str, parameters: Dict) -> Dict:
        """Execute a robot action via ROS 2"""
        try:
            if action_name == 'navigate_to':
                return self._execute_navigation(parameters)
            elif action_name == 'pick_up':
                return self._execute_manipulation(parameters)
            elif action_name == 'place':
                return self._execute_manipulation(parameters)
            elif action_name == 'speak':
                return self._execute_communication(parameters)
            else:
                return {'success': False, 'error': f'Unknown action: {action_name}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _execute_navigation(self, parameters: Dict) -> Dict:
        """Execute navigation action"""
        target_location = parameters.get('target_location', 'unknown')

        # In a real system, you'd look up the actual pose for the location
        # For now, we'll use a mock implementation
        if target_location in ['kitchen', 'bedroom', 'office']:
            # Mock successful navigation
            return {'success': True, 'result': f"Navigated to {target_location}"}
        else:
            return {'success': False, 'error': f"Unknown location: {target_location}"}

    def _execute_manipulation(self, parameters: Dict) -> Dict:
        """Execute manipulation action"""
        object_type = parameters.get('object_type', 'unknown')

        # Mock manipulation
        if object_type in ['cup', 'bottle', 'book', 'phone']:
            return {'success': True, 'result': f"Manipulated {object_type}"}
        else:
            return {'success': False, 'error': f"Unknown object: {object_type}"}

    def _execute_communication(self, parameters: Dict) -> Dict:
        """Execute communication action"""
        message = parameters.get('message', 'No message provided')
        # In a real system, this would trigger speech synthesis
        self.get_logger().info(f"Speaking: {message}")
        return {'success': True, 'result': f"Spoke: {message}"}

    def _send_feedback(self, message: str):
        """Send feedback to user"""
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)

    def timer_callback(self):
        """Periodic status updates"""
        for plan_id in list(self.active_plan_ids):
            status = self.monitor.get_plan_status(plan_id)
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                self.active_plan_ids.discard(plan_id)

            # Publish status
            status_msg = String()
            status_msg.data = json.dumps(status)
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanningNode()

    # Create timer for periodic status updates
    timer = node.create_timer(1.0, node.timer_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.monitor.stop_monitoring()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Plan Refinement and Adaptation

### Online Plan Adaptation

```python
class AdaptivePlanner:
    def __init__(self, base_planner: LLMPlanner):
        self.base_planner = base_planner
        self.execution_history = []
        self.adaptation_rules = self._initialize_adaptation_rules()

    def _initialize_adaptation_rules(self) -> Dict[str, callable]:
        """Initialize rules for plan adaptation"""
        return {
            'obstacle_detected': self._handle_obstacle,
            'object_not_found': self._handle_missing_object,
            'location_unreachable': self._handle_unreachable_location,
            'resource_unavailable': self._handle_unavailable_resource
        }

    def adapt_plan(self, plan: TaskPlan, execution_context: Dict) -> TaskPlan:
        """Adapt plan based on execution context and feedback"""
        adaptation_needed = self._analyze_execution_feedback(execution_context)

        if not adaptation_needed:
            return plan

        # Apply adaptation rules
        adapted_plan = self._apply_adaptation_rules(plan, execution_context)
        return adapted_plan

    def _analyze_execution_feedback(self, context: Dict) -> bool:
        """Analyze execution context to determine if adaptation is needed"""
        failure_count = context.get('failure_count', 0)
        success_rate = context.get('success_rate', 1.0)
        environmental_changes = context.get('environmental_changes', [])

        # Adapt if there are failures or environmental changes
        return failure_count > 0 or success_rate < 0.7 or len(environmental_changes) > 0

    def _apply_adaptation_rules(self, plan: TaskPlan, context: Dict) -> TaskPlan:
        """Apply adaptation rules to modify the plan"""
        adapted_steps = []

        for step in plan.steps:
            # Check if this step needs adaptation based on context
            adapted_step = self._adapt_step(step, context)
            adapted_steps.append(adapted_step)

        # Create new plan with adapted steps
        adapted_plan = TaskPlan(
            id=f"adapted_{plan.id}",
            goal=plan.goal,
            steps=adapted_steps,
            estimated_duration=plan.estimated_duration
        )

        return adapted_plan

    def _adapt_step(self, step: TaskStep, context: Dict) -> TaskStep:
        """Adapt a single step based on context"""
        # Check for specific issues that might affect this step
        if context.get('obstacles', {}).get(step.id):
            # Modify navigation steps to avoid obstacles
            if step.action == 'navigate_to':
                new_params = step.parameters.copy()
                new_params['avoid_obstacles'] = True
                new_params['alternative_route'] = True
                return TaskStep(
                    id=f"{step.id}_adapted",
                    description=f"{step.description} (avoiding obstacles)",
                    action=step.action,
                    parameters=new_params,
                    prerequisites=step.prerequisites,
                    expected_duration=step.expected_duration * 1.5,  # Longer due to obstacle avoidance
                    confidence=step.confidence * 0.8
                )

        # Check for object availability
        unavailable_objects = context.get('unavailable_objects', [])
        if (step.action in ['pick_up', 'manipulate'] and
            step.parameters.get('object_type') in unavailable_objects):
            # Try alternative object or skip
            return self._create_alternative_step(step, context)

        return step

    def _create_alternative_step(self, original_step: TaskStep, context: Dict) -> TaskStep:
        """Create an alternative step when original can't be executed"""
        # For now, return a step that reports the issue
        # In practice, this might involve finding alternatives
        return TaskStep(
            id=f"{original_step.id}_alternative",
            description=f"Report: Could not execute {original_step.description}",
            action='report_status',
            parameters={'message': f"Could not find {original_step.parameters.get('object_type')}"},
            prerequisites=original_step.prerequisites,
            expected_duration=5.0,
            confidence=1.0
        )

    def _handle_obstacle(self, plan: TaskPlan, context: Dict) -> TaskPlan:
        """Handle obstacle detection in plan"""
        # Replan navigation steps to avoid obstacles
        adapted_steps = []
        for step in plan.steps:
            if step.action == 'navigate_to':
                # Add obstacle avoidance parameters
                new_params = step.parameters.copy()
                new_params['avoid_obstacles'] = True
                step.parameters = new_params
            adapted_steps.append(step)
        plan.steps = adapted_steps
        return plan

    def _handle_missing_object(self, plan: TaskPlan, context: Dict) -> TaskPlan:
        """Handle missing object in plan"""
        # Modify plan to search for alternative or report issue
        return plan  # Simplified implementation

    def _handle_unreachable_location(self, plan: TaskPlan, context: Dict) -> TaskPlan:
        """Handle unreachable location in plan"""
        # Modify plan to find alternative route or location
        return plan  # Simplified implementation

    def _handle_unavailable_resource(self, plan: TaskPlan, context: Dict) -> TaskPlan:
        """Handle unavailable resource in plan"""
        # Modify plan to use alternative resources
        return plan  # Simplified implementation

class LearningBasedPlanner:
    """Planner that learns from execution history"""
    def __init__(self, base_planner: LLMPlanner):
        self.base_planner = base_planner
        self.execution_history = []
        self.performance_metrics = {}

    def create_improved_plan(self, goal: str, context: Dict = None) -> TaskPlan:
        """Create plan using learned patterns from execution history"""
        # Analyze similar past executions
        similar_goals = self._find_similar_goals(goal)

        if similar_goals:
            # Use patterns from successful executions
            learned_patterns = self._extract_successful_patterns(similar_goals)
            context = context or {}
            context['learned_patterns'] = learned_patterns

        # Create plan with learned context
        return self.base_planner.create_task_plan(goal, context)

    def _find_similar_goals(self, goal: str) -> List[Dict]:
        """Find similar goals in execution history"""
        similar = []
        goal_lower = goal.lower()

        for execution in self.execution_history:
            if execution['goal'].lower() == goal_lower or self._are_goals_similar(goal, execution['goal']):
                similar.append(execution)

        return similar

    def _are_goals_similar(self, goal1: str, goal2: str) -> bool:
        """Determine if two goals are similar"""
        # Simple keyword-based similarity
        g1_words = set(goal1.lower().split())
        g2_words = set(goal2.lower().split())
        intersection = g1_words.intersection(g2_words)
        union = g1_words.union(g2_words)

        if union:
            jaccard_similarity = len(intersection) / len(union)
            return jaccard_similarity > 0.3  # 30% overlap threshold

        return False

    def _extract_successful_patterns(self, similar_executions: List[Dict]) -> Dict:
        """Extract successful patterns from similar executions"""
        patterns = {
            'common_steps': [],
            'successful_sequences': [],
            'timing_patterns': {},
            'resource_usage': {}
        }

        successful_executions = [ex for ex in similar_executions if ex.get('success', False)]

        if not successful_executions:
            return patterns

        # Analyze common successful steps
        from collections import Counter
        all_steps = []
        for execution in successful_executions:
            for step in execution.get('steps', []):
                all_steps.append(step['action'])

        step_counter = Counter(all_steps)
        patterns['common_steps'] = [item for item, count in step_counter.most_common(5)]

        # Calculate average execution times
        for execution in successful_executions:
            for i, step in enumerate(execution.get('steps', [])):
                action = step['action']
                duration = step.get('actual_duration', step.get('expected_duration', 10.0))

                if action not in patterns['timing_patterns']:
                    patterns['timing_patterns'][action] = []
                patterns['timing_patterns'][action].append(duration)

        # Average the timing patterns
        for action in patterns['timing_patterns']:
            avg_time = sum(patterns['timing_patterns'][action]) / len(patterns['timing_patterns'][action])
            patterns['timing_patterns'][action] = avg_time

        return patterns

    def update_execution_history(self, plan: TaskPlan, result: Dict):
        """Update execution history with new results"""
        history_entry = {
            'plan_id': plan.id,
            'goal': plan.goal,
            'steps': [
                {
                    'action': step.action,
                    'parameters': step.parameters,
                    'expected_duration': step.expected_duration,
                    'actual_duration': result.get('step_durations', {}).get(step.id, step.expected_duration)
                }
                for step in plan.steps
            ],
            'success': result.get('success', False),
            'total_duration': result.get('total_duration', plan.estimated_duration),
            'timestamp': time.time()
        }
        self.execution_history.append(history_entry)

        # Keep only recent history (last 100 executions)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
```

## Performance Optimization

### Efficient Planning Pipeline

```python
import asyncio
import concurrent.futures
from typing import List
import time

class EfficientLLMPlanner:
    def __init__(self, api_key: str, max_concurrent: int = 3):
        self.planner = LLMPlanner(api_key)
        self.adaptive_planner = AdaptivePlanner(self.planner)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        self.cache = {}
        self.max_cache_size = 50

    async def create_multiple_plans(self, goals: List[str],
                                  contexts: List[Dict] = None) -> List[TaskPlan]:
        """Create multiple plans concurrently"""
        if contexts is None:
            contexts = [None] * len(goals)

        # Prepare tasks
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self.executor,
                self.planner.create_task_plan,
                goal,
                context
            )
            for goal, context in zip(goals, contexts)
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        plans = []
        for result in results:
            if isinstance(result, TaskPlan):
                plans.append(result)
            elif isinstance(result, Exception):
                print(f"Plan creation failed: {result}")

        return plans

    def create_plan_with_cache(self, goal: str, context: Dict = None) -> TaskPlan:
        """Create plan with caching for frequently requested goals"""
        cache_key = f"{goal}_{hash(str(context)) if context else ''}"

        # Check cache first
        if cache_key in self.cache:
            cached_plan, timestamp = self.cache[cache_key]
            # Check if cache is still valid (e.g., less than 1 hour old)
            if time.time() - timestamp < 3600:
                return cached_plan

        # Create new plan
        plan = self.planner.create_task_plan(goal, context)

        # Add to cache
        if plan:
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            self.cache[cache_key] = (plan, time.time())

        return plan

    def validate_and_optimize_plan(self, plan: TaskPlan) -> TaskPlan:
        """Validate and optimize a plan"""
        # First, validate the plan
        validation = self.planner.validate_plan(plan)
        if not validation['is_valid']:
            raise ValueError(f"Plan validation failed: {validation['issues']}")

        # Optimize the plan (simplified optimization)
        optimized_steps = self._optimize_steps(plan.steps)
        plan.steps = optimized_steps

        # Update estimated duration
        plan.estimated_duration = sum(s.expected_duration for s in optimized_steps)

        return plan

    def _optimize_steps(self, steps: List[TaskStep]) -> List[TaskStep]:
        """Optimize steps for better execution"""
        # Remove redundant steps
        optimized = []
        for step in steps:
            # Check if this step is redundant compared to previous ones
            is_redundant = False
            for prev_step in optimized:
                if (step.action == prev_step.action and
                    step.parameters == prev_step.parameters):
                    is_redundant = True
                    break

            if not is_redundant:
                optimized.append(step)

        return optimized

class PlanningOptimizer:
    """System for optimizing planning performance"""
    def __init__(self):
        self.planning_times = []
        self.success_rates = []
        self.plan_complexity = []

    def measure_planning_performance(self, goal: str, plan_creation_func) -> Dict:
        """Measure performance of plan creation"""
        start_time = time.time()

        try:
            plan = plan_creation_func(goal)
            success = plan is not None
            planning_time = time.time() - start_time

            # Store metrics
            self.planning_times.append(planning_time)
            self.success_rates.append(success)
            if plan:
                self.plan_complexity.append(len(plan.steps))

            return {
                'success': success,
                'planning_time': planning_time,
                'plan_steps': len(plan.steps) if plan else 0,
                'average_time': sum(self.planning_times) / len(self.planning_times) if self.planning_times else 0
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'planning_time': time.time() - start_time
            }

    def get_performance_summary(self) -> Dict:
        """Get summary of planning performance"""
        return {
            'total_plans': len(self.planning_times),
            'success_rate': sum(self.success_rates) / len(self.success_rates) if self.success_rates else 0,
            'avg_planning_time': sum(self.planning_times) / len(self.planning_times) if self.planning_times else 0,
            'avg_plan_complexity': sum(self.plan_complexity) / len(self.plan_complexity) if self.plan_complexity else 0,
            'total_time': sum(self.planning_times)
        }
```

## Summary

LLM-based cognitive planning for robotics provides:

- **Natural Language Interface**: Direct conversion of human goals to robot actions
- **Hierarchical Planning**: Multi-level task decomposition and organization
- **Adaptive Execution**: Real-time plan adjustment based on feedback
- **Learning Capabilities**: Improvement through execution experience
- **Validation and Monitoring**: Safe and reliable plan execution
- **Performance Optimization**: Efficient planning for real-time applications

These capabilities enable robots to understand complex, high-level commands and execute them reliably in dynamic environments.

## Exercises

1. Implement an LLM-based planner that creates task plans from natural language
2. Create a hierarchical planning system that decomposes complex tasks
3. Develop a plan monitoring and execution system with failure handling
4. Build an adaptive planning system that learns from execution experience