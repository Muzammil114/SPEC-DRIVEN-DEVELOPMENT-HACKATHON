---
sidebar_position: 2
---

# Capstone Project: Developing a Conversational Humanoid Robot

## Learning Objectives
- Integrate all modules into a complete conversational robot system
- Implement spoken command reception with natural language processing
- Create language-to-action conversion pipeline
- Develop navigation and manipulation capabilities
- Demonstrate sim-to-real transfer strategies

## Introduction

The conversational humanoid robot project synthesizes all the concepts learned in the previous modules. This capstone project challenges you to build a complete system that can receive spoken commands, understand natural language, plan and execute actions, and interact naturally with humans in a physical environment.

### Project Scope
- **Voice Command Processing**: Receive and interpret spoken commands
- **Natural Language Understanding**: Convert language to executable actions
- **Navigation**: Move through environments safely and efficiently
- **Object Manipulation**: Identify and manipulate objects
- **Conversation Management**: Engage in meaningful interactions
- **Sim-to-Real Transfer**: Deploy simulation-tested behaviors to real robots

## System Design and Architecture

### Overall System Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONVERSATIONAL ROBOT SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │   Voice Input   │  │   Language      │  │    Task Planning       │ │
│  │   Processing    │  │   Understanding │  │    & Action Graphs     │ │
│  │                 │  │                 │  │                        │ │
│  │ - Speech Recog. │  │ - Intent Class. │  │ - LLM Integration      │ │
│  │ - Audio Proc.   │  │ - Entity Extr.  │  │ - Action Sequencing    │ │
│  │ - Noise Red.    │  │ - Context Mgmt. │  │ - Dependency Mgmt.     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
│         │                       │                        │              │
│         ▼                       ▼                        ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    ACTION EXECUTION FRAMEWORK                       │ │
│  │                                                                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │ │
│  │  │ Navigation  │  │ Manipulation│  │ Perception  │  │ Safety &    │ │ │
│  │  │   System    │  │   System    │  │   System    │  │ Monitoring  │ │ │
│  │  │             │  │             │  │             │  │             │ │ │
│  │  │ - Path Plan │  │ - Grasp Sel │  │ - Object Det│  │ - Collision │ │ │
│  │  │ - Localize  │  │ - Motion Pl │  │ - Pose Est. │  │ - Emerg Stp │ │ │
│  │  │ - Navigate  │  │ - Execute   │  │ - Scene Gr  │  │ - Status    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│         │                       │                        │              │
│         ▼                       ▼                        ▼              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Robot Drivers  │  │  Hardware      │  │  Human Interaction     │ │
│  │   & Control     │  │  Interfaces     │  │    & Feedback          │ │
│  │                 │  │                 │  │                        │ │
│  │ - Motor Ctrl    │  │ - Actuators     │  │ - Speech Synthesis     │ │
│  │ - Feedback Loop │  │ - Sensors       │  │ - Visual Feedback      │ │
│  │ - Coordination  │  │ - Calibration   │  │ - Emotional Express.   │ │
│  └─────────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Integration
Each component must work seamlessly together to create a cohesive conversational robot experience.

## Implementation Phases

### Phase 1: Voice Command Processing and NLU (Weeks 1-2)

#### 1.1 Speech Recognition System
```python
import whisper
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class VoiceCommand:
    text: str
    confidence: float
    timestamp: float
    intent: Optional[str] = None
    entities: Optional[Dict] = None

class VoiceProcessor:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=self.device)

        # Voice activity detection
        self.energy_threshold = 1000
        self.silence_duration = 1.0

        # Command history for context
        self.command_history = []

    def process_audio(self, audio_data: np.ndarray) -> VoiceCommand:
        """Process audio data and return recognized command"""
        # Preprocess audio
        audio_tensor = torch.from_numpy(audio_data).float()

        # Transcribe using Whisper
        result = self.model.transcribe(
            audio_tensor,
            language="en",
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        command = VoiceCommand(
            text=result["text"].strip(),
            confidence=result.get("avg_logprob", 0.0),
            timestamp=time.time()
        )

        # Add to history
        self.command_history.append(command)
        if len(self.command_history) > 10:
            self.command_history.pop(0)

        return command

    def detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect if voice is present in audio chunk"""
        energy = np.mean(np.abs(audio_chunk))
        return energy > self.energy_threshold
```

#### 1.2 Natural Language Understanding
```python
import re
import spacy
from typing import Tuple, Dict

class NaturalLanguageUnderstanding:
    def __init__(self):
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define intent patterns
        self.intent_patterns = {
            'navigation': [
                r'go to (the )?(?P<location>\w+)',
                r'move to (the )?(?P<location>\w+)',
                r'navigate to (the )?(?P<location>\w+)',
                r'go (to )?(the )?(?P<location>\w+)'
            ],
            'manipulation': [
                r'pick up (the )?(?P<object>\w+)',
                r'get (the )?(?P<object>\w+)',
                r'bring (me )?(the )?(?P<object>\w+)',
                r'grab (the )?(?P<object>\w+)'
            ],
            'information': [
                r'what is (the )?(?P<information>\w+)',
                r'tell me about (the )?(?P<information>\w+)',
                r'where is (the )?(?P<information>\w+)'
            ]
        }

    def parse_command(self, text: str) -> Tuple[str, Dict]:
        """Parse command text to extract intent and entities"""
        text_lower = text.lower()

        # Try pattern matching first
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities = match.groupdict()
                    return intent, entities

        # If pattern matching fails, use NLP
        if self.nlp:
            return self._parse_with_nlp(text)

        # Fallback
        return 'unknown', {}

    def _parse_with_nlp(self, text: str) -> Tuple[str, Dict]:
        """Parse command using spaCy NLP"""
        doc = self.nlp(text)

        entities = {}
        intent = 'unknown'

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:
                entities['location'] = ent.text
            elif ent.label_ in ['OBJECT', 'PRODUCT']:
                entities['object'] = ent.text

        # Determine intent based on verbs
        for token in doc:
            if token.pos_ == 'VERB':
                if token.lemma_ in ['go', 'move', 'navigate']:
                    intent = 'navigation'
                elif token.lemma_ in ['pick', 'get', 'bring', 'grab']:
                    intent = 'manipulation'
                elif token.lemma_ in ['tell', 'what', 'where']:
                    intent = 'information'

        return intent, entities
```

### Phase 2: Task Planning and Action Graphs (Weeks 3-4)

#### 2.1 LLM-Based Task Planning
```python
import openai
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ActionStep:
    id: str
    action: str
    parameters: Dict
    description: str
    dependencies: List[str]
    timeout: float = 30.0

@dataclass
class TaskPlan:
    goal: str
    steps: List[ActionStep]
    estimated_duration: float

class LLMTaskPlanner:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def create_task_plan(self, goal: str, context: Dict = None) -> Optional[TaskPlan]:
        """Create task plan using LLM"""
        prompt = self._create_planning_prompt(goal, context)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
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

    def _create_planning_prompt(self, goal: str, context: Dict = None) -> str:
        """Create prompt for task planning"""
        context_str = json.dumps(context) if context else "{}"

        prompt = f"""
        Create a detailed task plan to achieve: "{goal}"

        Context: {context_str}

        Robot Capabilities:
        - Navigation: navigate_to_pose, move_to_location
        - Manipulation: pick_up_object, place_object, grasp_object
        - Perception: detect_object, find_location, scan_environment
        - Communication: speak, report_status

        Available Locations: kitchen, bedroom, office, living_room, bathroom, hallway
        Available Objects: cup, bottle, book, phone, laptop, keys, food, water

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
                    "action": "action_name",
                    "parameters": {{"param1": "value1"}},
                    "description": "What the robot should do",
                    "dependencies": ["step_0"],
                    "timeout": 30.0
                }}
            ],
            "estimated_duration": 120.0
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
            step = ActionStep(
                id=step_data['id'],
                action=step_data['action'],
                parameters=step_data.get('parameters', {}),
                description=step_data['description'],
                dependencies=step_data.get('dependencies', []),
                timeout=step_data.get('timeout', 30.0)
            )
            steps.append(step)

        plan = TaskPlan(
            goal=original_goal,
            steps=steps,
            estimated_duration=plan_data.get('estimated_duration', 0.0)
        )

        return plan
```

#### 2.2 Action Graph Execution
```python
import networkx as nx
from typing import Set, Dict
import time

class ActionGraphExecutor:
    def __init__(self, node_interface):
        self.node_interface = node_interface
        self.completed_actions = set()
        self.failed_actions = set()
        self.in_progress = set()

    def execute_task_plan(self, plan: TaskPlan) -> Dict:
        """Execute task plan with dependency management"""
        # Build dependency graph
        graph = nx.DiGraph()
        for step in plan.steps:
            graph.add_node(step.id, step=step)
            for dep in step.dependencies:
                graph.add_edge(dep, step.id)

        execution_order = list(nx.topological_sort(graph))
        results = {}

        for step_id in execution_order:
            step = graph.nodes[step_id]['step']

            # Check if dependencies are satisfied
            dependencies_satisfied = all(
                dep_id in self.completed_actions
                for dep_id in step.dependencies
            )

            if not dependencies_satisfied:
                results[step_id] = {
                    'success': False,
                    'error': 'Dependencies not satisfied'
                }
                self.failed_actions.add(step_id)
                continue

            # Execute the action
            start_time = time.time()
            try:
                result = self._execute_action(step)

                execution_time = time.time() - start_time

                if result.get('success', False):
                    self.completed_actions.add(step_id)
                    results[step_id] = {
                        'success': True,
                        'result': result,
                        'execution_time': execution_time
                    }
                else:
                    self.failed_actions.add(step_id)
                    results[step_id] = {
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'execution_time': execution_time
                    }

            except Exception as e:
                self.failed_actions.add(step_id)
                results[step_id] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }

        return {
            'plan_completed': len(self.failed_actions) == 0,
            'results': results,
            'completed_count': len(self.completed_actions),
            'failed_count': len(self.failed_actions),
            'total_count': len(plan.steps)
        }

    def _execute_action(self, step: ActionStep) -> Dict:
        """Execute a single action"""
        action_type = step.action

        if action_type == 'navigate_to_pose':
            return self._execute_navigation(step.parameters)
        elif action_type == 'pick_up_object':
            return self._execute_manipulation_pick(step.parameters)
        elif action_type == 'place_object':
            return self._execute_manipulation_place(step.parameters)
        elif action_type == 'detect_object':
            return self._execute_perception_detect(step.parameters)
        elif action_type == 'speak':
            return self._execute_communication_speak(step.parameters)
        else:
            return {
                'success': False,
                'error': f'Unknown action type: {action_type}'
            }

    def _execute_navigation(self, parameters: Dict) -> Dict:
        """Execute navigation action"""
        # This would interface with Navigation2
        # For now, return mock result
        target_location = parameters.get('target_location', 'unknown')

        # In real implementation, this would use ROS 2 Navigation2
        # nav_client = self.node_interface.nav_client
        # goal = NavigateToPose.Goal()
        # goal.pose = ...
        # result = nav_client.send_goal(goal)

        return {
            'success': True,
            'result': f'Navigated to {target_location}',
            'final_pose': parameters.get('target_location', 'unknown')
        }

    def _execute_manipulation_pick(self, parameters: Dict) -> Dict:
        """Execute manipulation pick action"""
        object_type = parameters.get('object_type', 'unknown')

        # In real implementation, this would use MoveIt2 and manipulation stack
        # manip_client = self.node_interface.manip_client
        # goal = ManipulationAction.Goal()
        # goal.object_type = object_type
        # result = manip_client.send_goal(goal)

        return {
            'success': True,
            'result': f'Picked up {object_type}',
            'object_grasped': object_type
        }

    def _execute_perception_detect(self, parameters: Dict) -> Dict:
        """Execute perception detection action"""
        object_type = parameters.get('object_type', 'any')

        # In real implementation, this would use perception pipelines
        # camera = self.node_interface.camera
        # objects = camera.detect_objects(object_type)

        return {
            'success': True,
            'result': f'Detected {object_type}',
            'objects_found': [object_type]
        }

    def _execute_communication_speak(self, parameters: Dict) -> Dict:
        """Execute communication speak action"""
        message = parameters.get('message', 'Hello')

        # In real implementation, this would use text-to-speech
        # tts = self.node_interface.text_to_speech
        # tts.speak(message)

        return {
            'success': True,
            'result': f'Spoke: {message}',
            'spoken_message': message
        }
```

### Phase 3: Navigation and Manipulation Systems (Weeks 5-6)

#### 3.1 Navigation System Integration
```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
import numpy as np

class NavigationSystem:
    def __init__(self, node_interface):
        self.node = node_interface
        self.nav_client = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        # Location database
        self.locations = {
            'kitchen': self._create_pose(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            'bedroom': self._create_pose(-1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            'office': self._create_pose(0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            'living_room': self._create_pose(1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        }

    def navigate_to_location(self, location_name: str) -> Dict:
        """Navigate to a named location"""
        if location_name not in self.locations:
            return {
                'success': False,
                'error': f'Unknown location: {location_name}'
            }

        target_pose = self.locations[location_name]
        return self.navigate_to_pose(target_pose)

    def navigate_to_pose(self, pose: PoseStamped) -> Dict:
        """Navigate to a specific pose"""
        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return {
                'success': False,
                'error': 'Navigation server not available'
            }

        # Create goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        # Send goal
        try:
            goal_handle = self.nav_client.send_goal_async(goal_msg)

            # Wait for result (in a real system, you'd want to do this asynchronously)
            # This is simplified for the example
            return {
                'success': True,
                'result': 'Navigation completed',
                'final_pose': pose
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _create_pose(self, x, y, z, ox, oy, oz, ow) -> PoseStamped:
        """Create a PoseStamped message"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = ox
        pose.pose.orientation.y = oy
        pose.pose.orientation.z = oz
        pose.pose.orientation.w = ow
        return pose

    def get_current_pose(self) -> PoseStamped:
        """Get current robot pose"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation

            return pose
        except Exception as e:
            self.node.get_logger().error(f'Failed to get current pose: {e}')
            return self._create_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
```

#### 3.2 Manipulation System Integration
```python
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
import numpy as np

class ManipulationSystem:
    def __init__(self, node_interface):
        self.node = node_interface
        self.move_group_client = ActionClient(self.node, MoveGroup, 'move_group')

        # Object database with properties
        self.objects = {
            'cup': {'dimensions': [0.08, 0.08, 0.1], 'weight': 0.2, 'grasp_type': 'top_grasp'},
            'bottle': {'dimensions': [0.07, 0.07, 0.25], 'weight': 0.5, 'grasp_type': 'side_grasp'},
            'book': {'dimensions': [0.2, 0.15, 0.03], 'weight': 0.3, 'grasp_type': 'edge_grasp'},
            'phone': {'dimensions': [0.15, 0.07, 0.01], 'weight': 0.2, 'grasp_type': 'corner_grasp'}
        }

    def pick_up_object(self, object_type: str, location: Pose = None) -> Dict:
        """Pick up an object of specified type"""
        if object_type not in self.objects:
            return {
                'success': False,
                'error': f'Unknown object type: {object_type}'
            }

        object_info = self.objects[object_type]

        # Plan grasp based on object properties
        grasp_pose = self._plan_grasp_pose(object_type, location, object_info)

        # Execute pick and place
        return self._execute_pick_and_place(grasp_pose, object_info)

    def _plan_grasp_pose(self, object_type: str, location: Pose, object_info: Dict) -> Pose:
        """Plan appropriate grasp pose for object"""
        grasp_pose = Pose()

        if location:
            # Use provided location
            grasp_pose.position = location.position
            grasp_pose.orientation = location.orientation
        else:
            # Default location (would come from perception)
            grasp_pose.position.x = 0.5  # 50cm in front of robot
            grasp_pose.position.y = 0.0
            grasp_pose.position.z = 0.1  # 10cm above ground

        # Adjust orientation based on grasp type
        grasp_type = object_info['grasp_type']

        if grasp_type == 'top_grasp':
            # Approach from top (for cups, bowls)
            grasp_pose.orientation.x = 0.0
            grasp_pose.orientation.y = 0.707
            grasp_pose.orientation.z = 0.0
            grasp_pose.orientation.w = 0.707
        elif grasp_type == 'side_grasp':
            # Approach from side (for bottles, cylinders)
            grasp_pose.orientation.x = 0.0
            grasp_pose.orientation.y = 0.0
            grasp_pose.orientation.z = 0.0
            grasp_pose.orientation.w = 1.0
        elif grasp_type == 'edge_grasp':
            # Approach from edge (for books, flat objects)
            grasp_pose.orientation.x = 0.707
            grasp_pose.orientation.y = 0.0
            grasp_pose.orientation.z = 0.0
            grasp_pose.orientation.w = 0.707

        return grasp_pose

    def _execute_pick_and_place(self, grasp_pose: Pose, object_info: Dict) -> Dict:
        """Execute the pick and place action"""
        # This would interface with MoveIt2 in a real system
        # For now, return mock result

        # Check if object is reachable
        if not self._is_reachable(grasp_pose):
            return {
                'success': False,
                'error': 'Object not reachable'
            }

        # Simulate grasp success/failure based on object properties
        success_probability = self._calculate_grasp_success(object_info)

        if np.random.random() < success_probability:
            return {
                'success': True,
                'result': f'Successfully picked up {object_info}',
                'object_grasped': True
            }
        else:
            return {
                'success': False,
                'error': 'Grasp failed'
            }

    def _is_reachable(self, pose: Pose) -> bool:
        """Check if pose is reachable by robot"""
        # Simplified reachability check
        # In real system, would use MoveIt2 IK solvers
        distance = np.sqrt(pose.position.x**2 + pose.position.y**2 + pose.position.z**2)
        return distance < 1.0  # Within 1 meter reach

    def _calculate_grasp_success(self, object_info: Dict) -> float:
        """Calculate probability of successful grasp"""
        # Factors affecting grasp success:
        # - Object weight
        # - Object shape/dimensions
        # - Grasp type
        # - Surface properties (not modeled here)

        base_success = 0.95  # Base success rate

        # Reduce success rate for heavier objects
        weight_factor = max(0.7, 1.0 - (object_info['weight'] / 2.0))

        # Reduce success rate for awkward shapes
        dimensions = object_info['dimensions']
        aspect_ratio = max(dimensions) / min(dimensions)
        shape_factor = max(0.8, 1.0 - (aspect_ratio - 1.0) * 0.1)

        final_success = base_success * weight_factor * shape_factor
        return min(1.0, final_success)

    def place_object(self, target_pose: Pose) -> Dict:
        """Place held object at target pose"""
        # In real system, would release object at target pose
        return {
            'success': True,
            'result': 'Successfully placed object',
            'placed_at': target_pose
        }
```

### Phase 4: Perception and Safety Systems (Weeks 7-8)

#### 4.1 Perception System
```python
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import torch
import clip

class PerceptionSystem:
    def __init__(self, node_interface):
        self.node = node_interface
        self.bridge = CvBridge()

        # Initialize CLIP for zero-shot object detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        except:
            self.clip_model = None
            self.clip_preprocess = None
            self.node.get_logger().warn("CLIP model not available, using basic detection")

        # Camera subscription
        self.image_sub = self.node.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.latest_image = None

    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.node.get_logger().error(f"Error processing image: {e}")

    def detect_objects(self, object_types: List[str] = None) -> Dict:
        """Detect objects in the environment"""
        if self.latest_image is None:
            return {
                'success': False,
                'error': 'No image available'
            }

        if self.clip_model and object_types:
            return self._detect_with_clip(self.latest_image, object_types)
        else:
            return self._detect_basic_objects(self.latest_image)

    def _detect_with_clip(self, image: np.ndarray, object_types: List[str]) -> Dict:
        """Use CLIP for zero-shot object detection"""
        # Preprocess image
        pil_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Create text descriptions
        text_descriptions = [f"a photo of {obj}" for obj in object_types]
        text_input = clip.tokenize(text_descriptions).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(pil_image)
            text_features = self.clip_model.encode_text(text_input)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(len(object_types))

        # Format results
        detections = []
        for i in range(len(indices)):
            obj_idx = indices[i].item()
            confidence = values[i].item()
            if confidence > 0.1:  # Threshold
                detections.append({
                    'object': object_types[obj_idx],
                    'confidence': confidence,
                    'location': 'center_of_image'  # Simplified
                })

        return {
            'success': True,
            'detections': detections,
            'image_processed': True
        }

    def _detect_basic_objects(self, image: np.ndarray) -> Dict:
        """Basic object detection using color and shape analysis"""
        # Convert to HSV for color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for common objects
        color_ranges = {
            'red_cup': ([0, 50, 50], [10, 255, 255]),
            'blue_bottle': ([100, 50, 50], [130, 255, 255]),
            'yellow_book': ([20, 50, 50], [30, 255, 255])
        }

        detections = []
        for obj_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate center
                    center_x, center_y = x + w//2, y + h//2

                    detections.append({
                        'object': obj_name.split('_')[1],  # Remove color prefix
                        'confidence': 0.7,  # Estimated confidence
                        'bbox': [x, y, w, h],
                        'center': [center_x, center_y]
                    })

        return {
            'success': True,
            'detections': detections,
            'image_processed': True
        }

    def get_object_pose(self, object_name: str) -> Dict:
        """Get 3D pose of an object"""
        # In a real system, this would use depth information
        # For now, return mock pose based on detection

        detections = self.detect_objects([object_name])

        if detections['success'] and detections['detections']:
            detection = detections['detections'][0]

            # Convert 2D detection to 3D pose (simplified)
            # In real system, would use depth camera and triangulation
            pose = {
                'x': 0.5,  # 50cm in front of robot
                'y': detection.get('center', [0, 0])[0] * 0.001,  # Convert pixel to meters
                'z': 0.1,  # 10cm above ground
                'orientation': [0, 0, 0, 1]  # Quaternion (identity)
            }

            return {
                'success': True,
                'pose': pose,
                'object_found': True
            }
        else:
            return {
                'success': False,
                'error': f'Object {object_name} not found',
                'object_found': False
            }
```

#### 4.2 Safety and Monitoring System
```python
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from rclpy.timer import Timer
import threading

class SafetySystem:
    def __init__(self, node_interface):
        self.node = node_interface

        # Safety parameters
        self.safety_distance = 0.3  # 30cm safety margin
        self.emergency_stop_distance = 0.1  # 10cm emergency stop
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s

        # Sensor subscriptions
        self.lidar_sub = self.node.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.imu_sub = self.node.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Velocity command publisher for emergency stop
        self.vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Safety state
        self.lidar_ranges = None
        self.imu_data = None
        self.emergency_stop_active = False
        self.safety_lock = threading.Lock()

        # Safety monitoring timer
        self.safety_timer = self.node.create_timer(0.1, self.safety_check)

    def lidar_callback(self, msg: LaserScan):
        """Process LiDAR data for obstacle detection"""
        with self.safety_lock:
            self.lidar_ranges = np.array(msg.ranges)
            self.lidar_ranges = self.lidar_ranges[~np.isnan(self.lidar_ranges)]  # Remove NaN values

    def imu_callback(self, msg: Imu):
        """Process IMU data for motion monitoring"""
        with self.safety_lock:
            self.imu_data = {
                'linear_acceleration': [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ],
                'angular_velocity': [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ]
            }

    def safety_check(self):
        """Continuously check safety conditions"""
        with self.safety_lock:
            # Check for obstacles
            if self.lidar_ranges is not None and len(self.lidar_ranges) > 0:
                min_distance = np.min(self.lidar_ranges)

                if min_distance < self.emergency_stop_distance:
                    self._emergency_stop()
                    self.node.get_logger().warn(f"EMERGENCY STOP: Obstacle at {min_distance:.2f}m")
                elif min_distance < self.safety_distance:
                    self.node.get_logger().warn(f"Safety warning: Obstacle at {min_distance:.2f}m")

            # Check IMU data for abnormal motion
            if self.imu_data is not None:
                linear_acc = np.linalg.norm(self.imu_data['linear_acceleration'])
                angular_vel = np.linalg.norm(self.imu_data['angular_velocity'])

                if linear_acc > 50.0:  # Excessive acceleration
                    self._emergency_stop()
                    self.node.get_logger().warn(f"EMERGENCY STOP: Excessive acceleration: {linear_acc:.2f}")

                if angular_vel > 10.0:  # Excessive rotation
                    self._emergency_stop()
                    self.node.get_logger().warn(f"EMERGENCY STOP: Excessive rotation: {angular_vel:.2f}")

    def _emergency_stop(self):
        """Activate emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True

            # Send stop command
            stop_cmd = Twist()
            self.vel_pub.publish(stop_cmd)

            # Log the event
            self.node.get_logger().error("EMERGENCY STOP ACTIVATED")

    def is_safe_to_move(self) -> bool:
        """Check if it's safe to move the robot"""
        with self.safety_lock:
            if self.emergency_stop_active:
                return False

            if self.lidar_ranges is not None and len(self.lidar_ranges) > 0:
                min_distance = np.min(self.lidar_ranges)
                return min_distance > self.safety_distance

            return True  # If no sensor data, assume safe (shouldn't happen in real system)

    def validate_command(self, cmd_vel: Twist) -> Tuple[bool, str]:
        """Validate velocity command for safety"""
        with self.safety_lock:
            # Check velocity limits
            if abs(cmd_vel.linear.x) > self.max_linear_speed:
                return False, f"Linear velocity too high: {cmd_vel.linear.x} > {self.max_linear_speed}"

            if abs(cmd_vel.angular.z) > self.max_angular_speed:
                return False, f"Angular velocity too high: {cmd_vel.angular.z} > {self.max_angular_speed}"

            # Check if movement is safe
            if not self.is_safe_to_move():
                return False, "Movement unsafe due to obstacles"

            return True, "Command is safe"
```

## Integration and Testing

### Main Robot System Integration
```python
class ConversationalRobot:
    def __init__(self):
        # Initialize ROS 2 node
        rclpy.init()
        self.node = Node('conversational_robot')

        # Initialize subsystems
        self.voice_processor = VoiceProcessor()
        self.nlu = NaturalLanguageUnderstanding()
        self.task_planner = LLMTaskPlanner(api_key="your-api-key")  # In practice, secure storage
        self.action_executor = ActionGraphExecutor(self.node)
        self.navigation_system = NavigationSystem(self.node)
        self.manipulation_system = ManipulationSystem(self.node)
        self.perception_system = PerceptionSystem(self.node)
        self.safety_system = SafetySystem(self.node)

        # State management
        self.current_task = None
        self.conversation_context = {}

        # Publishers/subscribers
        self.response_pub = self.node.create_publisher(String, 'robot_response', 10)
        self.status_pub = self.node.create_publisher(String, 'robot_status', 10)

    def process_command(self, command_text: str) -> Dict:
        """Process a command from start to finish"""
        self.node.get_logger().info(f"Processing command: {command_text}")

        # Update status
        self._publish_status("Processing command")

        try:
            # 1. Parse command with NLU
            intent, entities = self.nlu.parse_command(command_text)
            self.node.get_logger().info(f"Parsed: intent={intent}, entities={entities}")

            # 2. Create task plan
            context = {
                'current_location': self.navigation_system.get_current_pose(),
                'available_objects': self.perception_system.detect_objects(['cup', 'bottle', 'book'])['detections'],
                'intent': intent,
                'entities': entities
            }

            plan = self.task_planner.create_task_plan(command_text, context)
            if not plan:
                raise Exception("Could not create task plan")

            self.node.get_logger().info(f"Created plan with {len(plan.steps)} steps")

            # 3. Execute plan
            execution_result = self.action_executor.execute_task_plan(plan)

            # 4. Generate response
            response = self._generate_response(intent, entities, execution_result)
            self._publish_response(response)

            # 5. Update context
            self.conversation_context['last_command'] = command_text
            self.conversation_context['last_result'] = execution_result

            return {
                'success': execution_result['plan_completed'],
                'result': execution_result,
                'response': response
            }

        except Exception as e:
            error_msg = f"Error processing command: {str(e)}"
            self.node.get_logger().error(error_msg)
            self._publish_response(f"Sorry, I encountered an error: {str(e)}")
            return {
                'success': False,
                'error': error_msg
            }

    def _generate_response(self, intent: str, entities: Dict, execution_result: Dict) -> str:
        """Generate natural language response based on execution result"""
        if execution_result['plan_completed']:
            if intent == 'navigation':
                location = entities.get('location', 'unknown')
                return f"I have successfully navigated to the {location}."
            elif intent == 'manipulation':
                obj = entities.get('object', 'unknown')
                return f"I have successfully picked up the {obj}."
            else:
                return "I have completed the requested task successfully."
        else:
            failed_count = execution_result['failed_count']
            total_count = execution_result['total_count']
            return f"I completed {total_count - failed_count} out of {total_count} steps. Some steps failed."

    def _publish_response(self, response: str):
        """Publish response to user"""
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

    def _publish_status(self, status: str):
        """Publish robot status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

    def run(self):
        """Run the robot system"""
        self.node.get_logger().info("Conversational Robot System Started")

        # In a real system, this would listen for voice commands
        # For this example, we'll simulate command processing

        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.node.destroy_node()
            rclpy.shutdown()

def main():
    robot = ConversationalRobot()

    # Example commands to test
    test_commands = [
        "Go to the kitchen",
        "Pick up the red cup",
        "Bring me the book from the office"
    ]

    for command in test_commands:
        print(f"\nProcessing: {command}")
        result = robot.process_command(command)
        print(f"Result: {result}")

    # Keep the node running
    robot.run()

if __name__ == '__main__':
    main()
```

## Performance Evaluation and Testing

### Comprehensive Testing Framework
```python
import unittest
from unittest.mock import Mock, patch
import numpy as np

class TestConversationalRobot(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.robot = ConversationalRobot()

    def test_voice_processing(self):
        """Test voice processing functionality"""
        # Mock audio data
        audio_data = np.random.random(16000)  # 1 second of random audio

        with patch('whisper.load_model') as mock_model:
            mock_model.return_value.transcribe.return_value = {
                'text': 'Hello robot',
                'avg_logprob': 0.8
            }

            processor = VoiceProcessor()
            command = processor.process_audio(audio_data)

            self.assertEqual(command.text, 'Hello robot')
            self.assertGreaterEqual(command.confidence, 0.7)

    def test_nlu_parsing(self):
        """Test natural language understanding"""
        nlu = NaturalLanguageUnderstanding()

        # Test navigation command
        intent, entities = nlu.parse_command("Go to the kitchen")
        self.assertEqual(intent, 'navigation')
        self.assertIn('location', entities)
        self.assertEqual(entities['location'], 'kitchen')

        # Test manipulation command
        intent, entities = nlu.parse_command("Pick up the red cup")
        self.assertEqual(intent, 'manipulation')
        self.assertIn('object', entities)
        self.assertEqual(entities['object'], 'cup')

    def test_task_planning(self):
        """Test task planning with mock LLM"""
        with patch('openai.OpenAI') as mock_openai:
            # Mock the API response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '''
            {
                "goal": "Go to kitchen",
                "steps": [
                    {
                        "id": "step_1",
                        "action": "navigate_to_pose",
                        "parameters": {"target_location": "kitchen"},
                        "description": "Navigate to kitchen",
                        "dependencies": [],
                        "timeout": 30.0
                    }
                ],
                "estimated_duration": 60.0
            }
            '''

            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            planner = LLMTaskPlanner(api_key="test-key")
            plan = planner.create_task_plan("Go to the kitchen")

            self.assertIsNotNone(plan)
            self.assertEqual(len(plan.steps), 1)
            self.assertEqual(plan.steps[0].action, "navigate_to_pose")

    def test_safety_system(self):
        """Test safety system functionality"""
        safety = SafetySystem(Mock())

        # Test obstacle detection
        mock_scan = Mock()
        mock_scan.ranges = [0.2, 0.3, 0.4, 0.5]  # Obstacle at 0.2m
        safety.lidar_callback(mock_scan)

        # Safety check should detect unsafe condition
        with patch.object(safety, '_emergency_stop') as mock_stop:
            safety.safety_check()
            # Emergency stop should be triggered for obstacle at 0.2m
            # (which is less than safety distance of 0.3m but greater than emergency distance of 0.1m)
            # So no emergency stop, but warning would be logged

    def test_integration(self):
        """Test integration of multiple components"""
        # Test complete command processing pipeline
        with patch.object(self.robot, 'task_planner') as mock_planner, \
             patch.object(self.robot, 'action_executor') as mock_executor:

            # Mock plan creation
            mock_plan = Mock()
            mock_plan.steps = [Mock()]
            mock_planner.create_task_plan.return_value = mock_plan

            # Mock execution result
            mock_executor.execute_task_plan.return_value = {
                'plan_completed': True,
                'completed_count': 1,
                'failed_count': 0,
                'total_count': 1
            }

            result = self.robot.process_command("Go to kitchen")

            self.assertTrue(result['success'])
            self.assertIsNotNone(result['response'])

class PerformanceBenchmark:
    """Performance benchmarking for the robot system"""
    def __init__(self):
        self.metrics = {
            'command_processing_time': [],
            'planning_time': [],
            'execution_success_rate': [],
            'memory_usage': []
        }

    def benchmark_command_processing(self, commands: List[str], iterations: int = 10):
        """Benchmark command processing performance"""
        import time
        import psutil
        import os

        process = psutil.Process(os.getpid())

        for i in range(iterations):
            for cmd in commands:
                start_time = time.time()

                # Simulate command processing
                with patch('openai.OpenAI'), patch('whisper.load_model'):
                    robot = ConversationalRobot()
                    result = robot.process_command(cmd)

                processing_time = time.time() - start_time
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB

                self.metrics['command_processing_time'].append(processing_time)
                self.metrics['memory_usage'].append(memory_usage)
                self.metrics['execution_success_rate'].append(result.get('success', False))

    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        return {
            'avg_command_processing_time': np.mean(self.metrics['command_processing_time']),
            'std_command_processing_time': np.std(self.metrics['command_processing_time']),
            'avg_memory_usage_mb': np.mean(self.metrics['memory_usage']),
            'execution_success_rate': np.mean(self.metrics['execution_success_rate']),
            'total_commands_processed': len(self.metrics['command_processing_time'])
        }

# Example usage of performance benchmarking
def run_performance_tests():
    """Run performance tests"""
    benchmark = PerformanceBenchmark()

    test_commands = [
        "Go to the kitchen",
        "Pick up the red cup",
        "Navigate to office and bring me the book"
    ]

    benchmark.benchmark_command_processing(test_commands, iterations=5)
    report = benchmark.get_performance_report()

    print("Performance Report:")
    print(f"Average command processing time: {report['avg_command_processing_time']:.3f}s")
    print(f"Memory usage: {report['avg_memory_usage_mb']:.2f} MB")
    print(f"Success rate: {report['execution_success_rate']:.1%}")
    print(f"Commands processed: {report['total_commands_processed']}")
```

## Deployment and Sim-to-Real Transfer

### Simulation Environment Setup
```python
# Simulation-specific configuration
class SimulationEnvironment:
    def __init__(self):
        self.simulation_config = {
            'physics_engine': 'PhysX',
            'render_quality': 'high',
            'real_time_factor': 1.0,
            'gpu_acceleration': True
        }

        self.robot_models = {
            'humanoid': 'path/to/humanoid_model.usd',
            'sensors': {
                'camera': 'path/to/rgb_camera.usd',
                'lidar': 'path/to/lidar_sensor.usd',
                'microphone': 'path/to/microphone_array.usd'
            }
        }

    def setup_simulation(self):
        """Setup Isaac Sim environment"""
        # This would interface with Isaac Sim APIs
        # For now, return mock setup
        return {
            'success': True,
            'environment_ready': True,
            'simulation_time': 0.0
        }

    def transfer_to_real_robot(self, simulation_behavior):
        """Transfer simulation-tested behavior to real robot"""
        # Apply sim-to-real transfer techniques:
        # 1. Domain randomization parameters
        # 2. System identification corrections
        # 3. Latency compensation
        # 4. Safety factor adjustments

        real_robot_config = {
            'domain_randomization_applied': True,
            'system_id_corrections': self._apply_system_id_corrections(simulation_behavior),
            'latency_compensation': self._apply_latency_compensation(simulation_behavior),
            'safety_factors': self._increase_safety_factors(simulation_behavior)
        }

        return real_robot_config

    def _apply_system_id_corrections(self, behavior):
        """Apply system identification corrections"""
        # Apply corrections based on real robot characterization
        corrected_behavior = behavior.copy()
        # Example: Adjust velocities for real robot dynamics
        corrected_behavior['max_velocity'] *= 0.8  # 20% safety margin
        corrected_behavior['acceleration'] *= 0.7  # 30% safety margin
        return corrected_behavior

    def _apply_latency_compensation(self, behavior):
        """Apply latency compensation for real-time execution"""
        # Compensate for sensor and actuator latencies
        compensated_behavior = behavior.copy()
        # Example: Advance timing for actuator delays
        compensated_behavior['timing_offset'] = 0.05  # 50ms advance
        return compensated_behavior

    def _increase_safety_factors(self, behavior):
        """Increase safety factors for real-world deployment"""
        safety_enhanced = behavior.copy()
        safety_enhanced['safety_distance'] *= 1.5  # 50% extra safety margin
        safety_enhanced['obstacle_threshold'] *= 0.8  # More conservative
        return safety_enhanced
```

## Troubleshooting and Validation

### Common Issues and Solutions

#### 1. Voice Recognition Issues
- **Problem**: Poor speech recognition accuracy
- **Solutions**:
  - Improve microphone positioning
  - Apply noise reduction filters
  - Use speaker adaptation techniques
  - Implement multiple model ensemble

#### 2. Navigation Failures
- **Problem**: Robot fails to navigate successfully
- **Solutions**:
  - Calibrate localization system
  - Update environmental maps
  - Adjust navigation parameters
  - Implement dynamic replanning

#### 3. Manipulation Failures
- **Problem**: Object grasping fails frequently
- **Solutions**:
  - Improve object detection accuracy
  - Calibrate gripper and camera
  - Implement grasp learning
  - Add tactile feedback

#### 4. Safety Concerns
- **Problem**: Safety system too restrictive or not restrictive enough
- **Solutions**:
  - Calibrate sensor thresholds
  - Implement layered safety approach
  - Add human-in-the-loop validation
  - Regular safety audits

## Best Practices

### 1. System Design
- **Modular Architecture**: Keep components loosely coupled
- **Error Handling**: Implement graceful degradation
- **Logging**: Comprehensive logging for debugging
- **Testing**: Continuous integration and testing

### 2. Performance Optimization
- **Efficient Processing**: Optimize for real-time constraints
- **Resource Management**: Monitor and manage computational resources
- **Caching**: Cache frequently accessed data
- **Parallel Processing**: Use multi-threading where appropriate

### 3. Safety and Reliability
- **Defense in Depth**: Multiple safety layers
- **Fail-Safe Design**: Default to safe state on failure
- **Regular Testing**: Continuous safety validation
- **User Training**: Proper user education and warnings

## Conclusion

The conversational humanoid robot project demonstrates the integration of all the concepts learned throughout the textbook:

1. **Physical AI Foundation**: Understanding the connection between digital AI and physical systems
2. **ROS 2 Integration**: Building on robust robotics middleware
3. **Simulation and Real-World**: Bridging digital and physical environments
4. **Advanced AI**: Applying LLMs and perception for intelligent behavior

Success in this project indicates readiness to tackle advanced robotics challenges and contribute to the field of Physical AI and humanoid robotics.

## Exercises

1. **Implementation Challenge**: Build a complete conversational robot system using the architecture provided
2. **Performance Optimization**: Optimize the system for real-time performance constraints
3. **Safety Enhancement**: Implement additional safety features and validation
4. **User Study**: Conduct user interaction studies to evaluate system usability