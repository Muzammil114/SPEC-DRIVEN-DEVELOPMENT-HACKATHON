---
sidebar_position: 3
---

# Voice-to-Action Using OpenAI Whisper

## Learning Objectives
- Understand speech recognition and natural language processing for robotics
- Learn to implement OpenAI Whisper for voice command processing
- Create voice command interpretation systems
- Integrate voice processing with robot action planning
- Implement robust voice command validation and error handling

## Introduction to Voice-to-Action Systems

Voice-to-action systems enable robots to receive and interpret spoken commands, transforming natural language into executable actions. This technology is crucial for making robots accessible to non-technical users and enabling more natural human-robot interaction.

### Key Components of Voice-to-Action Systems

1. **Speech Recognition**: Converting audio to text
2. **Natural Language Understanding**: Interpreting the meaning of commands
3. **Intent Classification**: Determining what action to perform
4. **Entity Extraction**: Identifying objects, locations, and parameters
5. **Action Mapping**: Converting commands to robot actions
6. **Feedback Generation**: Providing confirmation to the user

### Voice Command Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Human User    │    │   VLA System    │    │  Robot Actions  │
│                 │    │                 │    │                 │
│   "Go to the   │───▶│ 1. Speech       │───▶│ 4. Action       │
│   kitchen and   │    │    Recognition  │    │    Execution    │
│   bring me the  │    │                 │    │                 │
│   red cup"      │    │ 2. Intent       │    │                 │
│                 │    │    Classification│    │                 │
└─────────────────┘    │                 │    └─────────────────┘
                       │ 3. Entity       │
                       │    Extraction    │
                       └─────────────────┘
```

## OpenAI Whisper for Speech Recognition

### Introduction to Whisper

OpenAI's Whisper is a robust automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data. It offers several advantages for robotics applications:

- **Multilingual Support**: Works with multiple languages
- **Robustness**: Handles various accents, background noise, and audio quality
- **Open Source**: Available for research and commercial use
- **Real-time Capabilities**: Can be optimized for real-time processing

### Whisper Model Variants

Whisper comes in several sizes with different performance characteristics:

| Model | Size | Required VRAM | Relative Speed |
|-------|------|---------------|----------------|
| tiny  | 74 MB | ~1 GB | 32x |
| base  | 145 MB | ~1 GB | 16x |
| small | 465 MB | ~2 GB | 6x |
| medium | 1.5 GB | ~5 GB | 2x |
| large | 3.0 GB | ~10 GB | 1x |

For robotics applications, the choice depends on computational resources and real-time requirements.

### Installing and Setting Up Whisper

```bash
pip install openai-whisper
# Additional dependencies for audio processing
pip install torch torchaudio
```

### Basic Whisper Implementation

```python
import whisper
import torch
import numpy as np
import pyaudio
import wave
import threading
import queue
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class VoiceCommand:
    """Represents a recognized voice command"""
    text: str
    confidence: float
    timestamp: float
    intent: Optional[str] = None
    entities: Optional[Dict] = None
    action: Optional[str] = None

class WhisperSpeechRecognizer:
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Whisper speech recognizer

        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = whisper.load_model(model_size, device=device)
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.audio_queue = queue.Queue()

        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = self.sample_rate

        self.is_listening = False
        self.audio_thread = None

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text using Whisper"""
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Transcribe using Whisper
        result = self.model.transcribe(
            audio_data,
            language="en",
            temperature=0.0,  # Deterministic output
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        return result["text"].strip()

    def transcribe_file(self, audio_file_path: str) -> str:
        """Transcribe audio from file"""
        result = self.model.transcribe(audio_file_path)
        return result["text"].strip()

    def record_audio(self, duration: float = 5.0) -> np.ndarray:
        """Record audio for specified duration"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        return audio_array

    def start_continuous_listening(self, callback_func=None):
        """Start continuous audio listening in a separate thread"""
        self.is_listening = True
        self.audio_thread = threading.Thread(
            target=self._continuous_recognition,
            args=(callback_func,)
        )
        self.audio_thread.start()

    def stop_listening(self):
        """Stop continuous audio listening"""
        self.is_listening = False
        if self.audio_thread:
            self.audio_thread.join()

    def _continuous_recognition(self, callback_func):
        """Internal method for continuous audio recognition"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        # Buffer to hold audio chunks
        audio_buffer = []
        silence_threshold = 500  # Adjust based on your microphone
        max_buffer_size = int(self.rate * 2)  # 2 seconds of audio

        while self.is_listening:
            data = stream.read(self.chunk)
            audio_array = np.frombuffer(data, dtype=np.int16)

            # Check if this is "loud" enough to be speech (simple VAD)
            if np.abs(audio_array).mean() > silence_threshold:
                audio_buffer.extend(audio_array)
            else:
                # If we have accumulated audio and it's been quiet for a while
                if len(audio_buffer) > int(self.rate * 0.5):  # At least 0.5 seconds
                    # Convert buffer to numpy array
                    full_audio = np.array(audio_buffer, dtype=np.float32) / 32768.0

                    # Transcribe
                    try:
                        text = self.transcribe_audio(full_audio)
                        if text and len(text.strip()) > 3:  # Filter out short/empty results
                            if callback_func:
                                command = VoiceCommand(
                                    text=text,
                                    confidence=0.9,  # Placeholder - Whisper doesn't provide confidence
                                    timestamp=time.time()
                                )
                                callback_func(command)
                    except Exception as e:
                        print(f"Error during transcription: {e}")

                    # Clear buffer
                    audio_buffer = []

        stream.stop_stream()
        stream.close()
        p.terminate()
```

## Voice Command Processing Pipeline

### Natural Language Understanding for Robotics

```python
import re
from typing import List, Tuple, Dict
import spacy
from dataclasses import dataclass

# Load spaCy model for NLP processing
# python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class ParsedCommand:
    """Parsed representation of a voice command"""
    intent: str
    entities: Dict[str, List[str]]
    confidence: float
    raw_text: str

class VoiceCommandParser:
    def __init__(self):
        self.intent_patterns = {
            'navigation': [
                r'go to (the )?(?P<location>\w+)',
                r'move to (the )?(?P<location>\w+)',
                r'go (to )?(the )?(?P<location>\w+)',
                r'bring me to (the )?(?P<location>\w+)',
                r'navigate to (the )?(?P<location>\w+)'
            ],
            'object_interaction': [
                r'bring me (the )?(?P<object>[\w\s]+)',
                r'get (the )?(?P<object>[\w\s]+)',
                r'pick up (the )?(?P<object>[\w\s]+)',
                r'grab (the )?(?P<object>[\w\s]+)',
                r'hand me (the )?(?P<object>[\w\s]+)'
            ],
            'manipulation': [
                r'pick up (the )?(?P<object>[\w\s]+)',
                r'put (the )?(?P<object>[\w\s]+) (on|in) (the )?(?P<destination>\w+)',
                r'place (the )?(?P<object>[\w\s]+) (on|in) (the )?(?P<destination>\w+)',
                r'move (the )?(?P<object>[\w\s]+) (to|on|in) (the )?(?P<destination>\w+)'
            ],
            'status_request': [
                r'where are you',
                r'what are you doing',
                r'stop',
                r'wait',
                r'help'
            ]
        }

    def parse_command(self, text: str) -> ParsedCommand:
        """Parse a voice command to extract intent and entities"""
        text_lower = text.lower().strip()

        # First, try pattern matching
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    entities = match.groupdict()
                    return ParsedCommand(
                        intent=intent,
                        entities=entities,
                        confidence=0.8,  # High confidence for pattern matches
                        raw_text=text
                    )

        # If pattern matching fails, use NLP processing
        if nlp:
            return self._parse_with_nlp(text)

        # Fallback: return as unknown with low confidence
        return ParsedCommand(
            intent='unknown',
            entities={},
            confidence=0.3,
            raw_text=text
        )

    def _parse_with_nlp(self, text: str) -> ParsedCommand:
        """Parse command using spaCy NLP"""
        doc = nlp(text)

        # Extract named entities and dependencies
        entities = {}
        intent = 'unknown'
        confidence = 0.5

        # Look for action verbs that indicate intent
        for token in doc:
            if token.pos_ == 'VERB':
                if token.lemma_ in ['go', 'move', 'navigate']:
                    intent = 'navigation'
                elif token.lemma_ in ['bring', 'get', 'pick', 'grab', 'hand']:
                    intent = 'object_interaction'
                elif token.lemma_ in ['put', 'place', 'move']:
                    intent = 'manipulation'

        # Extract entities based on dependencies
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geopolitical entity, location, facility
                entities['location'] = entities.get('location', []) + [ent.text]
            elif ent.label_ in ['OBJECT', 'PRODUCT']:  # Note: spaCy doesn't have OBJECT by default
                entities['object'] = entities.get('object', []) + [ent.text]

        # Extract objects based on noun chunks
        for chunk in doc.noun_chunks:
            if any(token.pos_ in ['NOUN', 'PROPN'] for token in chunk):
                # Determine if it's a location or object based on context
                if any(loc_word in chunk.text.lower() for loc_word in ['kitchen', 'bedroom', 'office', 'room', 'table', 'shelf']):
                    entities['location'] = entities.get('location', []) + [chunk.text]
                else:
                    entities['object'] = entities.get('object', []) + [chunk.text]

        return ParsedCommand(
            intent=intent,
            entities=entities,
            confidence=confidence,
            raw_text=text
        )

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {
            'locations': [],
            'objects': [],
            'quantities': [],
            'descriptors': []
        }

        # Simple regex-based extraction
        location_patterns = [
            r'kitchen', r'bedroom', r'office', r'living room', r'room',
            r'table', r'shelf', r'cabinet', r'counter'
        ]

        object_patterns = [
            r'cup', r'glass', r'bottle', r'book', r'phone', r'computer',
            r'food', r'water', r'item', r'thing', r'object'
        ]

        for pattern in location_patterns:
            matches = re.findall(pattern, text.lower())
            entities['locations'].extend(matches)

        for pattern in object_patterns:
            matches = re.findall(pattern, text.lower())
            entities['objects'].extend(matches)

        # Extract quantities
        quantity_matches = re.findall(r'(\d+)', text)
        entities['quantities'] = [int(q) for q in quantity_matches]

        # Extract color descriptors
        color_patterns = [r'red', r'blue', r'green', r'yellow', r'black', r'white']
        for pattern in color_patterns:
            matches = re.findall(pattern, text.lower())
            entities['descriptors'].extend(matches)

        return entities
```

## Intent Classification and Action Mapping

### Intent Classification System

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os

class IntentClassifier:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.pipeline = None
        self.intents = [
            'navigation', 'object_interaction', 'manipulation',
            'status_request', 'stop', 'help', 'unknown'
        ]

        if model_path and os.path.exists(model_path):
            self.load_model()
        else:
            self.train_default_model()

    def train_default_model(self):
        """Train a default intent classification model"""
        # Training data - in practice, this would be much more extensive
        training_data = [
            # Navigation examples
            ("go to kitchen", "navigation"),
            ("move to bedroom", "navigation"),
            ("navigate to office", "navigation"),
            ("go to living room", "navigation"),
            ("move to table", "navigation"),

            # Object interaction examples
            ("bring me water", "object_interaction"),
            ("get the red cup", "object_interaction"),
            ("pick up the book", "object_interaction"),
            ("grab the phone", "object_interaction"),
            ("hand me the bottle", "object_interaction"),

            # Manipulation examples
            ("put the cup on table", "manipulation"),
            ("place book on shelf", "manipulation"),
            ("move object to counter", "manipulation"),

            # Status requests
            ("where are you", "status_request"),
            ("what are you doing", "status_request"),
            ("report status", "status_request"),

            # Stop commands
            ("stop", "stop"),
            ("halt", "stop"),
            ("pause", "stop"),

            # Help
            ("help", "help"),
            ("what can you do", "help"),
            ("how to use", "help")
        ]

        texts, labels = zip(*training_data)

        # Create and train pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
            ('classifier', MultinomialNB())
        ])

        self.pipeline.fit(texts, labels)

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of a text command"""
        if not self.pipeline:
            return 'unknown', 0.0

        # Predict intent
        predicted_intent = self.pipeline.predict([text])[0]

        # Get confidence scores
        confidence_scores = self.pipeline.predict_proba([text])[0]
        max_confidence = max(confidence_scores)

        return predicted_intent, max_confidence

    def save_model(self, path: str = None):
        """Save the trained model"""
        if not path:
            path = self.model_path
        if path:
            with open(path, 'wb') as f:
                pickle.dump(self.pipeline, f)

    def load_model(self, path: str = None):
        """Load a pre-trained model"""
        if not path:
            path = self.model_path
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                self.pipeline = pickle.load(f)
```

### Action Mapping System

```python
from enum import Enum
from typing import Any, Dict, List
import json

class RobotAction(Enum):
    """Enumeration of possible robot actions"""
    NAVIGATE_TO_LOCATION = "navigate_to_location"
    PICK_UP_OBJECT = "pick_up_object"
    PLACE_OBJECT = "place_object"
    GRASP_OBJECT = "grasp_object"
    RELEASE_OBJECT = "release_object"
    REPORT_STATUS = "report_status"
    STOP = "stop"
    WAIT = "wait"
    FOLLOW_PERSON = "follow_person"
    AVOID_OBSTACLE = "avoid_obstacle"

@dataclass
class ActionPlan:
    """Plan of actions to execute"""
    actions: List[Dict[str, Any]]
    confidence: float
    description: str

class VoiceCommandToActionMapper:
    def __init__(self):
        self.intent_to_actions = {
            'navigation': [RobotAction.NAVIGATE_TO_LOCATION],
            'object_interaction': [RobotAction.NAVIGATE_TO_LOCATION, RobotAction.GRASP_OBJECT, RobotAction.NAVIGATE_TO_LOCATION],
            'manipulation': [RobotAction.GRASP_OBJECT, RobotAction.NAVIGATE_TO_LOCATION, RobotAction.RELEASE_OBJECT],
            'status_request': [RobotAction.REPORT_STATUS],
            'stop': [RobotAction.STOP],
            'help': [RobotAction.REPORT_STATUS]
        }

    def map_command_to_action(self, parsed_command: ParsedCommand) -> ActionPlan:
        """Map a parsed command to a sequence of robot actions"""
        intent = parsed_command.intent
        entities = parsed_command.entities

        if intent == 'navigation':
            return self._create_navigation_plan(entities, parsed_command.confidence)
        elif intent == 'object_interaction':
            return self._create_object_interaction_plan(entities, parsed_command.confidence)
        elif intent == 'manipulation':
            return self._create_manipulation_plan(entities, parsed_command.confidence)
        elif intent == 'status_request':
            return self._create_status_plan(parsed_command.confidence)
        elif intent == 'stop':
            return self._create_stop_plan(parsed_command.confidence)
        elif intent == 'help':
            return self._create_help_plan(parsed_command.confidence)
        else:
            return self._create_unknown_plan(parsed_command.confidence)

    def _create_navigation_plan(self, entities: Dict, confidence: float) -> ActionPlan:
        """Create a navigation action plan"""
        location = entities.get('location', ['unknown'])[0] if entities.get('location') else 'unknown'

        actions = [{
            'action': RobotAction.NAVIGATE_TO_LOCATION.value,
            'parameters': {
                'target_location': location
            }
        }]

        return ActionPlan(
            actions=actions,
            confidence=confidence,
            description=f"Navigate to {location}"
        )

    def _create_object_interaction_plan(self, entities: Dict, confidence: float) -> ActionPlan:
        """Create an object interaction action plan"""
        obj = entities.get('object', ['unknown'])[0] if entities.get('object') else 'unknown'

        actions = [
            {
                'action': RobotAction.NAVIGATE_TO_LOCATION.value,
                'parameters': {
                    'target_location': 'object_location'  # This would be determined by perception
                }
            },
            {
                'action': RobotAction.GRASP_OBJECT.value,
                'parameters': {
                    'object_type': obj
                }
            },
            {
                'action': RobotAction.NAVIGATE_TO_LOCATION.value,
                'parameters': {
                    'target_location': 'user_location'
                }
            }
        ]

        return ActionPlan(
            actions=actions,
            confidence=confidence,
            description=f"Get {obj} and bring it to user"
        )

    def _create_manipulation_plan(self, entities: Dict, confidence: float) -> ActionPlan:
        """Create a manipulation action plan"""
        obj = entities.get('object', ['unknown'])[0] if entities.get('object') else 'unknown'
        dest = entities.get('destination', ['unknown'])[0] if entities.get('destination') else 'unknown'

        actions = [
            {
                'action': RobotAction.GRASP_OBJECT.value,
                'parameters': {
                    'object_type': obj
                }
            },
            {
                'action': RobotAction.NAVIGATE_TO_LOCATION.value,
                'parameters': {
                    'target_location': dest
                }
            },
            {
                'action': RobotAction.RELEASE_OBJECT.value,
                'parameters': {}
            }
        ]

        return ActionPlan(
            actions=actions,
            confidence=confidence,
            description=f"Move {obj} to {dest}"
        )

    def _create_status_plan(self, confidence: float) -> ActionPlan:
        """Create a status reporting action plan"""
        actions = [{
            'action': RobotAction.REPORT_STATUS.value,
            'parameters': {}
        }]

        return ActionPlan(
            actions=actions,
            confidence=confidence,
            description="Report current status"
        )

    def _create_stop_plan(self, confidence: float) -> ActionPlan:
        """Create a stop action plan"""
        actions = [{
            'action': RobotAction.STOP.value,
            'parameters': {}
        }]

        return ActionPlan(
            actions=actions,
            confidence=confidence,
            description="Stop current action"
        )

    def _create_help_plan(self, confidence: float) -> ActionPlan:
        """Create a help action plan"""
        actions = [{
            'action': RobotAction.REPORT_STATUS.value,
            'parameters': {}
        }]

        return ActionPlan(
            actions=actions,
            confidence=confidence,
            description="Provide help information"
        )

    def _create_unknown_plan(self, confidence: float) -> ActionPlan:
        """Create a plan for unknown commands"""
        actions = []

        return ActionPlan(
            actions=actions,
            confidence=confidence,
            description="Unknown command"
        )
```

## Integration with ROS 2

### Voice Command Processing Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Initialize voice processing components
        self.speech_recognizer = WhisperSpeechRecognizer(model_size="base")
        self.command_parser = VoiceCommandParser()
        self.intent_classifier = IntentClassifier()
        self.action_mapper = VoiceCommandToActionMapper()

        # ROS 2 interfaces
        self.command_pub = self.create_publisher(String, 'robot_command', 10)
        self.feedback_pub = self.create_publisher(String, 'voice_feedback', 10)
        self.speech_sub = self.create_subscription(
            String, 'speech_recognized', self.speech_callback, 10
        )

        # Action clients for robot navigation and manipulation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manip_client = ActionClient(self, ManipulationAction, 'manipulation')

        # Internal state
        self.current_action_plan = None
        self.is_executing = False

        # Start voice recognition
        self.get_logger().info("Voice Command Node initialized")

    def start_voice_recognition(self):
        """Start continuous voice recognition"""
        def callback(command: VoiceCommand):
            self.process_voice_command(command)

        self.speech_recognizer.start_continuous_listening(callback)

    def speech_callback(self, msg: String):
        """Handle speech recognition results from external source"""
        self.get_logger().info(f"Received speech: {msg.data}")
        command = VoiceCommand(
            text=msg.data,
            confidence=0.8,  # Assume external system provides confidence
            timestamp=time.time()
        )
        self.process_voice_command(command)

    def process_voice_command(self, command: VoiceCommand):
        """Process a recognized voice command"""
        self.get_logger().info(f"Processing command: {command.text}")

        # Publish feedback
        feedback_msg = String()
        feedback_msg.data = f"Heard: {command.text}"
        self.feedback_pub.publish(feedback_msg)

        try:
            # Parse the command
            parsed_command = self.command_parser.parse_command(command.text)
            self.get_logger().info(f"Parsed intent: {parsed_command.intent}")

            # Classify intent with confidence
            intent, intent_conf = self.intent_classifier.classify_intent(command.text)
            parsed_command.intent = intent
            parsed_command.confidence = max(parsed_command.confidence, intent_conf)

            # Map to actions
            action_plan = self.action_mapper.map_command_to_action(parsed_command)
            self.get_logger().info(f"Generated action plan: {action_plan.description}")

            # Validate confidence
            if action_plan.confidence < 0.5:
                self.get_logger().warn(f"Low confidence command: {action_plan.confidence}")
                feedback_msg.data = "Command not understood clearly, please repeat"
                self.feedback_pub.publish(feedback_msg)
                return

            # Execute action plan
            self.execute_action_plan(action_plan)

        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            feedback_msg.data = "Error processing command"
            self.feedback_pub.publish(feedback_msg)

    def execute_action_plan(self, action_plan: ActionPlan):
        """Execute a sequence of robot actions"""
        if self.is_executing:
            self.get_logger().warn("Already executing, ignoring new command")
            return

        self.is_executing = True
        self.current_action_plan = action_plan

        try:
            for action_step in action_plan.actions:
                if not self.is_executing:  # Check if stop was requested
                    break

                action_type = action_step['action']
                params = action_step['parameters']

                self.get_logger().info(f"Executing action: {action_type}")

                if action_type == RobotAction.NAVIGATE_TO_LOCATION.value:
                    success = self.execute_navigation_action(params)
                elif action_type == RobotAction.GRASP_OBJECT.value:
                    success = self.execute_manipulation_action(params)
                elif action_type == RobotAction.RELEASE_OBJECT.value:
                    success = self.execute_manipulation_action(params)
                elif action_type == RobotAction.REPORT_STATUS.value:
                    success = self.execute_status_action(params)
                elif action_type == RobotAction.STOP.value:
                    success = self.execute_stop_action(params)
                else:
                    self.get_logger().warn(f"Unknown action type: {action_type}")
                    success = False

                if not success:
                    self.get_logger().error(f"Action failed: {action_type}")
                    break

            # Publish completion feedback
            feedback_msg = String()
            feedback_msg.data = f"Completed: {action_plan.description}"
            self.feedback_pub.publish(feedback_msg)

        except Exception as e:
            self.get_logger().error(f"Error executing action plan: {e}")
            feedback_msg = String()
            feedback_msg.data = "Error executing command"
            self.feedback_pub.publish(feedback_msg)
        finally:
            self.is_executing = False
            self.current_action_plan = None

    def execute_navigation_action(self, params: Dict) -> bool:
        """Execute navigation action"""
        target_location = params.get('target_location', 'unknown')

        # In a real implementation, you would look up the actual coordinates
        # For now, we'll use a simple approach
        if target_location == 'kitchen':
            target_pose = self.get_location_pose('kitchen')
        elif target_location == 'bedroom':
            target_pose = self.get_location_pose('bedroom')
        else:
            # Try to find the location using perception
            target_pose = self.find_location_pose(target_location)

        if target_pose:
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = target_pose

            self.nav_client.wait_for_server()
            future = self.nav_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)

            result = future.result()
            return result.status == GoalStatus.STATUS_SUCCEEDED
        else:
            self.get_logger().error(f"Could not find location: {target_location}")
            return False

    def execute_manipulation_action(self, params: Dict) -> bool:
        """Execute manipulation action"""
        object_type = params.get('object_type', 'unknown')

        goal_msg = ManipulationAction.Goal()
        goal_msg.object_type = object_type
        goal_msg.action_type = params.get('action_type', 'grasp')

        self.manip_client.wait_for_server()
        future = self.manip_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        return result.status == GoalStatus.STATUS_SUCCEEDED

    def execute_status_action(self, params: Dict) -> bool:
        """Execute status reporting action"""
        # Report current status
        status_msg = String()
        status_msg.data = f"Robot is at {self.get_current_pose()}, battery: 85%"
        self.command_pub.publish(status_msg)
        return True

    def execute_stop_action(self, params: Dict) -> bool:
        """Execute stop action"""
        self.is_executing = False
        return True

    def get_location_pose(self, location_name: str) -> PoseStamped:
        """Get pre-defined pose for a location"""
        # This would be populated with actual location coordinates
        location_poses = {
            'kitchen': PoseStamped(),
            'bedroom': PoseStamped(),
            'office': PoseStamped()
        }
        return location_poses.get(location_name)

    def find_location_pose(self, location_name: str) -> PoseStamped:
        """Find pose for a location using perception"""
        # This would use the multimodal perception system
        # to locate the requested area in the environment
        return None

    def get_current_pose(self) -> str:
        """Get current robot pose"""
        # This would interface with localization system
        return "unknown_pose"

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()

    # Start voice recognition in a separate thread
    voice_thread = threading.Thread(target=node.start_voice_recognition)
    voice_thread.start()

    try:
        # Use multi-threaded executor to handle both ROS callbacks and voice processing
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.speech_recognizer.stop_listening()
        node.destroy_node()
        rclpy.shutdown()
        voice_thread.join()

if __name__ == '__main__':
    main()
```

## Error Handling and Validation

### Robust Voice Command Processing

```python
import asyncio
import time
from typing import Callable, Optional

class RobustVoiceCommandProcessor:
    def __init__(self, speech_recognizer: WhisperSpeechRecognizer):
        self.speech_recognizer = speech_recognizer
        self.command_parser = VoiceCommandParser()
        self.intent_classifier = IntentClassifier()
        self.action_mapper = VoiceCommandToActionMapper()

        # Validation parameters
        self.min_confidence = 0.5
        self.max_command_length = 100
        self.min_command_length = 3
        self.command_history = []
        self.max_history = 10

    def validate_command(self, command: VoiceCommand) -> Tuple[bool, str]:
        """Validate a voice command for execution"""
        errors = []

        # Check command length
        if len(command.text) < self.min_command_length:
            errors.append("Command too short")
        elif len(command.text) > self.max_command_length:
            errors.append("Command too long")

        # Check confidence
        if command.confidence < self.min_confidence:
            errors.append(f"Low confidence: {command.confidence:.2f}")

        # Check for profanity or inappropriate content (basic check)
        inappropriate_words = ['shut down', 'turn off', 'kill', 'destroy']
        if any(word in command.text.lower() for word in inappropriate_words):
            errors.append("Command contains inappropriate content")

        # Check for repeated commands (to prevent spam)
        recent_commands = self.command_history[-3:]  # Check last 3 commands
        if command.text in recent_commands:
            errors.append("Command repeated too frequently")

        is_valid = len(errors) == 0
        error_msg = "; ".join(errors) if errors else ""

        return is_valid, error_msg

    def process_command_with_validation(self, command: VoiceCommand) -> Optional[ParsedCommand]:
        """Process a command with full validation"""
        # Validate command
        is_valid, error_msg = self.validate_command(command)

        if not is_valid:
            print(f"Command validation failed: {error_msg}")
            return None

        # Add to history
        self.command_history.append(command.text)
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)

        try:
            # Parse the command
            parsed_command = self.command_parser.parse_command(command.text)

            # Classify intent
            intent, intent_conf = self.intent_classifier.classify_intent(command.text)
            parsed_command.intent = intent
            parsed_command.confidence = max(parsed_command.confidence, intent_conf)

            # Validate intent confidence
            if parsed_command.confidence < self.min_confidence:
                print(f"Intent confidence too low: {parsed_command.confidence:.2f}")
                return None

            return parsed_command

        except Exception as e:
            print(f"Error processing command: {e}")
            return None

    def request_clarification(self, command: VoiceCommand) -> str:
        """Request clarification for ambiguous commands"""
        # This would typically involve asking the user for clarification
        # For now, return a default clarification request

        # Analyze the command to determine what clarification is needed
        if command.text.lower().strip() in ['go', 'move', 'pick', 'get']:
            return "Where should I go to?" if command.text.lower() in ['go', 'move'] else "What should I pick up?"
        elif 'the' not in command.text.lower():
            # Command might be missing specific object reference
            return "Which specific object do you mean?"
        else:
            return "Could you please repeat that more clearly?"

    def handle_ambiguous_command(self, command: VoiceCommand) -> Optional[ParsedCommand]:
        """Handle commands that are ambiguous or unclear"""
        # Try to disambiguate using context
        if len(self.command_history) > 1:
            previous_command = self.command_history[-2]  # Previous command
            # Use context from previous command to clarify current one
            # This is a simplified approach
            if 'kitchen' in previous_command and 'water' in command.text.lower():
                # User probably wants water from kitchen
                disambiguated_text = f"bring me water from the kitchen"
                disambiguated_command = VoiceCommand(
                    text=disambiguated_text,
                    confidence=command.confidence * 0.8,  # Lower confidence due to assumption
                    timestamp=command.timestamp
                )
                return self.process_command_with_validation(disambiguated_command)

        # If context doesn't help, request clarification
        clarification = self.request_clarification(command)
        print(f"Clarification needed: {clarification}")
        return None
```

## Performance Optimization

### Efficient Voice Processing Pipeline

```python
import asyncio
import concurrent.futures
from typing import List, Callable
import time

class EfficientVoiceProcessor:
    def __init__(self, max_workers=2):
        self.speech_recognizer = WhisperSpeechRecognizer(model_size="base")
        self.command_parser = VoiceCommandParser()
        self.intent_classifier = IntentClassifier()
        self.action_mapper = VoiceCommandToActionMapper()

        # Thread pool for CPU-intensive tasks
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Caching for frequently used commands
        self.command_cache = {}
        self.max_cache_size = 100

    async def process_audio_batch(self, audio_segments: List[np.ndarray]) -> List[ParsedCommand]:
        """Process multiple audio segments concurrently"""
        loop = asyncio.get_event_loop()

        # Submit transcription tasks
        transcription_tasks = [
            loop.run_in_executor(self.executor, self.speech_recognizer.transcribe_audio, segment)
            for segment in audio_segments
        ]

        # Wait for transcriptions
        transcriptions = await asyncio.gather(*transcription_tasks)

        # Process transcriptions
        results = []
        for text in transcriptions:
            if text.strip():
                parsed = await self.process_text_command(text)
                if parsed:
                    results.append(parsed)

        return results

    async def process_text_command(self, text: str) -> Optional[ParsedCommand]:
        """Process a text command asynchronously"""
        loop = asyncio.get_event_loop()

        # Check cache first
        if text in self.command_cache:
            return self.command_cache[text]

        # Parse command
        parsed_command = await loop.run_in_executor(
            self.executor,
            self.command_parser.parse_command,
            text
        )

        # Classify intent
        intent, confidence = await loop.run_in_executor(
            self.executor,
            self.intent_classifier.classify_intent,
            text
        )

        parsed_command.intent = intent
        parsed_command.confidence = max(parsed_command.confidence, confidence)

        # Add to cache
        if len(self.command_cache) < self.max_cache_size:
            self.command_cache[text] = parsed_command

        return parsed_command

    def optimize_for_real_time(self) -> None:
        """Optimize processing for real-time performance"""
        # Use smaller model for real-time processing
        self.speech_recognizer = WhisperSpeechRecognizer(model_size="tiny")

        # Optimize NLP processing
        if nlp:
            # Disable unnecessary components
            nlp.disable_pipes('parser', 'ner')  # Only keep tokenizer and tagger if needed

class VoiceActivityDetector:
    """Simple voice activity detection to reduce unnecessary processing"""
    def __init__(self, threshold=500, silence_frames=10):
        self.threshold = threshold
        self.silence_frames = silence_frames
        self.silence_count = 0
        self.is_speaking = False

    def is_voice_present(self, audio_chunk: np.ndarray) -> bool:
        """Detect if voice is present in audio chunk"""
        energy = np.abs(audio_chunk).mean()

        if energy > self.threshold:
            self.silence_count = 0
            self.is_speaking = True
        else:
            self.silence_count += 1
            if self.silence_count > self.silence_frames:
                self.is_speaking = False

        return self.is_speaking
```

## Summary

Voice-to-action systems using OpenAI Whisper enable robots to understand and respond to natural language commands:

- **Speech Recognition**: Converting audio to text using Whisper
- **Natural Language Understanding**: Interpreting command intent and entities
- **Action Mapping**: Converting commands to executable robot actions
- **Validation**: Ensuring commands are appropriate and clear
- **Integration**: Connecting voice processing with ROS 2 robotics systems

These systems make robots more accessible and intuitive to use, enabling natural human-robot interaction.

## Exercises

1. Implement a voice command system using Whisper for speech recognition
2. Create an intent classification model for robotics commands
3. Develop a command validation system with error handling
4. Integrate voice processing with a ROS 2 navigation system