---
sidebar_position: 4
---

# Natural Language Understanding for Robotics

## Learning Objectives
- Understand the fundamentals of natural language processing for robotics
- Learn to implement semantic parsing for robot commands
- Develop context-aware language understanding systems
- Create dialogue management for human-robot interaction
- Implement error handling and clarification strategies

## Introduction to Natural Language Understanding in Robotics

Natural Language Understanding (NLU) in robotics involves interpreting human language commands and converting them into structured representations that robots can execute. This is fundamentally different from general NLU because it must account for the physical world, spatial relationships, and action execution.

### Challenges in Robotic NLU

1. **Physical Grounding**: Connecting language to physical objects and locations
2. **Spatial Reasoning**: Understanding spatial relationships and navigation commands
3. **Action Interpretation**: Converting high-level commands to executable actions
4. **Context Awareness**: Understanding commands in the context of the current situation
5. **Ambiguity Resolution**: Handling ambiguous or underspecified commands
6. **Real-time Processing**: Operating within practical time constraints

### Robotic NLU Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Human Input   │    │   NLU System    │    │  Robot Actions  │
│                 │    │                 │    │                 │
│ "Go to the     │───▶│ 1. Tokenization │───▶│ 4. Action       │
│   kitchen and   │    │    & Parsing    │    │    Execution    │
│   bring me the  │    │                 │    │                 │
│   red cup"      │    │ 2. Semantic     │    │                 │
│                 │    │    Analysis     │    │                 │
└─────────────────┘    │                 │    └─────────────────┘
                       │ 3. Context      │
                       │    Integration   │
                       └─────────────────┘
```

## Semantic Parsing for Robot Commands

### Understanding Command Structure

Robot commands typically follow predictable patterns that can be parsed semantically:

- **Action**: What to do (navigate, pick, place, etc.)
- **Object**: What to act upon (cup, book, person, etc.)
- **Location**: Where to perform the action (kitchen, table, etc.)
- **Modifiers**: How to perform the action (carefully, quickly, etc.)

### Semantic Parser Implementation

```python
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RobotActionType(Enum):
    NAVIGATE = "navigate"
    PICK_UP = "pick_up"
    PLACE = "place"
    FOLLOW = "follow"
    FIND = "find"
    REPORT = "report"
    STOP = "stop"
    WAIT = "wait"

@dataclass
class SemanticEntity:
    """Represents an entity extracted from natural language"""
    entity_type: str  # 'object', 'location', 'action', 'modifier', etc.
    value: str
    confidence: float = 1.0
    position: Tuple[int, int] = None  # Start and end positions in original text

@dataclass
class ParsedCommand:
    """Represents a parsed robot command"""
    action: RobotActionType
    entities: List[SemanticEntity]
    confidence: float
    raw_text: str
    structured_query: Dict

class SemanticParser:
    def __init__(self):
        # Action keywords and their corresponding RobotActionType
        self.action_keywords = {
            'navigate': RobotActionType.NAVIGATE,
            'go': RobotActionType.NAVIGATE,
            'move': RobotActionType.NAVIGATE,
            'travel': RobotActionType.NAVIGATE,
            'walk': RobotActionType.NAVIGATE,
            'pick': RobotActionType.PICK_UP,
            'get': RobotActionType.PICK_UP,
            'take': RobotActionType.PICK_UP,
            'grab': RobotActionType.PICK_UP,
            'collect': RobotActionType.PICK_UP,
            'place': RobotActionType.PLACE,
            'put': RobotActionType.PLACE,
            'set': RobotActionType.PLACE,
            'drop': RobotActionType.PLACE,
            'follow': RobotActionType.FOLLOW,
            'chase': RobotActionType.FOLLOW,
            'find': RobotActionType.FIND,
            'locate': RobotActionType.FIND,
            'look': RobotActionType.FIND,
            'search': RobotActionType.FIND,
            'report': RobotActionType.REPORT,
            'tell': RobotActionType.REPORT,
            'show': RobotActionType.REPORT,
            'stop': RobotActionType.STOP,
            'halt': RobotActionType.STOP,
            'wait': RobotActionType.WAIT,
            'pause': RobotActionType.WAIT
        }

        # Location keywords
        self.location_keywords = [
            'kitchen', 'bedroom', 'office', 'living room', 'bathroom',
            'dining room', 'hallway', 'garage', 'garden', 'patio',
            'table', 'chair', 'couch', 'bed', 'desk', 'shelf',
            'counter', 'refrigerator', 'microwave', 'cabinet'
        ]

        # Object keywords
        self.object_keywords = [
            'cup', 'bottle', 'glass', 'plate', 'bowl', 'fork', 'spoon',
            'knife', 'book', 'phone', 'laptop', 'computer', 'keys',
            'wallet', 'glasses', 'hat', 'coat', 'bag', 'box',
            'food', 'drink', 'water', 'coffee', 'tea', 'person',
            'human', 'robot', 'pet', 'cat', 'dog'
        ]

        # Color keywords
        self.color_keywords = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple',
            'pink', 'brown', 'black', 'white', 'gray', 'grey'
        ]

        # Size keywords
        self.size_keywords = [
            'big', 'large', 'small', 'tiny', 'huge', 'enormous',
            'little', 'mini', 'massive', 'giant'
        ]

        # Pattern-based rules for complex commands
        self.pattern_rules = [
            # Navigation patterns
            (r'go to (the )?(?P<location>\w+)', self._parse_navigation),
            (r'move to (the )?(?P<location>\w+)', self._parse_navigation),
            (r'navigate to (the )?(?P<location>\w+)', self._parse_navigation),
            (r'bring me to (the )?(?P<location>\w+)', self._parse_navigation),

            # Object interaction patterns
            (r'bring me (the )?(?P<descriptor>(?P<color>\w+ )?(?P<object>\w+))', self._parse_object_interaction),
            (r'get (the )?(?P<descriptor>(?P<color>\w+ )?(?P<object>\w+))', self._parse_object_interaction),
            (r'pick up (the )?(?P<descriptor>(?P<color>\w+ )?(?P<object>\w+))', self._parse_object_interaction),
            (r'grab (the )?(?P<descriptor>(?P<color>\w+ )?(?P<object>\w+))', self._parse_object_interaction),

            # Place/put patterns
            (r'put (the )?(?P<object>\w+) (in|on|at) (the )?(?P<location>\w+)', self._parse_placement),
            (r'place (the )?(?P<object>\w+) (in|on|at) (the )?(?P<location>\w+)', self._parse_placement),
        ]

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse a natural language command into structured format"""
        text_lower = text.lower().strip()

        # Try pattern-based parsing first
        for pattern, parser_func in self.pattern_rules:
            match = re.search(pattern, text_lower)
            if match:
                entities, action_type = parser_func(match)
                confidence = 0.9  # High confidence for pattern matches
                return ParsedCommand(
                    action=action_type,
                    entities=entities,
                    confidence=confidence,
                    raw_text=text,
                    structured_query=self._create_structured_query(action_type, entities)
                )

        # Fallback to keyword-based parsing
        return self._keyword_based_parse(text_lower, text)

    def _keyword_based_parse(self, text_lower: str, original_text: str) -> Optional[ParsedCommand]:
        """Parse command using keyword matching"""
        entities = []
        action_type = None

        # Extract action
        for keyword, action in self.action_keywords.items():
            if keyword in text_lower:
                action_type = action
                # Add action entity
                entities.append(SemanticEntity(
                    entity_type='action',
                    value=keyword,
                    confidence=0.8
                ))
                break

        if not action_type:
            return None

        # Extract locations
        for location in self.location_keywords:
            if location in text_lower:
                entities.append(SemanticEntity(
                    entity_type='location',
                    value=location,
                    confidence=0.7
                ))

        # Extract objects
        for obj in self.object_keywords:
            if obj in text_lower:
                entities.append(SemanticEntity(
                    entity_type='object',
                    value=obj,
                    confidence=0.7
                ))

        # Extract colors
        for color in self.color_keywords:
            if color in text_lower:
                entities.append(SemanticEntity(
                    entity_type='color',
                    value=color,
                    confidence=0.6
                ))

        # Extract sizes
        for size in self.size_keywords:
            if size in text_lower:
                entities.append(SemanticEntity(
                    entity_type='size',
                    value=size,
                    confidence=0.6
                ))

        confidence = 0.6  # Lower confidence for keyword-based parsing

        return ParsedCommand(
            action=action_type,
            entities=entities,
            confidence=confidence,
            raw_text=original_text,
            structured_query=self._create_structured_query(action_type, entities)
        )

    def _parse_navigation(self, match) -> Tuple[List[SemanticEntity], RobotActionType]:
        """Parse navigation commands"""
        entities = []
        if match.group('location'):
            entities.append(SemanticEntity(
                entity_type='location',
                value=match.group('location'),
                confidence=0.9
            ))
        return entities, RobotActionType.NAVIGATE

    def _parse_object_interaction(self, match) -> Tuple[List[SemanticEntity], RobotActionType]:
        """Parse object interaction commands"""
        entities = []
        if match.group('color'):
            entities.append(SemanticEntity(
                entity_type='color',
                value=match.group('color').strip(),
                confidence=0.8
            ))
        if match.group('object'):
            entities.append(SemanticEntity(
                entity_type='object',
                value=match.group('object'),
                confidence=0.8
            ))
        return entities, RobotActionType.PICK_UP

    def _parse_placement(self, match) -> Tuple[List[SemanticEntity], RobotActionType]:
        """Parse placement commands"""
        entities = []
        if match.group('object'):
            entities.append(SemanticEntity(
                entity_type='object',
                value=match.group('object'),
                confidence=0.8
            ))
        if match.group('location'):
            entities.append(SemanticEntity(
                entity_type='location',
                value=match.group('location'),
                confidence=0.8
            ))
        return entities, RobotActionType.PLACE

    def _create_structured_query(self, action_type: RobotActionType, entities: List[SemanticEntity]) -> Dict:
        """Create a structured query from parsed entities"""
        query = {
            'action': action_type.value,
            'parameters': {}
        }

        for entity in entities:
            if entity.entity_type in ['location', 'object', 'color', 'size']:
                query['parameters'][entity.entity_type] = entity.value

        return query

    def extract_spatial_relationships(self, text: str) -> List[Dict]:
        """Extract spatial relationships from text"""
        spatial_patterns = [
            r'on (the )?(?P<object>\w+)',
            r'in (the )?(?P<object>\w+)',
            r'next to (the )?(?P<object>\w+)',
            r'behind (the )?(?P<object>\w+)',
            r'in front of (the )?(?P<object>\w+)',
            r'above (the )?(?P<object>\w+)',
            r'below (the )?(?P<object>\w+)',
            r'near (the )?(?P<object>\w+)',
            r'beside (the )?(?P<object>\w+)'
        ]

        relationships = []
        for pattern in spatial_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                if match.group('object'):
                    relationships.append({
                        'relationship': pattern.split('(')[0].strip(),
                        'object': match.group('object'),
                        'confidence': 0.8
                    })

        return relationships
```

## Context-Aware Language Understanding

### Maintaining Context for Conversational Robots

Context-aware NLU systems maintain information about the ongoing conversation and environment to better understand ambiguous references and follow-up commands.

```python
from datetime import datetime, timedelta
from typing import Any

class ContextManager:
    def __init__(self):
        self.current_context = {
            'robot_location': 'unknown',
            'user_location': 'unknown',
            'last_action': None,
            'last_action_result': None,
            'current_task': None,
            'conversation_history': [],
            'object_references': {},  # Maps pronouns to specific objects
            'spatial_context': {},    # Maps locations to known objects
            'time': datetime.now()
        }
        self.max_history = 10

    def update_context(self, **kwargs):
        """Update context with new information"""
        for key, value in kwargs.items():
            self.current_context[key] = value

        # Update time
        self.current_context['time'] = datetime.now()

        # Limit conversation history size
        if len(self.current_context['conversation_history']) > self.max_history:
            self.current_context['conversation_history'] = \
                self.current_context['conversation_history'][-self.max_history:]

    def resolve_pronouns(self, text: str) -> str:
        """Resolve pronouns in text using context"""
        resolved_text = text.lower()

        # Replace "it" with last referenced object
        if 'it' in resolved_text and self.current_context['object_references']:
            last_obj = list(self.current_context['object_references'].values())[-1]
            resolved_text = resolved_text.replace('it', last_obj)

        # Replace "there" with last referenced location
        if 'there' in resolved_text and self.current_context['last_action']:
            if 'location' in self.current_context['last_action']:
                location = self.current_context['last_action']['location']
                resolved_text = resolved_text.replace('there', location)

        return resolved_text

    def get_contextual_entities(self) -> Dict[str, Any]:
        """Get entities from current context"""
        return {
            'current_location': self.current_context['robot_location'],
            'user_location': self.current_context['user_location'],
            'last_action': self.current_context['last_action'],
            'current_task': self.current_context['current_task']
        }

    def add_conversation_turn(self, user_input: str, robot_response: str = None):
        """Add a turn to the conversation history"""
        turn = {
            'user_input': user_input,
            'robot_response': robot_response,
            'timestamp': datetime.now()
        }
        self.current_context['conversation_history'].append(turn)

class ContextualSemanticParser(SemanticParser):
    def __init__(self):
        super().__init__()
        self.context_manager = ContextManager()

    def parse_command_with_context(self, text: str) -> Optional[ParsedCommand]:
        """Parse command using contextual information"""
        # Resolve pronouns and contextual references
        resolved_text = self.context_manager.resolve_pronouns(text)

        # Parse the resolved command
        parsed_command = self.parse_command(resolved_text)

        if parsed_command:
            # Enhance with contextual information
            self._enhance_with_context(parsed_command)

        return parsed_command

    def _enhance_with_context(self, command: ParsedCommand):
        """Enhance parsed command with contextual information"""
        # Add current location if navigating but no location specified
        if command.action == RobotActionType.NAVIGATE:
            entities = {e.entity_type: e.value for e in command.entities}
            if 'location' not in entities:
                context_entities = self.context_manager.get_contextual_entities()
                if context_entities.get('user_location') != 'unknown':
                    command.entities.append(SemanticEntity(
                        entity_type='location',
                        value=context_entities['user_location'],
                        confidence=0.6
                    ))

        # Add last referenced object if picking up but no object specified
        if command.action == RobotActionType.PICK_UP:
            entities = {e.entity_type: e.value for e in command.entities}
            if 'object' not in entities:
                context_obj_refs = self.context_manager.current_context.get('object_references', {})
                if context_obj_refs:
                    last_obj = list(context_obj_refs.values())[-1]
                    command.entities.append(SemanticEntity(
                        entity_type='object',
                        value=last_obj,
                        confidence=0.6
                    ))

        # Update context with this command
        self.context_manager.add_conversation_turn(command.raw_text)
        self.context_manager.update_context(
            last_action={
                'action': command.action.value,
                'entities': [e.value for e in command.entities]
            }
        )
```

## Dialogue Management for Human-Robot Interaction

### State-Based Dialogue Manager

```python
from enum import Enum
from typing import Union

class DialogueState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    ERROR = "error"
    COMPLETED = "completed"

class DialogueManager:
    def __init__(self):
        self.state = DialogueState.IDLE
        self.current_intent = None
        self.pending_information = {}
        self.conversation_context = ContextManager()
        self.max_ambiguity_attempts = 3

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return response"""
        response = {
            'action': 'listen',  # Default action
            'message': '',
            'state': self.state.value
        }

        if self.state == DialogueState.IDLE:
            return self._handle_idle_state(user_input, response)
        elif self.state == DialogueState.AWAITING_CLARIFICATION:
            return self._handle_clarification_state(user_input, response)
        else:
            # Handle other states as needed
            return self._handle_general_state(user_input, response)

    def _handle_idle_state(self, user_input: str, response: Dict) -> Dict:
        """Handle input when in idle state"""
        # Parse the command
        parser = ContextualSemanticParser()
        parsed_command = parser.parse_command_with_context(user_input)

        if not parsed_command:
            response['message'] = "I didn't understand that command. Could you please rephrase?"
            response['action'] = 'listen'
            self.state = DialogueState.IDLE
        elif parsed_command.confidence < 0.7:
            # Low confidence - ask for clarification
            response['message'] = self._generate_clarification_request(parsed_command)
            response['action'] = 'listen'
            self.state = DialogueState.AWAITING_CLARIFICATION
            self.current_intent = parsed_command
        else:
            # High confidence - proceed with execution
            response['message'] = f"Okay, I will {self._describe_action(parsed_command)}."
            response['action'] = 'execute'
            response['command'] = parsed_command
            self.state = DialogueState.PROCESSING
            self.current_intent = parsed_command

        return response

    def _handle_clarification_state(self, user_input: str, response: Dict) -> Dict:
        """Handle input when awaiting clarification"""
        # Process clarification input
        clarification_result = self._process_clarification(user_input)

        if clarification_result['success']:
            response['message'] = f"Okay, I will {self._describe_action(clarification_result['command'])}."
            response['action'] = 'execute'
            response['command'] = clarification_result['command']
            self.state = DialogueState.PROCESSING
            self.current_intent = clarification_result['command']
        else:
            self.pending_information = clarification_result.get('needed_info', {})
            response['message'] = clarification_result['message']
            response['action'] = 'listen'

        # Check if we've asked too many times
        if self._get_clarification_attempts() >= self.max_ambiguity_attempts:
            response['message'] = "I'm having trouble understanding. Let's try a simpler command."
            response['action'] = 'listen'
            self._reset_dialogue()

        return response

    def _process_clarification(self, user_input: str) -> Dict:
        """Process clarification input"""
        # Simple approach: treat clarification as additional information
        # In practice, this would be more sophisticated
        parser = ContextualSemanticParser()

        if self.current_intent:
            # Try to incorporate new information into existing intent
            # This is a simplified approach
            combined_text = f"{self.current_intent.raw_text} {user_input}"
            new_command = parser.parse_command_with_context(combined_text)

            if new_command and new_command.confidence > 0.6:
                return {
                    'success': True,
                    'command': new_command
                }

        # If combination didn't work, ask again
        return {
            'success': False,
            'message': self._generate_clarification_request(self.current_intent),
            'needed_info': self._determine_needed_information(self.current_intent)
        }

    def _generate_clarification_request(self, command: ParsedCommand) -> str:
        """Generate a request for clarification"""
        if not command:
            return "Could you please repeat that more clearly?"

        entities = {e.entity_type: e.value for e in command.entities}

        # Check for missing critical information
        if command.action == RobotActionType.NAVIGATE and 'location' not in entities:
            return "Where would you like me to go?"
        elif command.action in [RobotActionType.PICK_UP, RobotActionType.PLACE] and 'object' not in entities:
            return "What would you like me to pick up?"
        elif command.action == RobotActionType.FOLLOW and 'object' not in entities:
            return "Whom should I follow?"

        # For general ambiguity
        return f"Could you clarify what you mean by '{command.raw_text}'?"

    def _determine_needed_information(self, command: ParsedCommand) -> Dict:
        """Determine what information is needed for clarification"""
        needed = {}

        if command:
            entities = {e.entity_type: e.value for e in command.entities}

            if command.action == RobotActionType.NAVIGATE and 'location' not in entities:
                needed['location'] = 'destination'
            elif command.action in [RobotActionType.PICK_UP, RobotActionType.PLACE] and 'object' not in entities:
                needed['object'] = 'item to interact with'
            elif command.action == RobotActionType.FOLLOW and 'object' not in entities:
                needed['object'] = 'entity to follow'

        return needed

    def _get_clarification_attempts(self) -> int:
        """Get number of clarification attempts for current intent"""
        # In a real implementation, this would track attempts
        return 1

    def _reset_dialogue(self):
        """Reset dialogue state"""
        self.state = DialogueState.IDLE
        self.current_intent = None
        self.pending_information = {}

    def _describe_action(self, command: ParsedCommand) -> str:
        """Generate a human-readable description of the action"""
        action_descriptions = {
            RobotActionType.NAVIGATE: "navigate to a location",
            RobotActionType.PICK_UP: "pick up an object",
            RobotActionType.PLACE: "place an object",
            RobotActionType.FOLLOW: "follow someone",
            RobotActionType.FIND: "find an object",
            RobotActionType.REPORT: "report status",
            RobotActionType.STOP: "stop current action",
            RobotActionType.WAIT: "wait for further instructions"
        }

        base_desc = action_descriptions.get(command.action, "perform an action")

        # Add specific details
        entities = {e.entity_type: e.value for e in command.entities}
        details = []

        if 'location' in entities:
            details.append(f"to {entities['location']}")
        if 'object' in entities:
            details.append(f"{entities['object']}")

        if details:
            return f"{base_desc} {' '.join(details)}"
        else:
            return base_desc

    def update_state(self, new_state: DialogueState):
        """Update dialogue state"""
        self.state = new_state

    def get_state(self) -> DialogueState:
        """Get current dialogue state"""
        return self.state
```

## Advanced NLU Techniques

### Using Large Language Models for Understanding

```python
import openai
from typing import Optional
import json

class LLMEnhancedNLU:
    def __init__(self, api_key: str = None):
        if api_key:
            openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key) if api_key else None

    def parse_with_llm(self, text: str, context: Dict = None) -> Optional[ParsedCommand]:
        """Parse command using large language model"""
        if not self.client:
            return None

        # Create a structured prompt for the LLM
        prompt = self._create_nlu_prompt(text, context)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            # Parse the LLM response
            result = self._parse_llm_response(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"LLM parsing error: {e}")
            return None

    def _create_nlu_prompt(self, text: str, context: Dict) -> str:
        """Create prompt for NLU task"""
        context_str = json.dumps(context) if context else "{}"

        prompt = f"""
        Parse the following natural language command for a robot. Extract the action, objects, locations, and any other relevant entities.

        Context: {context_str}
        Command: "{text}"

        Respond in JSON format with the following structure:
        {{
            "action": "action_type",
            "entities": [
                {{"type": "entity_type", "value": "entity_value", "confidence": 0.8}}
            ],
            "confidence": 0.9,
            "structured_query": {{"action": "action_type", "parameters": {{"key": "value"}}}}
        }}

        Action types: navigate, pick_up, place, follow, find, report, stop, wait
        Entity types: location, object, color, size, modifier
        """
        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for NLU task"""
        return """
        You are a natural language understanding system for robotics. Your job is to parse human commands into structured robot actions.
        Focus on extracting clear action types and relevant entities. Be concise and accurate.
        """

    def _parse_llm_response(self, response_text: str) -> Optional[ParsedCommand]:
        """Parse LLM response into ParsedCommand"""
        try:
            # Clean up response if it contains explanations
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)

                # Convert to ParsedCommand
                entities = []
                for entity_data in data.get('entities', []):
                    entity = SemanticEntity(
                        entity_type=entity_data['type'],
                        value=entity_data['value'],
                        confidence=entity_data.get('confidence', 0.8)
                    )
                    entities.append(entity)

                return ParsedCommand(
                    action=RobotActionType(data['action']),
                    entities=entities,
                    confidence=data.get('confidence', 0.8),
                    raw_text="",  # Original text would come from input
                    structured_query=data.get('structured_query', {})
                )
        except json.JSONDecodeError:
            print(f"Could not parse LLM response as JSON: {response_text}")
        except KeyError as e:
            print(f"Missing key in LLM response: {e}")
        except Exception as e:
            print(f"Error parsing LLM response: {e}")

        return None

class HybridNLU:
    """Combine rule-based and LLM-based NLU"""
    def __init__(self, llm_api_key: str = None):
        self.rule_parser = SemanticParser()
        self.llm_nlu = LLMEnhancedNLU(api_key=llm_api_key) if llm_api_key else None
        self.context_manager = ContextManager()

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse command using hybrid approach"""
        # Try rule-based parsing first (faster)
        rule_result = self.rule_parser.parse_command(text)

        if rule_result and rule_result.confidence > 0.8:
            # High confidence from rule-based, return it
            return rule_result

        # Low confidence from rule-based, try LLM if available
        if self.llm_nlu:
            llm_result = self.llm_nlu.parse_with_llm(text, self.context_manager.current_context)
            if llm_result:
                # Use LLM result, but adjust confidence based on rule result
                if rule_result:
                    # Combine confidences
                    combined_confidence = max(rule_result.confidence * 0.7, llm_result.confidence * 0.3)
                    llm_result.confidence = combined_confidence
                return llm_result

        # Fall back to rule-based result
        return rule_result
```

## Error Handling and Clarification Strategies

### Robust Error Handling System

```python
from typing import Callable, Any

class ErrorHandlingNLU:
    def __init__(self):
        self.nlu_system = HybridNLU()
        self.error_handlers = {}
        self.user_preferences = {}  # Store user preferences for interaction style

    def robust_parse(self, text: str, max_attempts: int = 3) -> Optional[ParsedCommand]:
        """Parse command with error handling and retries"""
        last_error = None

        for attempt in range(max_attempts):
            try:
                result = self.nlu_system.parse_command(text)

                if result and result.confidence > 0.5:
                    return result
                elif result:
                    # Low confidence result - might need clarification
                    clarification = self.request_clarification(result, text)
                    if clarification:
                        # In a real system, you'd get user input here
                        # For simulation, we'll return the low confidence result
                        return result

            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")

                # Apply error-specific handling
                error_type = self._categorize_error(e)
                handler = self.error_handlers.get(error_type)
                if handler:
                    text = handler(text, e)

        # All attempts failed
        if last_error:
            raise last_error
        return None

    def request_clarification(self, command: ParsedCommand, original_text: str) -> str:
        """Request clarification for ambiguous command"""
        # Identify what needs clarification
        issues = self._identify_clarification_needs(command)

        if not issues:
            return ""

        # Generate appropriate clarification request
        if 'location' in issues:
            return "Could you specify where you'd like me to go?"
        elif 'object' in issues:
            return "Could you be more specific about what you want me to pick up?"
        elif 'action' in issues:
            return f"I'm not sure what you mean by '{original_text}'. Could you rephrase?"
        else:
            return f"I didn't quite understand '{original_text}'. Could you say that again?"

    def _identify_clarification_needs(self, command: ParsedCommand) -> List[str]:
        """Identify what aspects of command need clarification"""
        needs = []

        if command.confidence < 0.7:
            needs.append('general')

        entities = {e.entity_type: e.value for e in command.entities}

        # Check for missing critical information
        if command.action == RobotActionType.NAVIGATE and not entities.get('location'):
            needs.append('location')
        elif command.action in [RobotActionType.PICK_UP, RobotActionType.PLACE] and not entities.get('object'):
            needs.append('object')
        elif command.action == RobotActionType.FOLLOW and not entities.get('object'):
            needs.append('object')

        return needs

    def _categorize_error(self, error: Exception) -> str:
        """Categorize error for appropriate handling"""
        error_str = str(error).lower()

        if 'api' in error_str or 'key' in error_str:
            return 'api_error'
        elif 'timeout' in error_str:
            return 'timeout_error'
        elif 'network' in error_str:
            return 'network_error'
        else:
            return 'parsing_error'

    def add_error_handler(self, error_type: str, handler: Callable[[str, Exception], str]):
        """Add error handler for specific error type"""
        self.error_handlers[error_type] = handler

    def register_api_error_handler(self):
        """Register handler for API errors (fallback to rule-based)"""
        def handle_api_error(text: str, error: Exception) -> str:
            print("API error occurred, falling back to rule-based parsing")
            # Return original text to be parsed differently
            return text
        self.add_error_handler('api_error', handle_api_error)

    def register_timeout_handler(self):
        """Register handler for timeout errors"""
        def handle_timeout(text: str, error: Exception) -> str:
            print("Request timed out, trying simplified parsing")
            # Try with simplified approach
            return text
        self.add_error_handler('timeout_error', handle_timeout)

# Example usage of error handling
def example_error_handling():
    """Example of error handling in NLU system"""
    error_nlu = ErrorHandlingNLU()
    error_nlu.register_api_error_handler()
    error_nlu.register_timeout_handler()

    # Example commands
    commands = [
        "Go to the kitchen and bring me the red cup",
        "Pick up that thing over there",  # Ambiguous
        "Do the thing"  # Very ambiguous
    ]

    for cmd in commands:
        try:
            result = error_nlu.robust_parse(cmd)
            if result:
                print(f"Command: {cmd}")
                print(f"Parsed: {result.action.value} with confidence {result.confidence:.2f}")
                print(f"Entities: {[e.value for e in result.entities]}")
                print()
            else:
                print(f"Could not parse: {cmd}")
        except Exception as e:
            print(f"Error parsing '{cmd}': {e}")
```

## Integration with Robotics Systems

### ROS 2 Integration Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import json

class NaturalLanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('nlu_node')

        # Initialize NLU system
        self.nlu_system = HybridNLU()
        self.dialogue_manager = DialogueManager()

        # ROS 2 interfaces
        self.command_sub = self.create_subscription(
            String,
            'voice_command',
            self.command_callback,
            10
        )

        self.response_pub = self.create_publisher(
            String,
            'nlu_response',
            10
        )

        self.action_plan_pub = self.create_publisher(
            String,
            'action_plan',
            10
        )

        # Action clients for robot capabilities
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manip_client = ActionClient(self, ManipulationAction, 'manipulation')

        self.get_logger().info("NLU Node initialized")

    def command_callback(self, msg: String):
        """Process incoming voice commands"""
        self.get_logger().info(f"Processing command: {msg.data}")

        try:
            # Parse the command using our NLU system
            parsed_command = self.nlu_system.parse_command(msg.data)

            if not parsed_command:
                self.get_logger().warn("Could not parse command")
                self._send_response("I didn't understand that command")
                return

            self.get_logger().info(f"Parsed action: {parsed_command.action.value}")

            # Generate response
            response_text = f"I understand you want me to {self._describe_action(parsed_command)}"
            self._send_response(response_text)

            # Create and publish action plan
            action_plan = self._create_action_plan(parsed_command)
            self._publish_action_plan(action_plan)

        except Exception as e:
            self.get_logger().error(f"Error in NLU processing: {e}")
            self._send_response("Sorry, I encountered an error processing your command")

    def _create_action_plan(self, parsed_command: ParsedCommand) -> Dict:
        """Create executable action plan from parsed command"""
        plan = {
            'action': parsed_command.action.value,
            'entities': [
                {
                    'type': entity.entity_type,
                    'value': entity.value,
                    'confidence': entity.confidence
                }
                for entity in parsed_command.entities
            ],
            'confidence': parsed_command.confidence,
            'timestamp': self.get_clock().now().to_msg()
        }

        # Add specific parameters based on action type
        if parsed_command.action == RobotActionType.NAVIGATE:
            # Look up location coordinates
            for entity in parsed_command.entities:
                if entity.entity_type == 'location':
                    plan['target_location'] = entity.value
                    break

        elif parsed_command.action in [RobotActionType.PICK_UP, RobotActionType.PLACE]:
            # Add object information
            for entity in parsed_command.entities:
                if entity.entity_type == 'object':
                    plan['target_object'] = entity.value
                    break
                if entity.entity_type == 'color':
                    plan['object_color'] = entity.value

        return plan

    def _send_response(self, text: str):
        """Send response back to user"""
        response_msg = String()
        response_msg.data = text
        self.response_pub.publish(response_msg)

    def _publish_action_plan(self, plan: Dict):
        """Publish action plan for execution"""
        plan_msg = String()
        plan_msg.data = json.dumps(plan)
        self.action_plan_pub.publish(plan_msg)

    def _describe_action(self, command: ParsedCommand) -> str:
        """Create a human-readable description of the action"""
        action_descriptions = {
            RobotActionType.NAVIGATE: "navigate to a location",
            RobotActionType.PICK_UP: "pick up an object",
            RobotActionType.PLACE: "place an object",
            RobotActionType.FOLLOW: "follow someone",
            RobotActionType.FIND: "find an object",
            RobotActionType.REPORT: "report status",
            RobotActionType.STOP: "stop current action",
            RobotActionType.WAIT: "wait for further instructions"
        }

        base_desc = action_descriptions.get(command.action, "perform an action")

        # Add specific details
        entities = {e.entity_type: e.value for e in command.entities}
        details = []

        if 'location' in entities:
            details.append(f"to {entities['location']}")
        if 'object' in entities:
            details.append(f"{entities['object']}")

        if details:
            return f"{base_desc} {' '.join(details)}"
        else:
            return base_desc

def main(args=None):
    rclpy.init(args=args)
    node = NaturalLanguageUnderstandingNode()

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

## Evaluation and Testing

### NLU System Evaluation

```python
import unittest
from typing import List, Tuple

class NLUTestSuite:
    def __init__(self):
        self.parser = SemanticParser()
        self.test_cases = [
            # (input, expected_action, expected_entities)
            ("Go to kitchen", RobotActionType.NAVIGATE, ['kitchen']),
            ("Pick up the red cup", RobotActionType.PICK_UP, ['cup', 'red']),
            ("Bring me the book", RobotActionType.PICK_UP, ['book']),
            ("Put the glass on the table", RobotActionType.PLACE, ['glass', 'table']),
            ("Find the keys", RobotActionType.FIND, ['keys']),
            ("Stop", RobotActionType.STOP, []),
        ]

    def run_tests(self) -> Dict[str, Any]:
        """Run all NLU tests"""
        results = {
            'total': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for input_text, expected_action, expected_entities in self.test_cases:
            result = self._test_single_case(input_text, expected_action, expected_entities)
            results['details'].append(result)

            if result['passed']:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def _test_single_case(self, input_text: str, expected_action: RobotActionType,
                         expected_entities: List[str]) -> Dict[str, Any]:
        """Test a single NLU case"""
        result = {
            'input': input_text,
            'expected_action': expected_action.value,
            'expected_entities': expected_entities,
            'passed': False,
            'actual_action': None,
            'actual_entities': [],
            'confidence': 0.0
        }

        try:
            parsed = self.parser.parse_command(input_text)
            if parsed:
                result['actual_action'] = parsed.action.value
                result['actual_entities'] = [e.value for e in parsed.entities]
                result['confidence'] = parsed.confidence

                # Check if action matches
                action_match = parsed.action == expected_action

                # Check if entities are present (subset check)
                expected_set = set(expected_entities)
                actual_set = set(result['actual_entities'])
                entities_match = expected_set.issubset(actual_set)

                result['passed'] = action_match and entities_match

        except Exception as e:
            result['error'] = str(e)

        return result

# Example usage
def run_nlu_evaluation():
    """Run NLU system evaluation"""
    test_suite = NLUTestSuite()
    results = test_suite.run_tests()

    print(f"NLU Evaluation Results:")
    print(f"Total tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Accuracy: {results['passed']/results['total']*100:.1f}%")

    print("\nDetailed Results:")
    for detail in results['details']:
        status = "✓" if detail['passed'] else "✗"
        print(f"{status} '{detail['input']}' -> "
              f"Expected: {detail['expected_action']}, "
              f"Actual: {detail['actual_action']}, "
              f"Conf: {detail['confidence']:.2f}")
```

## Summary

Natural Language Understanding for robotics involves:

- **Semantic Parsing**: Converting natural language to structured robot commands
- **Context Awareness**: Maintaining conversation and environmental context
- **Dialogue Management**: Handling multi-turn conversations and clarifications
- **Error Handling**: Managing ambiguous or incorrect commands
- **Integration**: Connecting NLU with robot action systems
- **Evaluation**: Testing and validating NLU performance

These capabilities enable robots to understand and respond to natural language commands in a meaningful way, making human-robot interaction more intuitive and accessible.

## Exercises

1. Implement a semantic parser for robot commands with pattern matching
2. Create a context manager for maintaining conversation state
3. Develop a dialogue system that handles ambiguous commands
4. Integrate NLU with a ROS 2 robotics system