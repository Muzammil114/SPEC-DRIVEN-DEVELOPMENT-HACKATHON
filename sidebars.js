// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'index',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1/index',
        'module-1/ros2-architecture',
        'module-1/nodes-topics-services',
        'module-1/rclpy-development',
        'module-1/launch-files',
        'module-1/urdf-robot-models',
        'module-1/module-1-exercises'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2/index',
        'module-2/gazebo-setup',
        'module-2/physics-simulation',
        'module-2/urdf-vs-sdf',
        'module-2/sensor-simulation',
        'module-2/unity-integration',
        'module-2/module-2-exercises'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-3/index',
        'module-3/isaac-sim-setup',
        'module-3/synthetic-data-generation',
        'module-3/isaac-ros-pipelines',
        'module-3/vslam-navigation',
        'module-3/nav2-navigation',
        'module-3/reinforcement-learning',
        'module-3/sim-to-real-transfer',
        'module-3/module-3-exercises'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4/index',
        'module-4/multimodal-perception',
        'module-4/voice-to-action',
        'module-4/natural-language-understanding',
        'module-4/llm-planning',
        'module-4/language-to-ros-actions',
        'module-4/module-4-exercises'
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/index',
        'capstone/conversational-robot',
        'capstone/validation-testing'
      ],
    },
    'conclusion'
  ],
};

module.exports = sidebars;