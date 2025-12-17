---
sidebar_position: 3
---

# Physics Simulation in Gazebo

## Learning Objectives
- Understand the physics simulation concepts in Gazebo
- Learn to configure physics properties for realistic simulation
- Implement gravity, collisions, and rigid body dynamics
- Optimize physics simulation for performance

## Physics Simulation Fundamentals

Gazebo uses physics engines to simulate real-world physics behaviors. The most commonly used engines are:

- **ODE (Open Dynamics Engine)**: Default engine, good for rigid body simulation
- **Bullet**: Fast and robust, good for real-time applications
- **DART**: Advanced constraints and stability
- **Simbody**: High-fidelity multibody dynamics

### Physics Engine Configuration

Physics properties are defined in world files:

```xml
<world name="physics_example">
  <physics name="ode_physics" type="ode">
    <!-- Time step for physics updates -->
    <max_step_size>0.001</max_step_size>

    <!-- Real-time update rate -->
    <real_time_update_rate>1000</real_time_update_rate>

    <!-- Real-time factor (1.0 = real-time) -->
    <real_time_factor>1.0</real_time_factor>

    <!-- Gravity vector -->
    <gravity>0 0 -9.8</gravity>

    <!-- ODE-specific parameters -->
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
</world>
```

## Gravity and Environmental Forces

### Gravity Configuration
Gravity is a fundamental force in physics simulation:

```xml
<gravity>0 0 -9.8</gravity>  <!-- Earth's gravity -->
<!-- Other examples -->
<!-- <gravity>0 0 -1.62</gravity>  Lunar gravity -->
<!-- <gravity>0 0 -3.71</gravity>  Martian gravity -->
```

### Custom Forces
You can apply custom forces using plugins:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
  class CustomForcePlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      this->world = _model->GetWorld();

      // Connect to physics update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&CustomForcePlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Apply custom force to a link
      auto link = this->model->GetLink("body_link");
      math::Vector3 force(0, 0, 10);  // 10N upward force
      link->AddForce(force);
    }

    private: physics::ModelPtr model;
    private: physics::WorldPtr world;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(CustomForcePlugin)
}
```

## Collision Detection and Response

### Collision Properties
Collision properties are defined within links:

```xml
<link name="collision_example">
  <collision name="collision">
    <geometry>
      <box><size>1 1 1</size></box>
    </geometry>

    <!-- Surface properties -->
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>      <!-- Static friction coefficient -->
          <mu2>1.0</mu2>    <!-- Secondary friction coefficient -->
        </ode>
      </friction>

      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>

      <contact>
        <ode>
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Contact Sensors
Monitor contacts between objects:

```xml
<sensor name="contact_sensor" type="contact">
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <contact>
    <collision>collision_example_collision</collision>
  </contact>
  <plugin name="contact_plugin" filename="libgazebo_ros_contact.so">
    <ros>
      <namespace>/robot</namespace>
      <remapping>~/out:=contact_states</remapping>
    </ros>
  </plugin>
</sensor>
```

## Rigid Body Dynamics

### Inertial Properties
Accurate inertial properties are crucial for realistic simulation:

```xml
<link name="rigid_body">
  <inertial>
    <!-- Mass in kg -->
    <mass>1.0</mass>

    <!-- Origin offset -->
    <origin xyz="0 0 0" rpy="0 0 0"/>

    <!-- Inertia matrix -->
    <inertia>
      <!-- Diagonal elements -->
      <ixx>0.166667</ixx>
      <ixy>0</ixy>
      <ixz>0</ixz>
      <iyy>0.166667</iyy>
      <iyz>0</iyz>
      <izz>0.166667</izz>
    </inertia>
  </inertial>
</link>
```

### Calculating Inertial Properties

For common shapes (mass m, dimensions as appropriate):

**Box (width w, depth d, height h):**
```
ixx = 1/12 * m * (d² + h²)
iyy = 1/12 * m * (w² + h²)
izz = 1/12 * m * (w² + d²)
```

**Cylinder (radius r, height h):**
```
ixx = iyy = 1/12 * m * (3*r² + h²)
izz = 1/2 * m * r²
```

**Sphere (radius r):**
```
ixx = iyy = izz = 2/5 * m * r²
```

## Advanced Physics Concepts

### Joint Dynamics
Configure joint behavior with dynamics properties:

```xml
<joint name="motor_joint" type="revolute">
  <parent link="base_link"/>
  <child link="arm_link"/>

  <dynamics damping="0.1" friction="0.05"/>

  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

### Soft Contacts
For more realistic contact simulation:

```xml
<world name="soft_contacts">
  <physics name="ode_physics" type="ode">
    <ode>
      <constraints>
        <contact_surface_layer>0.005</contact_surface_layer>
        <max_contacts>20</max_contacts>
      </constraints>
    </ode>
  </physics>
</world>
```

## Performance Optimization

### Physics Update Rate
Balance accuracy and performance:

```xml
<physics>
  <!-- For fast simulation: larger time step, lower update rate -->
  <max_step_size>0.01</max_step_size>
  <real_time_update_rate>100</real_time_update_rate>

  <!-- For accurate simulation: smaller time step, higher update rate -->
  <!-- <max_step_size>0.001</max_step_size> -->
  <!-- <real_time_update_rate>1000</real_time_update_rate> -->
</physics>
```

### Collision Optimization
- Use simpler collision geometries than visual geometries
- Use bounding boxes for complex meshes when possible
- Reduce the number of collision elements

### Solver Optimization
```xml
<physics type="ode">
  <ode>
    <solver>
      <!-- Quick solver for real-time performance -->
      <type>quick</type>
      <iters>20</iters>  <!-- Increase for stability -->
      <sor>1.3</sor>     <!-- Successive over-relaxation parameter -->
    </solver>
  </ode>
</physics>
```

## Physics Debugging

### Visualizing Physics Properties
Enable physics visualization in Gazebo:

```xml
<world>
  <!-- Physics visualization -->
  <physics name="ode_physics" type="ode">
    <real_time_factor>1.0</real_time_factor>
    <!-- Enable contact visualization -->
    <ode>
      <solver>
        <type>quick</type>
      </solver>
    </ode>
  </physics>

  <gui>
    <plugin name="world_stats" filename="WorldStats">
      <!-- Physics statistics display -->
    </plugin>
  </gui>
</world>
```

### Common Physics Issues and Solutions

1. **Objects falling through surfaces:**
   - Check collision geometry
   - Increase physics update rate
   - Adjust surface layer depth

2. **Unstable simulation:**
   - Reduce time step
   - Increase solver iterations
   - Check inertial properties

3. **Objects bouncing unrealistically:**
   - Adjust restitution coefficients
   - Verify mass values
   - Check for coincident surfaces

## Integration with ROS 2

### Physics State Monitoring
Monitor physics state through ROS 2 topics:

```python
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import LinkStates, ModelStates
from geometry_msgs.msg import Twist

class PhysicsMonitor(Node):
    def __init__(self):
        super().__init__('physics_monitor')

        # Subscribe to Gazebo link states
        self.link_state_sub = self.create_subscription(
            LinkStates,
            '/gazebo/link_states',
            self.link_states_callback,
            10
        )

        # Subscribe to Gazebo model states
        self.model_state_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10
        )

        # Publisher for commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def link_states_callback(self, msg):
        # Process link states for physics analysis
        for i, name in enumerate(msg.name):
            if name == 'robot::base_link':
                linear_vel = msg.twist[i].linear
                angular_vel = msg.twist[i].angular
                self.get_logger().info(
                    f'Velocity - Linear: {linear_vel}, Angular: {angular_vel}'
                )

    def model_states_callback(self, msg):
        # Process model states for position tracking
        for i, name in enumerate(msg.name):
            if name == 'robot':
                position = msg.pose[i].position
                orientation = msg.pose[i].orientation
                self.get_logger().info(
                    f'Position: ({position.x}, {position.y}, {position.z})'
                )

def main(args=None):
    rclpy.init(args=args)
    monitor = PhysicsMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Physics simulation in Gazebo requires understanding several key concepts:

- **Physics engines**: Different engines offer trade-offs between accuracy and performance
- **Gravity and forces**: Environmental forces drive the simulation
- **Collision detection**: Proper collision properties ensure realistic interactions
- **Rigid body dynamics**: Accurate inertial properties are essential for realistic behavior
- **Performance optimization**: Balance accuracy with simulation speed
- **ROS 2 integration**: Monitor and control physics through ROS 2 interfaces

## Exercises

1. Create a world with different physics materials and test object interactions
2. Implement a custom force plugin for a specific application
3. Configure physics properties for a humanoid robot to ensure stable walking
4. Optimize physics simulation for a complex multi-robot scenario