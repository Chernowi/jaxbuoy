from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="buoy_agent_inference",
                executable="inference_node",
                name="buoy_inference_node",
                output="screen",
                parameters=["config/inference_params.yaml"],
            )
        ]
    )
