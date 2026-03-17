from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray

from .minimal_policy import MinimalRNNPolicy


class BuoyInferenceNode(Node):
    def __init__(self):
        super().__init__("buoy_inference_node")

        self.declare_parameter(
            "model_path",
            "ros2_agent_integration/model_assets/wind_entropy_curiosity_high_policy.npz",
        )
        self.declare_parameter(
            "metadata_path",
            "ros2_agent_integration/model_assets/wind_entropy_curiosity_high_metadata.json",
        )
        self.declare_parameter("obs_topic", "/agent/observation")
        self.declare_parameter("action_topic", "/agent/action")
        self.declare_parameter("reset_topic", "/agent/reset_rnn")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("clip_actions", True)

        model_path = Path(self.get_parameter("model_path").value)
        metadata_path = Path(self.get_parameter("metadata_path").value)
        self.deterministic = bool(self.get_parameter("deterministic").value)
        self.clip_actions = bool(self.get_parameter("clip_actions").value)

        self.policy = MinimalRNNPolicy(model_path=model_path, metadata_path=metadata_path)

        obs_topic = str(self.get_parameter("obs_topic").value)
        action_topic = str(self.get_parameter("action_topic").value)
        reset_topic = str(self.get_parameter("reset_topic").value)

        self.action_pub = self.create_publisher(Float32MultiArray, action_topic, 10)
        self.obs_sub = self.create_subscription(Float32MultiArray, obs_topic, self._on_obs, 10)
        self.reset_sub = self.create_subscription(Bool, reset_topic, self._on_reset, 10)

        self.get_logger().info(
            f"Loaded policy model={model_path} obs_dim={self.policy.obs_dim} action_dim={self.policy.action_dim}"
        )

    def _on_reset(self, msg: Bool):
        if msg.data:
            self.policy.reset()

    def _on_obs(self, msg: Float32MultiArray):
        obs = np.asarray(msg.data, dtype=np.float32)
        action = self.policy.infer(
            obs,
            done=False,
            deterministic=self.deterministic,
            clip_actions=self.clip_actions,
        )
        out = Float32MultiArray()
        out.data = action.tolist()
        self.action_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = BuoyInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
