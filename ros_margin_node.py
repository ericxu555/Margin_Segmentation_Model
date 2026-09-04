"""ROS 2 node: two-stage tumor margin segmentation on a live camera topic.

Runs the same two models as run_margin.py, on frames arriving from a ROS topic
instead of a folder, and publishes the margin drawn over the live image.

Unlike the toolhead-following pipeline, this model is stateless per frame:
each frame gets its own margin mask, with only the EMA carried across frames.
There is no episode to bracket, so the node has no start/stop services. It
segments and publishes continuously from launch.

    Subscribes  /ves_camera/image            sensor_msgs/Image
    Publishes   /margin/overlay              sensor_msgs/Image  (bgr8)
                /margin/mask                 sensor_msgs/Image  (mono8, 255=margin)

Run:
    python3 ros_margin_node.py
    python3 ros_margin_node.py --ros-args -p image_topic:=/ves_camera/image_rect

The pipeline matches run_margin.py exactly: Stage 1 segments the tumor and is
dilated by 20 px to constrain where the margin may appear; Stage 2 segments the
margin at 0.6 confidence, EMA-smoothed with alpha 0.3, then ANDed with the
Stage 1 mask.
"""
import os
import sys
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_IMAGE_TOPIC = "/ves_camera/image"

# The models were trained on 540x540 frames, while the reference camera
# publishes 1080x1080. Frames are resized to PROC_SIZE before processing so the
# input distribution matches training; overlays are published at the camera's
# native resolution.
PROC_SIZE = 540

# Inference settings, identical to run_margin.py's validated defaults.
THRESHOLD = 0.6          # confidence a pixel needs to be called margin
EMA_ALPHA = 0.3          # temporal smoothing of the confidence map
SPATIAL_DILATE = 20      # dilation of the Stage 1 tumor mask before gating

MARGIN_MODEL = os.environ.get(
    "MARGIN_MODEL", os.path.join(HERE, "checkpoints_margin58", "best_model_epoch_swa.pth"))
MARGIN_META = os.environ.get(
    "MARGIN_META", os.path.join(HERE, "checkpoints_margin58", "best_model_epoch_swa_metadata.json"))
TUMOR_MODEL = os.environ.get(
    "SPATIAL_MODEL", os.path.join(HERE, "checkpoints_tumor_binary10", "best_model_epoch_swa.pth"))
TUMOR_META = os.environ.get(
    "SPATIAL_META", os.path.join(HERE, "checkpoints_tumor_binary10", "best_model_epoch_swa_metadata.json"))

MASK_COLOR_BGR = (0, 0, 255)   # red, matching the offline pipeline
OVERLAY_ALPHA = 0.5


class MarginNode(Node):

    def __init__(self):
        super().__init__("margin_segmentation")

        self.declare_parameter("image_topic", DEFAULT_IMAGE_TOPIC)
        self.declare_parameter("proc_size", PROC_SIZE)
        self.declare_parameter("threshold", THRESHOLD)
        self.declare_parameter("ema_alpha", EMA_ALPHA)
        self.declare_parameter("spatial_dilate", SPATIAL_DILATE)
        self.declare_parameter("device", "cuda")

        self.image_topic = self.get_parameter("image_topic").value
        self.proc_size = int(self.get_parameter("proc_size").value)
        self.threshold = float(self.get_parameter("threshold").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.spatial_dilate = int(self.get_parameter("spatial_dilate").value)
        self.device = self.get_parameter("device").value

        self.lock = threading.Lock()
        self.busy = False
        self.ema = None
        self.native_hw = None
        self.n_received = self.n_processed = self.n_dropped = 0

        self._load_models()

        self.pub_overlay = self.create_publisher(Image, "/margin/overlay", 1)
        self.pub_mask = self.create_publisher(Image, "/margin/mask", 1)

        # Keep only the newest frame: a slow pass must not build a backlog that
        # makes the overlay lag the camera. RELIABLE matches the publisher.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, self.image_topic, self._on_image, qos)

        self.create_timer(10.0, self._report)
        self.get_logger().info(
            f"Segmenting {self.image_topic} -> /margin/overlay, /margin/mask")

    # ---------------------------------------------------------------- models

    def _load_models(self):
        self.get_logger().info("Loading models...")
        missing = [p for p in (MARGIN_MODEL, MARGIN_META, TUMOR_MODEL, TUMOR_META)
                   if not os.path.isfile(p)]
        if missing:
            for m in missing:
                self.get_logger().error(f"Missing: {m}")
            raise SystemExit(
                "Model weights not found. Download them (README step 4) or set "
                "MARGIN_MODEL / SPATIAL_MODEL.")

        from src.inference_engine import InferenceEngine
        self.margin_engine = InferenceEngine(
            model_path=MARGIN_MODEL, metadata_path=MARGIN_META, device=self.device)
        self.tumor_engine = InferenceEngine(
            model_path=TUMOR_MODEL, metadata_path=TUMOR_META, device=self.device)

        import torch
        self.torch = torch
        self.get_logger().info(
            f"Models loaded on {self.device} "
            f"(threshold={self.threshold}, ema={self.ema_alpha}, "
            f"dilate={self.spatial_dilate}px).")

    # ---------------------------------------------------------------- stages

    def _margin_confidence(self, frame_bgr, h, w):
        """Stage 2: per-pixel margin probability at the frame's own size."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        probs, _ = self.margin_engine.predict_single(rgb)   # (2, 512, 512)
        red_512 = probs[1].float().cpu().numpy()
        return cv2.resize(red_512, (w, h), interpolation=cv2.INTER_LINEAR)

    def _tumor_mask(self, frame_bgr, h, w):
        """Stage 1: tumor region, dilated, used to gate the margin."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        probs, _ = self.tumor_engine.predict_single(rgb)
        tumor_512 = (probs.argmax(dim=0).cpu().numpy() > 0).astype(np.uint8)
        tumor = cv2.resize(tumor_512, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.spatial_dilate > 0:
            k = max(self.spatial_dilate | 1, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            tumor = cv2.dilate(tumor, kernel)
        return tumor.astype(bool)

    # -------------------------------------------------------------- callback

    def _on_image(self, msg):
        with self.lock:
            if self.busy:
                # Drop rather than queue: the newest frame is what matters.
                self.n_dropped += 1
                return
            self.busy = True
        try:
            self.n_received += 1
            native = self._to_bgr(msg)
            self.native_hw = native.shape[:2]
            if native.shape[:2] != (self.proc_size, self.proc_size):
                frame = cv2.resize(native, (self.proc_size, self.proc_size),
                                   interpolation=cv2.INTER_AREA)
            else:
                frame = native
            self._process(frame, msg.header)
            self.n_processed += 1
        except Exception as e:
            self.get_logger().error(f"Frame failed: {e}")
        finally:
            with self.lock:
                self.busy = False

    def _process(self, frame, header):
        h, w = frame.shape[:2]
        with self.torch.no_grad():
            conf = self._margin_confidence(frame, h, w)
            tumor = self._tumor_mask(frame, h, w)

        # EMA across frames, exactly as the offline pipeline does, so one noisy
        # frame cannot flicker the margin on and off.
        if self.ema_alpha < 1.0:
            self.ema = conf.copy() if self.ema is None else (
                self.ema_alpha * conf + (1.0 - self.ema_alpha) * self.ema)
            conf_for_thresh = self.ema
        else:
            conf_for_thresh = conf

        margin = (conf_for_thresh >= self.threshold) & tumor

        vis = frame.copy()
        vis[margin] = (OVERLAY_ALPHA * np.array(MASK_COLOR_BGR, dtype=float)
                       + (1.0 - OVERLAY_ALPHA) * vis[margin]).astype(np.uint8)

        nh, nw = self.native_hw or (h, w)
        mask_u8 = (margin.astype(np.uint8) * 255)
        if (nh, nw) != (h, w):
            vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask_u8 = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)

        self.pub_overlay.publish(self._to_msg(vis, header, "bgr8"))
        self.pub_mask.publish(self._to_msg(mask_u8, header, "mono8"))

    # --------------------------------------------------------------- helpers

    def _report(self):
        if self.n_received == 0:
            self.get_logger().warn(
                f"No frames received on {self.image_topic}. Is the camera "
                f"publishing? Check with: ros2 topic hz {self.image_topic}")
            return
        self.get_logger().info(
            f"{self.n_processed} frames processed, {self.n_dropped} dropped.")

    def _to_bgr(self, msg):
        """Convert sensor_msgs/Image to BGR without requiring cv_bridge."""
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        if enc in ("bgr8", "rgb8"):
            img = buf.reshape(msg.height, msg.width, 3)
            return img[:, :, ::-1].copy() if enc == "rgb8" else img.copy()
        if enc == "mono8":
            return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
        if enc in ("bgra8", "rgba8"):
            img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
            return img[:, :, ::-1].copy() if enc == "rgba8" else img.copy()
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    def _to_msg(self, img, header, encoding):
        msg = Image()
        msg.header = header
        msg.height, msg.width = img.shape[0], img.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = img.shape[1] * (3 if encoding == "bgr8" else 1)
        msg.data = img.tobytes()
        return msg


def main():
    rclpy.init()
    node = MarginNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
