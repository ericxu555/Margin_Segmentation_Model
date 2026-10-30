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
used as is (no dilation) to constrain where the margin may appear; Stage 2 segments the
margin at 0.6 confidence, EMA-smoothed with alpha 0.3, then ANDed with the
Stage 1 mask.
"""
import os
import sys
import time
import json
import threading
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_IMAGE_TOPIC = "/ves_camera/image"
DEFAULT_OUTPUT_ROOT = os.path.join(HERE, "Output")

# Tool-button trigger, identical to the toolhead-following node. The Virtuoso
# publishes its joint state continuously; the entry named TOOL_JOINT_NAME reads
# 1.0 while the cut button is held and 0.0 when it is released, so the button
# can bracket a recording without anyone touching the terminal.
DEFAULT_TOOL_TOPIC = "/ves/right/joint/measured_jp"
TOOL_JOINT_NAME = "tool"
TOOL_PRESSED = 0.5        # midpoint of the 0.0/1.0 signal
TOOL_RELEASE_HOLD = 0.5   # seconds the button must read released before ending

# The models were trained on 540x540 frames, while the reference camera
# publishes 1080x1080. Frames are resized to PROC_SIZE before processing so the
# input distribution matches training; overlays are published at the camera's
# native resolution.
PROC_SIZE = 540

# Inference settings, identical to run_margin.py's validated defaults.
THRESHOLD = 0.6          # confidence a pixel needs to be called margin
EMA_ALPHA = 0.3          # temporal smoothing of the confidence map
SPATIAL_DILATE = 0       # dilation of the Stage 1 tumor mask before gating (0: the mask itself, as evaluated)

MARGIN_MODEL = os.environ.get(
    "MARGIN_MODEL", os.path.join(HERE, "checkpoints_margin58", "best_model_epoch_swa.pth"))
MARGIN_META = os.environ.get(
    "MARGIN_META", os.path.join(HERE, "checkpoints_margin58", "best_model_epoch_swa_metadata.json"))
TUMOR_MODEL = os.environ.get(
    "SPATIAL_MODEL", os.path.join(HERE, "checkpoints_tumor_binary11", "best_model_epoch_swa.pth"))
TUMOR_META = os.environ.get(
    "SPATIAL_META", os.path.join(HERE, "checkpoints_tumor_binary11", "best_model_epoch_swa_metadata.json"))

MASK_COLOR_BGR = (0, 0, 255)   # red, matching the offline pipeline
TUMOR_COLOR_BGR = (255, 128, 0)  # blue, to read apart from the red margin
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
        # "off": segment and publish continuously, as before. "tool": also
        # record an episode to disk while the cut button is held. Publishing is
        # never gated -- the overlays stay live either way, so the surgeon's
        # view does not change when recording starts or stops.
        self.declare_parameter("trigger_mode", "off")
        self.declare_parameter("tool_topic", DEFAULT_TOOL_TOPIC)
        self.declare_parameter("output_root", DEFAULT_OUTPUT_ROOT)
        # Recording every frame of both models means two videos and two PNG
        # sequences per episode, which is a lot of disk over a long procedure.
        # With last_only an episode keeps only the final overlay and mask of
        # each -- enough to check the result, nothing more.
        self.declare_parameter("last_only", False)

        self.image_topic = self.get_parameter("image_topic").value
        self.proc_size = int(self.get_parameter("proc_size").value)
        self.threshold = float(self.get_parameter("threshold").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)
        self.spatial_dilate = int(self.get_parameter("spatial_dilate").value)
        self.device = self.get_parameter("device").value
        self.trigger_mode = str(self.get_parameter("trigger_mode").value).lower()
        self.tool_topic = self.get_parameter("tool_topic").value
        self.output_root = self.get_parameter("output_root").value
        self.last_only = bool(self.get_parameter("last_only").value)

        self.lock = threading.Lock()
        self.busy = False
        self.ema = None
        self.native_hw = None
        self.n_received = self.n_processed = self.n_dropped = 0

        # Recording state. Both models are stateless per frame, so an episode
        # is a span of frames written to disk rather than something that
        # accumulates; self.recording says whether frames are being kept.
        self.recording = False
        self.episode_dir = None
        self.writers = {}          # tag -> cv2.VideoWriter
        self.last = {}             # tag -> (overlay, mask) of the newest frame
        self.n_recorded = 0
        self.t_start = None
        self.frames_log = None         # frames.csv of the episode being recorded
        self._tool_held = False        # last debounced button state
        self._tool_release_t = None    # when the button first read released
        self._tool_seen = False        # has any joint message arrived
        self._tool_ignore = False      # ignore a press held at startup

        self._load_models()

        self.pub_overlay = self.create_publisher(Image, "/margin/overlay", 1)
        self.pub_mask = self.create_publisher(Image, "/margin/mask", 1)
        # Stage 1 is computed every frame anyway, to gate the margin. These
        # expose it rather than recompute it, so the tumor segmentation costs
        # no extra inference and no second copy of the weights.
        self.pub_tumor_overlay = self.create_publisher(Image, "/tumor/overlay", 1)
        self.pub_tumor_mask = self.create_publisher(Image, "/tumor/mask", 1)

        # Keep only the newest frame: a slow pass must not build a backlog that
        # makes the overlay lag the camera. RELIABLE matches the publisher.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, self.image_topic, self._on_image, qos)

        if self.trigger_mode == "tool":
            jqos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(JointState, self.tool_topic,
                                     self._on_joint, jqos)
            self.get_logger().info(
                f"Recording trigger: tool button on {self.tool_topic} "
                f"(joint '{TOOL_JOINT_NAME}'). Hold to record; release to save."
                + ("  [last_only]" if self.last_only else ""))
        else:
            self.get_logger().info(
                "Recording trigger: off (publishing only, nothing written).")

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
        """Stage 1: the tumor region, returned RAW and DILATED.

        The dilated mask gates the margin -- widening it lets the margin sit
        slightly outside the predicted tumor edge. The raw mask is the model's
        actual boundary and is what gets published and recorded; saving the
        dilated one would overstate the tumor by the dilation.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        probs, _ = self.tumor_engine.predict_single(rgb)
        tumor_512 = (probs.argmax(dim=0).cpu().numpy() > 0).astype(np.uint8)
        raw = cv2.resize(tumor_512, (w, h), interpolation=cv2.INTER_NEAREST)
        gate = raw
        if self.spatial_dilate > 0:
            k = max(self.spatial_dilate | 1, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            gate = cv2.dilate(raw, kernel)
        return raw.astype(bool), gate.astype(bool)

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
            tumor_raw, tumor = self._tumor_mask(frame, h, w)

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

        # Stage 1's own overlay, from the RAW mask.
        tvis = frame.copy()
        tvis[tumor_raw] = (OVERLAY_ALPHA * np.array(TUMOR_COLOR_BGR, dtype=float)
                           + (1.0 - OVERLAY_ALPHA) * tvis[tumor_raw]).astype(np.uint8)
        tmask_u8 = (tumor_raw.astype(np.uint8) * 255)
        if (nh, nw) != (h, w):
            tvis = cv2.resize(tvis, (nw, nh), interpolation=cv2.INTER_LINEAR)
            tmask_u8 = cv2.resize(tmask_u8, (nw, nh),
                                  interpolation=cv2.INTER_NEAREST)

        self.pub_overlay.publish(self._to_msg(vis, header, "bgr8"))
        self.pub_mask.publish(self._to_msg(mask_u8, header, "mono8"))
        self.pub_tumor_overlay.publish(self._to_msg(tvis, header, "bgr8"))
        self.pub_tumor_mask.publish(self._to_msg(tmask_u8, header, "mono8"))

        # Publishing is never gated: the overlays stay live whether or not a
        # recording is running, so the view does not change mid-procedure.
        self._record_frame("margin", vis, mask_u8)
        self._record_frame("tumor", tvis, tmask_u8)
        if self.recording:
            log = self.frames_log
            if log is not None:
                try:
                    log.write(f"{self.n_recorded},{header.stamp.sec},{header.stamp.nanosec},"
                              f"{time.time():.6f}\n")
                except ValueError:     # closed by the release on the other thread
                    pass
            self.n_recorded += 1

    # ------------------------------------------------------------- recording

    def _begin_episode(self):
        """Start recording. Frames keep being published either way."""
        with self.lock:
            if self.recording:
                return False, "Already recording."
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            existing = ([d for d in os.listdir(self.output_root)
                         if d.startswith("episode_")]
                        if os.path.isdir(self.output_root) else [])
            n = len(existing) + 1
            self.episode_dir = os.path.join(self.output_root,
                                            f"episode_{n:03d}_{stamp}")
            os.makedirs(self.episode_dir, exist_ok=True)
            if not self.last_only:
                for tag in ("margin", "tumor"):
                    os.makedirs(os.path.join(self.episode_dir, tag, "masks"),
                                exist_ok=True)
            # Each episode starts from a clean temporal filter, so a recording
            # never inherits smoothing from whatever was on screen before it.
            self.ema = None
            self.writers = {}
            self.last = {}
            self.n_recorded = 0
            self.t_start = time.time()
            # Camera stamp of every recorded frame. This node and the toolhead
            # node drop different frames while busy, so mask numbers do not line
            # up between them; the stamps do.
            self.frames_log = open(os.path.join(self.episode_dir, "frames.csv"), "w")
            self.frames_log.write("frame,stamp_sec,stamp_nsec,wall_time\n")
            self.recording = True
        return True, f"Recording -> {self.episode_dir}"

    def _end_episode(self):
        """Stop recording and finalise the episode's files."""
        with self.lock:
            if not self.recording:
                return False, "Not recording."
            self.recording = False
            episode_dir = self.episode_dir
            writers, last = self.writers, dict(self.last)
            n, t0 = self.n_recorded, self.t_start
            self.writers = {}
            frames_log, self.frames_log = self.frames_log, None

        for w in writers.values():
            w.release()
        if frames_log is not None:
            frames_log.close()
        if n == 0:
            msg = "Stopped before any frame arrived; nothing was written."
            self.get_logger().warn(msg)
            return True, msg

        # The final frame of each stream is always written, whether or not the
        # per-frame sequences were kept.
        for tag, (vis, mask) in last.items():
            cv2.imwrite(os.path.join(episode_dir, f"{tag}_final_overlay.png"), vis)
            cv2.imwrite(os.path.join(episode_dir, f"{tag}_final_mask.png"), mask)

        dt = time.time() - t0
        meta = {
            "frames": int(n),
            "seconds": round(float(dt), 1),
            "last_only": self.last_only,
            "threshold": self.threshold,
            "ema_alpha": self.ema_alpha,
            "spatial_dilate": self.spatial_dilate,
            "proc_size": self.proc_size,
            "image_topic": self.image_topic,
        }
        with open(os.path.join(episode_dir, "episode.json"), "w") as f:
            json.dump(meta, f, indent=2)
        msg = f"Episode saved: {n} frames, {dt:.1f}s -> {episode_dir}"
        self.get_logger().info(msg)
        return True, msg

    def _record_frame(self, tag, vis, mask_u8):
        """Append one frame of a stream ("margin" or "tumor") to the episode.

        The newest frame of every stream is always retained so the final
        overlay and mask can be written when the episode ends, even under
        last_only where nothing per-frame is kept.
        """
        if not self.recording or self.episode_dir is None:
            return
        self.last[tag] = (vis, mask_u8)
        if self.last_only:
            return
        w = self.writers.get(tag)
        if w is None:
            hh, ww = vis.shape[:2]
            path = os.path.join(self.episode_dir, f"{tag}_overlay.mp4")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                (ww, hh))
            if not w.isOpened():
                self.get_logger().error(f"Could not open {path} for writing.")
                return
            self.writers[tag] = w
        w.write(vis)
        cv2.imwrite(os.path.join(self.episode_dir, tag, "masks",
                                 f"mask_{self.n_recorded:06d}.png"), mask_u8)

    def _on_joint(self, msg):
        """Bracket a recording with the Virtuoso's cut button.

        Same logic as the toolhead-following node: the joint state carries
        several entries and only TOOL_JOINT_NAME reports the button, reading
        1.0 while it is held. Pressing starts a recording, releasing ends it
        and finalises the files, and each press is a separate episode.

        A press acts immediately, but a release must persist for
        TOOL_RELEASE_HOLD before the episode ends: the signal can dip for a
        sample or two mid-cut, and acting on that would split one recording
        into several partial ones.
        """
        try:
            idx = list(msg.name).index(TOOL_JOINT_NAME)
        except ValueError:
            if not self._tool_seen:
                self._tool_seen = True
                self.get_logger().error(
                    f"No joint named '{TOOL_JOINT_NAME}' on {self.tool_topic}. "
                    f"Found: {list(msg.name)}. The tool trigger is inactive.")
            return
        if idx >= len(msg.position):
            return

        held = float(msg.position[idx]) >= TOOL_PRESSED

        if not self._tool_seen:
            self._tool_seen = True
            self.get_logger().info(
                f"Tool button detected on {self.tool_topic} "
                f"(currently {'held' if held else 'released'}).")
            if held:
                # Already held at startup: the cut is part-way through, so wait
                # for a fresh press rather than record a partial episode. The
                # held flag stays False, so the eventual release ends nothing.
                self._tool_ignore = True
                self.get_logger().warn(
                    "Button already held at startup; waiting for it to be "
                    "released before the first recording.")
                return

        if self._tool_ignore:
            if held:
                return
            self._tool_ignore = False
            self.get_logger().info("Button released; ready for the next cut.")
            return

        if held:
            self._tool_release_t = None
            if not self._tool_held:
                self._tool_held = True
                ok, message = self._begin_episode()
                self.get_logger().info(f"Tool pressed -> {message}")
            return

        if self._tool_held:
            now = time.time()
            if self._tool_release_t is None:
                self._tool_release_t = now
                return
            if now - self._tool_release_t < TOOL_RELEASE_HOLD:
                return
            self._tool_held = False
            self._tool_release_t = None
            ok, message = self._end_episode()
            self.get_logger().info(f"Tool released -> {message}")

    # --------------------------------------------------------------- helpers

    def _report(self):
        if self.n_received == 0:
            self.get_logger().warn(
                f"No frames received on {self.image_topic}. Is the camera "
                f"publishing? Check with: ros2 topic hz {self.image_topic}")
            return
        rec = f", recording {self.n_recorded}" if self.recording else ""
        self.get_logger().info(
            f"{self.n_processed} frames processed, {self.n_dropped} dropped{rec}.")

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
