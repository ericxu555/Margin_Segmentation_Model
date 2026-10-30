#!/usr/bin/env bash
# One command for a live session: both models, the operator UI, all recorded by the cut button.
#
#   ./run_all.sh              toolhead node + margin node + UI
#   ./run_all.sh --bag        also record the raw ROS bag (tools/record_session.sh)
#   ./run_all.sh --no-ui      nodes only
#   ./run_all.sh --last-only  margin node keeps only each episode's final overlay and mask
#   ./run_all.sh --no-organize  leave the outputs where the nodes wrote them
#
# On Ctrl-C the session is gathered into ~/resection_data/session_<stamp>/, one
# cut_NN folder per cut with toolhead/ and margin/ side by side and a
# session_summary.csv (tools/organize_session.py; DATA_DIR overrides the place).
#
#   toolhead node   ros_toolhead_pose_node.py: visual-servoing tip (pose_only), rectified
#                   camera, experiment logging on (raw frames, tips.csv, a mask every
#                   frame, calibration per episode), no extra tip smoothing
#                   -> /toolhead/overlay, /toolhead/status, /toolhead/mask
#   margin node     ros_margin_node.py, recording on the cut button
#                   -> /margin/overlay, /margin/mask, /tumor/overlay, /tumor/mask
#   UI              run_ui.py --mode live (reads the topics above)
#
# Start with the tool out of view, so the toolhead node can capture its baseline.
# Hold the cut button for a cut; release after the tool has left the tumor.
# Ctrl-C here stops everything; a cut in progress is still written out.

set -u

TOOLHEAD_DIR="${TOOLHEAD_DIR:-$HOME/Toolhead_Tracking}"
MARGIN_DIR="${MARGIN_DIR:-$HOME/Margin_Segmentation_Model}"
VENV="${VENV:-$TOOLHEAD_DIR/toolhead_env}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/ves_camera/image_rect}"

DATA_DIR="${DATA_DIR:-$HOME/resection_data}"
LAUNCH_EPOCH="$(date +%s)"

LAST_ONLY="false"; BAG="false"; UI="true"; ORGANIZE="true"
for arg in "$@"; do
    case "$arg" in
        --last-only) LAST_ONLY="true" ;;
        --bag) BAG="true" ;;
        --no-ui) UI="false" ;;
        --no-organize) ORGANIZE="false" ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

for d in "$TOOLHEAD_DIR" "$MARGIN_DIR" "$VENV"; do
    [ -e "$d" ] || { echo "Not found: $d" >&2; exit 1; }
done

# ROS 2 for the nodes' rclpy and for `ros2 bag`, then the Python environment.
if [ -z "${ROS_DISTRO:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

LOG_DIR="${LOG_DIR:-$TOOLHEAD_DIR/run_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
TOOLHEAD_LOG="$LOG_DIR/toolhead_$STAMP.log"
MARGIN_LOG="$LOG_DIR/margin_$STAMP.log"
UI_LOG="$LOG_DIR/ui_$STAMP.log"
BAG_LOG="$LOG_DIR/bag_$STAMP.log"

echo "Free disk:"
df -h "$TOOLHEAD_DIR" "$MARGIN_DIR" | sed 's/^/  /'

echo "Starting toolhead-following node ..."
( cd "$TOOLHEAD_DIR" && exec python3 edge_anchored_tracking/ros_toolhead_pose_node.py --ros-args \
    -p tip_source:=pose_only -p trigger_mode:=tool \
    -p image_topic:="$IMAGE_TOPIC" -p camera_model:=rect \
    -p show_distance_overlay:=false -p tip_smooth:=1 \
    -p experiment:=true -p snapshot_every:=1 ) >"$TOOLHEAD_LOG" 2>&1 &
TOOLHEAD_PID=$!

echo "Starting margin + tumor node ..."
( cd "$MARGIN_DIR" && exec python3 ros_margin_node.py --ros-args \
    -p image_topic:="$IMAGE_TOPIC" -p trigger_mode:=tool \
    -p last_only:="$LAST_ONLY" ) >"$MARGIN_LOG" 2>&1 &
MARGIN_PID=$!

UI_PID=""
if [ "$UI" = "true" ]; then
    echo "Starting the operator UI ..."
    ( cd "$TOOLHEAD_DIR" && exec python3 edge_anchored_tracking/run_ui.py --mode live ) >"$UI_LOG" 2>&1 &
    UI_PID=$!
fi

BAG_PID=""
if [ "$BAG" = "true" ]; then
    echo "Recording the raw bag ..."
    ( cd "$TOOLHEAD_DIR" && exec bash tools/record_session.sh ) >"$BAG_LOG" 2>&1 &
    BAG_PID=$!
fi

cleanup() {
    echo
    echo "Stopping ..."
    # SIGINT, as Ctrl-C would: the nodes write out a cut in progress and the
    # bag recorder closes its file cleanly.
    kill -INT $TOOLHEAD_PID $MARGIN_PID $UI_PID $BAG_PID 2>/dev/null
    wait $TOOLHEAD_PID $MARGIN_PID $UI_PID $BAG_PID 2>/dev/null
    echo "Logs: $LOG_DIR/*_$STAMP.log"
    if [ "$ORGANIZE" = "true" ]; then
        echo "Organizing the session into $DATA_DIR ..."
        python3 "$TOOLHEAD_DIR/tools/organize_session.py" --after "$LAUNCH_EPOCH" \
            --toolhead-out "$TOOLHEAD_DIR/Output" --margin-out "$MARGIN_DIR/Output" \
            --dest "$DATA_DIR" --logs "$TOOLHEAD_LOG" "$MARGIN_LOG" "$UI_LOG" "$BAG_LOG" \
            || echo "Organizing failed; the data is still in both Output/ folders." >&2
    fi
    exit 0
}
trap cleanup INT TERM

cat <<EOF

The nodes are loading their models; the first frame takes a few seconds.

  session data    $( [ "$ORGANIZE" = "true" ] && echo "$DATA_DIR/session_*/cut_NN_*/ (gathered on Ctrl-C)" || echo "$TOOLHEAD_DIR/Output and $MARGIN_DIR/Output" )
  margin          $( [ "$LAST_ONLY" = "true" ] && echo "final frame only" || echo "videos + per-frame masks" )
  raw bag         $( [ "$BAG" = "true" ] && echo "$HOME/toolhead_bags/session_*" || echo "off (add --bag)" )

Hold the cut button for a cut; release after the tool has left the tumor.
Wait a few seconds after the last release, then Ctrl-C here to stop everything.

Following the node logs:
EOF

tail -n +1 -f "$TOOLHEAD_LOG" "$MARGIN_LOG" &
TAIL_PID=$!

# If a node dies, stop following and say so rather than sitting silent.
while kill -0 "$TOOLHEAD_PID" 2>/dev/null && kill -0 "$MARGIN_PID" 2>/dev/null; do
    sleep 2
done
kill "$TAIL_PID" 2>/dev/null
echo
echo "A node exited. Check the logs above." >&2
cleanup
