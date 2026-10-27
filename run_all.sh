#!/usr/bin/env bash
# Run all three models against the live camera, recorded by the cut button.
#
#   ./run_all.sh              full recording: videos + per-frame masks
#   ./run_all.sh --last-only  only each episode's final overlay and mask
#
# Two processes, not three: the margin node already runs binary10swa as its
# Stage 1 to gate the margin, so the tumor segmentation comes from the same
# forward pass rather than a second copy of the weights.
#
#   toolhead node   /toolhead/overlay  /toolhead/mask   cumulative resection
#   margin node     /margin/overlay    /margin/mask     margin
#                   /tumor/overlay     /tumor/mask      tumor (binary10swa)
#
# Both watch the same joint topic, so one press of the cut button brackets an
# episode in both and each writes to its own output directory. Ctrl-C stops
# both together.

set -u

TOOLHEAD_DIR="${TOOLHEAD_DIR:-$HOME/Toolhead_Tracking}"
MARGIN_DIR="${MARGIN_DIR:-$HOME/Margin_Segmentation_Model}"
VENV="${VENV:-$TOOLHEAD_DIR/toolhead_env}"

LAST_ONLY="false"
for arg in "$@"; do
    case "$arg" in
        --last-only) LAST_ONLY="true" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

for d in "$TOOLHEAD_DIR" "$MARGIN_DIR" "$VENV"; do
    [ -e "$d" ] || { echo "Not found: $d" >&2; exit 1; }
done

# shellcheck disable=SC1091
source "$VENV/bin/activate"

LOG_DIR="${LOG_DIR:-$PWD/run_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
TOOLHEAD_LOG="$LOG_DIR/toolhead_$STAMP.log"
MARGIN_LOG="$LOG_DIR/margin_$STAMP.log"

echo "Starting toolhead-following node ..."
( cd "$TOOLHEAD_DIR" && exec python3 edge_anchored_tracking/ros_toolhead_node.py \
    --ros-args -p trigger_mode:=tool ) >"$TOOLHEAD_LOG" 2>&1 &
TOOLHEAD_PID=$!

echo "Starting margin + tumor node ..."
( cd "$MARGIN_DIR" && exec python3 ros_margin_node.py \
    --ros-args -p trigger_mode:=tool -p last_only:="$LAST_ONLY" ) >"$MARGIN_LOG" 2>&1 &
MARGIN_PID=$!

cleanup() {
    echo
    echo "Stopping ..."
    # A running episode is finalised on SIGINT the same way Ctrl-C does it
    # interactively, so a recording in progress is still written out.
    kill -INT "$TOOLHEAD_PID" "$MARGIN_PID" 2>/dev/null
    wait "$TOOLHEAD_PID" "$MARGIN_PID" 2>/dev/null
    echo "Logs: $TOOLHEAD_LOG"
    echo "      $MARGIN_LOG"
}
trap cleanup INT TERM

cat <<EOF

Both nodes are loading their models; first inference takes a few seconds.

  recording   $( [ "$LAST_ONLY" = "true" ] && echo "final frame only" || echo "videos + per-frame masks" )
  outputs     $TOOLHEAD_DIR/Output/episode_*
              $MARGIN_DIR/Output/episode_*

Hold the cut button to record, release to save. Ctrl-C here stops both.

Following both logs:
EOF

tail -n +1 -f "$TOOLHEAD_LOG" "$MARGIN_LOG" &
TAIL_PID=$!

# If either node dies, stop following and report it rather than sitting silent.
while kill -0 "$TOOLHEAD_PID" 2>/dev/null && kill -0 "$MARGIN_PID" 2>/dev/null; do
    sleep 2
done
kill "$TAIL_PID" 2>/dev/null
echo
echo "A node exited. Check the logs above." >&2
cleanup
