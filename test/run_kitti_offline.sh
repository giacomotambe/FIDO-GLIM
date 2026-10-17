#!/usr/bin/env bash
# Runs the offline dynamic-rejection harness on one sequence in SemanticKITTI layout stored on the NAS
# (SemanticKITTI by default; other datasets via DATA_ROOT/RESULTS_ROOT, e.g. DOALS/kitti).
# The container cannot read the gvfs SMB mount, so the sequence is copied to a temporary local
# stage, processed, evaluated, and the stage is deleted. Results go back to the NAS.
#
# Usage: run_kitti_offline.sh <seq> <tag> [config_dir] [modes...]
#   seq        : e.g. 08 (or a DOALS name such as small_town_simulation)
#   tag        : name of the result folder (results/offline_harness/<tag>_seq<seq>/)
#   config_dir : glim config dir relative to glim/ (default: config); several comma-separated
#                dirs run on the same staged copy, results named <config>_<mode>
#   modes      : any of exact ros ros_cv ros_cv2 lag1 lag2 (default: exact ros)
# Env: DATA_ROOT    dir with sequences/ and poses/ (default: NAS semanticKITTI/dataset)
#      RESULTS_ROOT where <tag>_seq<seq>/ is written (default: NAS semanticKITTI/results/offline_harness)
#      EVAL_ARGS    extra arguments for kitti_dynamic_eval.py (e.g. "--ignore 0")
#      BUILD_DIR    container build dir of the harness and libglim (default /home/ros/build/glim)
#      EVAL_SPLIT   optional file of frame numbers: also writes <key>_split.json restricted to them
set -euo pipefail

SEQ="$1"; TAG="$2"; CFG="${3:-config}"; shift $(( $# < 3 ? $# : 3 ))
MODES=("$@"); [ ${#MODES[@]} -eq 0 ] && MODES=(exact ros)

GLIM=/home/punk-opc/FIDO_glim/glim
NAS="/run/user/1000/gvfs/smb-share:server=konstanz.local,share=kloster%20reichenau/GT/semanticKITTI"
CONTAINER=fido_glim-jo-zotac-1
PY=/home/punk-opc/evo_env/bin/python3
RUN_ID="${TAG}_seq${SEQ}"
STAGE_HOST="$GLIM/.kitti_eval/stage/$RUN_ID"          # dataset copy + raw predictions (temporary)
STAGE_CT="/home/ros/src/glim/.kitti_eval/stage/$RUN_ID"
DATA="${DATA_ROOT:-$NAS/dataset}"
OUT="${RESULTS_ROOT:-$NAS/results/offline_harness}/$RUN_ID"

[ -d "$DATA/sequences/$SEQ" ] || { echo "sequence $SEQ not found on NAS"; exit 1; }
cleanup() { rm -rf "$STAGE_HOST"; }
trap cleanup EXIT

echo "[$RUN_ID] staging sequence from NAS..."
mkdir -p "$STAGE_HOST/seq" "$STAGE_HOST/log"
cp -r "$DATA/sequences/$SEQ/." "$STAGE_HOST/seq/"
cp "$DATA/poses/$SEQ.txt" "$STAGE_HOST/poses.txt"

IFS=',' read -ra CFGS <<< "$CFG"
key() {  # result name for config $1 and mode $2 (just the mode when a single config is given)
  if [ ${#CFGS[@]} -eq 1 ]; then echo "$2"; else echo "$(basename "$1")_$2"; fi
}

echo "[$RUN_ID] running modes: ${MODES[*]} (configs: ${CFGS[*]})"
CMD="source /opt/ros/jazzy/setup.bash; source /home/ros/install/setup.bash; export LD_LIBRARY_PATH=${BUILD_DIR:-/home/ros/build/glim}:\$LD_LIBRARY_PATH;"   # use the libglim just built, not the installed one
for c in "${CFGS[@]}"; do
  for m in "${MODES[@]}"; do
    case "$m" in
      exact|ros|ros_cv|ros_cv2) ARGS="--pose_mode $m" ;;
      lag*)      ARGS="--pose_mode lag --lag ${m#lag}" ;;
      *) echo "unknown mode $m"; exit 1 ;;
    esac
    k=$(key "$c" "$m")
    CMD+=" ${BUILD_DIR:-/home/ros/build/glim}/kitti_dynamic_eval --config /home/ros/src/glim/$c --seq $STAGE_CT/seq"
    CMD+=" --poses $STAGE_CT/poses.txt --out $STAGE_CT/pred_$k $ARGS > $STAGE_CT/log/$k.log 2>&1 &"
  done
done
docker exec "$CONTAINER" bash -c "$CMD wait"

mkdir -p "$OUT"
# From here on a failed evaluation must not abort the run: whatever was computed is still saved.
set +e
for c in "${CFGS[@]}"; do
  for m in "${MODES[@]}"; do
    k=$(key "$c" "$m")
    echo "[$RUN_ID] evaluating $k..."
    "$PY" "$GLIM/test/kitti_dynamic_eval.py" --seq "$STAGE_HOST/seq" --pred "$STAGE_HOST/pred_$k" \
      --json "$STAGE_HOST/$k.json" ${EVAL_ARGS:-} 2>/dev/null | tee "$STAGE_HOST/log/${k}_eval.txt"
    if [ -n "${EVAL_SPLIT:-}" ]; then   # same metrics restricted to the frames listed in EVAL_SPLIT
      "$PY" "$GLIM/test/kitti_dynamic_eval.py" --seq "$STAGE_HOST/seq" --pred "$STAGE_HOST/pred_$k" --frames "$EVAL_SPLIT" \
        --json "$STAGE_HOST/${k}_split.json" ${EVAL_ARGS:-} > "$STAGE_HOST/log/${k}_split_eval.txt" 2>/dev/null
      [ -d "$STAGE_HOST/pred_$k/rej" ] && "$PY" "$GLIM/test/kitti_dynamic_eval.py" --seq "$STAGE_HOST/seq" --pred "$STAGE_HOST/pred_$k/rej" \
        --frames "$EVAL_SPLIT" --json "$STAGE_HOST/${k}_rej_split.json" ${EVAL_ARGS:-} > "$STAGE_HOST/log/${k}_rej_split_eval.txt" 2>/dev/null
    fi
    if [ -n "$(ls "$STAGE_HOST/pred_$k/rej" 2>/dev/null | head -1)" ]; then   # labels on the rejection's own points
      echo "[$RUN_ID] evaluating ${k}_rej..."
      "$PY" "$GLIM/test/kitti_dynamic_eval.py" --seq "$STAGE_HOST/seq" --pred "$STAGE_HOST/pred_$k/rej" \
        --json "$STAGE_HOST/${k}_rej.json" ${EVAL_ARGS:-} 2>/dev/null | tee "$STAGE_HOST/log/${k}_rej_eval.txt"
    fi
  done
done

echo "[$RUN_ID] saving results to NAS..."
cp "$STAGE_HOST"/*.json "$OUT/"
cp -r "$STAGE_HOST/log" "$OUT/"
for c in "${CFGS[@]}"; do
  if [ ${#CFGS[@]} -eq 1 ]; then cp -r "$GLIM/$c" "$OUT/config"; else cp -r "$GLIM/$c" "$OUT/config_$(basename "$c")"; fi
done
git -C "$GLIM" rev-parse HEAD > "$OUT/glim_commit.txt"
git -C "$GLIM" diff --stat >> "$OUT/glim_commit.txt"
echo "[$RUN_ID] done -> $OUT"
