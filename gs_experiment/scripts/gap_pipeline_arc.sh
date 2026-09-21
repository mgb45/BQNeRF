#!/bin/bash
# The arc half of the gap row: the four captures no trajectory prefix can gap.
#
# FINDINGS s26 found room, truck, train and drjohnson rejected by the
# admission rule, and that halving their training set does not rescue them:
# they are multi-loop captures, so any contiguous PREFIX already covers every
# viewing direction. A gap in such a capture is not a moment in time, it is a
# DIRECTION -- a loop left incomplete, which is the most ordinary partial
# capture there is. A 30 degree arc gives them 25.5 to 31.8 degrees of median
# isolation, in the same range as the trajectory cells (kitchen 24.2,
# stump 22.1), so they are measured on the same scale as the rest.
#
# 30 degrees is the SMALLEST width tested that admits all four, not a tuned
# one; at 45 degrees `room` blows up to 137 degrees -- held-out views never
# observed from any direction at all, which is not a capture anyone would
# make. See EPISTEMIC_PLAN.md for the amendment this records.
set -u
ROOT=/home/michael/Documents/Projects/BQNeRF
PY=$ROOT/.venv-gsplat/bin/python
GSU=$ROOT/third_party/GS-U
GAPS=$ROOT/gs_experiment/local_runs/gapscenes

export PATH="$ROOT/.venv-gsplat/bin:/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_HOME=/usr/local/cuda-12.3
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"

JOBS=("room|images_2" "truck|images" "train|images" "drjohnson|images")

stage () { local n="$1" wd="$2"; shift 2
  ( cd "$wd" && PYTHONPATH=. /usr/bin/time -f "peak_RSS=%M kB elapsed=%E" "$@" ) \
      > "$LOGD/$n.log" 2>&1
  local rc=$?
  echo "  [$n] rc=$rc $(grep -ao 'peak_RSS=[0-9]* kB elapsed=[0-9:.]*' "$LOGD/$n.log"|tail -1)"
  [ $rc -eq 0 ] || { echo "    tail:"; tr '\r' '\n' < "$LOGD/$n.log" | grep -vE '^$' | tail -5 | sed 's/^/      /'; }
  return $rc
}

for spec in "${JOBS[@]}"; do
  L="${spec%%|*}"; IMG="${spec#*|}"
  SRC=$GAPS/${L}_cone
  M=$ROOT/gs_experiment/local_runs/arc_$L
  LOGD=$ROOT/gs_experiment/local_runs/pipeline_logs/${L}_arc; mkdir -p "$LOGD"
  echo "######## $L ########"; date
  [ -d "$SRC" ] || { echo "  !! no arc scene at $SRC, SKIP"; continue; }
  if [ ! -f "$M/point_cloud/iteration_30000/point_cloud.ply" ]; then
    rm -rf "$M"
    stage train "$GSU" $PY train.py -s $SRC -i $IMG -m $M --eval --iterations 30000 --quiet || continue
    rm -rf "$M/point_cloud/iteration_7000"
  fi
  stage fit_theirs "$GSU" $PY train_errors.py -m $M -s $SRC -i $IMG --eval --load_iteration 30000 || continue
  stage fit_ours "$ROOT" $PY gs_experiment/scripts/our_uncertainty_for_3dgs_model.py \
      -m $M -s $SRC -i $IMG --prior population || continue
  stage reduce "$ROOT" $PY gs_experiment/scripts/reduce_per_view.py "$M" || continue
  for F in error_masks error_masks_ours; do
    stage score_$F "$GSU" $PY uncertainty_metrics.py -i $M -f $F --split test
    cp "$M/renders/eval/test/uncertainty_metrics__$F.json" \
       "$ROOT/gs_experiment/results/per_view/arc_${L}__$F.json" 2>/dev/null
  done
  rm -rf "$M/renders/test"
  echo "  DONE $L"; df -h /home/michael | tail -1
done
echo "===== ARC ROW COMPLETE ====="; date
