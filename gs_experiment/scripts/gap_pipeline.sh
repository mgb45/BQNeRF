#!/bin/bash
# The gap row of EPISTEMIC_PLAN.md's 2x2, on every scene the admission rule
# lets through.
#
# Admission is >= 5 degrees median angular isolation, computed by
# build_colmap_gap_scene.py before any training and therefore blind to
# results. Nine of thirteen benchmark scenes qualify at the pre-registered
# train_fraction = 0.7; room, truck, train and drjohnson do not, and halving
# their training set does not rescue them (FINDINGS s26). Three of the nine
# were already run for section 21 and only need their per-view maps
# regenerated, so they skip training.
set -u
ROOT=/home/michael/Documents/Projects/BQNeRF
PY=$ROOT/.venv-gsplat/bin/python
GSU=$ROOT/third_party/GS-U
GAPS=$ROOT/gs_experiment/local_runs/gapscenes
HEROES=" garden "    # keep full maps for the figure: worst within-view, wins per-view

export PATH="$ROOT/.venv-gsplat/bin:/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_HOME=/usr/local/cuda-12.3
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"

# scene | images_dir   (resolution follows their full_eval.py: outdoor
# images_4, indoor images_2, Deep Blending the shipped full-res images)
JOBS=("kitchen|images_2" "counter|images_2" "bonsai|images_2"
      "bicycle|images_4" "flowers|images_4" "garden|images_4"
      "stump|images_4"   "treehill|images_4" "playroom|images")

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
  SRC=$GAPS/${L}_trajectory
  M=$ROOT/gs_experiment/local_runs/slam_$L
  LOGD=$ROOT/gs_experiment/local_runs/pipeline_logs/${L}_gap; mkdir -p "$LOGD"
  echo "######## $L ########"; date
  [ -d "$SRC" ] || { echo "  !! no gap scene at $SRC, SKIP"; continue; }
  if [ ! -f "$M/point_cloud/iteration_30000/point_cloud.ply" ]; then
    rm -rf "$M"
    stage train "$GSU" $PY train.py -s $SRC -i $IMG -m $M --eval --iterations 30000 --quiet || continue
    rm -rf "$M/point_cloud/iteration_7000"
  else
    echo "  [train] reusing existing checkpoint"
  fi
  stage fit_theirs "$GSU" $PY train_errors.py -m $M -s $SRC -i $IMG --eval --load_iteration 30000 || continue
  stage fit_ours "$ROOT" $PY gs_experiment/scripts/our_uncertainty_for_3dgs_model.py \
      -m $M -s $SRC -i $IMG --prior population || continue
  stage reduce "$ROOT" $PY gs_experiment/scripts/reduce_per_view.py "$M" || continue
  for F in error_masks error_masks_ours; do
    stage score_$F "$GSU" $PY uncertainty_metrics.py -i $M -f $F --split test
    cp "$M/renders/eval/test/uncertainty_metrics__$F.json" \
       "$ROOT/gs_experiment/results/per_view/slam_${L}__$F.json" 2>/dev/null
  done
  case "$HEROES" in *" $L "*) echo "  keeping maps (figure hero)";; *) rm -rf "$M/renders/test";; esac
  echo "  DONE $L"; df -h /home/michael | tail -1
done
echo "===== GAP ROW COMPLETE ====="; date
