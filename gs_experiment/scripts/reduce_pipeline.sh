#!/bin/bash
# Recover the per-view record that scene_pipeline.sh threw away.
#
# Its last line, `rm -rf $M/renders/test`, saved 8 GB and destroyed the only
# evidence of per-VIEW behaviour on all thirteen benchmark scenes. That is the
# empty cell of the 2x2 in EPISTEMIC_PLAN.md. The checkpoints are still here,
# so this costs the two fit stages (~4 min/scene) rather than a 23-min retrain
# -- which is why it has to happen BEFORE any checkpoint cleanup.
#
# Per scene: refit both uncertainties, reduce to per-view scalars (a few KB,
# kept forever), rescore to check the refit reproduces the archived numbers,
# then drop the maps again unless the scene is a designated figure hero.
set -u
ROOT=/home/michael/Documents/Projects/BQNeRF
PY=$ROOT/.venv-gsplat/bin/python
GSU=$ROOT/third_party/GS-U
RAW=$ROOT/gs_experiment/local_runs/mipnerf360_raw
TDB=$ROOT/gs_experiment/local_runs/tandt_db
HEROES=" drjohnson garden "   # keep full maps: one we win, one we lose

export PATH="$ROOT/.venv-gsplat/bin:/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_HOME=/usr/local/cuda-12.3
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"

JOBS=(
  "bicycle|images_4|$RAW/bicycle"      "flowers|images_4|$RAW/flowers"
  "garden|images_4|$RAW/garden"        "stump|images_4|$RAW/stump"
  "treehill|images_4|$RAW/treehill"    "room|images_2|$RAW/room"
  "counter|images_2|$RAW/counter"      "kitchen|images_2|$RAW/kitchen"
  "bonsai|images_2|$RAW/bonsai"        "truck|images|$TDB/tandt/truck"
  "train|images|$TDB/tandt/train"      "drjohnson|images|$TDB/db/drjohnson"
  "playroom|images|$TDB/db/playroom"
)

stage () { local n="$1" wd="$2"; shift 2
  ( cd "$wd" && PYTHONPATH=. /usr/bin/time -f "peak_RSS=%M kB elapsed=%E" "$@" ) \
      > "$LOGD/$n.log" 2>&1
  local rc=$?
  echo "  [$n] rc=$rc $(grep -ao 'peak_RSS=[0-9]* kB elapsed=[0-9:.]*' "$LOGD/$n.log"|tail -1)"
  [ $rc -eq 0 ] || { echo "    tail:"; tr '\r' '\n' < "$LOGD/$n.log" | grep -vE '^$' | tail -5 | sed 's/^/      /'; }
  return $rc
}

for spec in "${JOBS[@]}"; do
  L="${spec%%|*}"; rest="${spec#*|}"; IMG="${rest%%|*}"; SRC="${rest#*|}"
  M=$ROOT/gs_experiment/local_runs/gsu_$L
  LOGD=$ROOT/gs_experiment/local_runs/pipeline_logs/${L}_reduce; mkdir -p "$LOGD"
  echo "######## $L ########"; date
  [ -f "$M/point_cloud/iteration_30000/point_cloud.ply" ] || { echo "  !! no checkpoint, SKIP"; continue; }
  stage fit_theirs "$GSU" $PY train_errors.py -m $M -s $SRC -i $IMG --eval --load_iteration 30000 || continue
  stage fit_ours "$ROOT" $PY gs_experiment/scripts/our_uncertainty_for_3dgs_model.py \
      -m $M -s $SRC -i $IMG --prior population || continue
  stage reduce "$ROOT" $PY gs_experiment/scripts/reduce_per_view.py "$M" || continue
  for F in error_masks error_masks_ours; do
    stage rescore_$F "$GSU" $PY uncertainty_metrics.py -i $M -f $F --split test
    cp "$M/renders/eval/test/uncertainty_metrics__$F.json" \
       "$ROOT/gs_experiment/results/per_view/gsu_${L}__rescore__$F.json" 2>/dev/null
  done
  case "$HEROES" in *" $L "*) echo "  keeping maps (figure hero)";; *) rm -rf "$M/renders/test";; esac
  echo "  DONE $L"; df -h /home/michael | tail -1
done
echo "===== REDUCE SWEEP COMPLETE ====="; date
