#!/bin/bash
# Capacity withheld: same scenes, same views, fewer splats.
#
# The companion to the coverage-gap experiment. There, information was
# removed from the DATA; here it is removed from the REPRESENTATION, with the
# train/test split untouched so capacity is the only varying quantity.
#
# EPISTEMIC_PLAN.md section 5 registers the prediction before this runs: our
# sigma should FALL as splats are removed while error RISES, because D_i
# accumulates squared compositing weight over pixels and fewer splats means
# each survivor covers more of them. If so, the posterior does not see
# representational inadequacy -- which is what the paper claims, made
# falsifiable rather than asserted.
set -u
ROOT=/home/michael/Documents/Projects/BQNeRF
PY=$ROOT/.venv-gsplat/bin/python
GSU=$ROOT/third_party/GS-U
RAW=$ROOT/gs_experiment/local_runs/mipnerf360_raw
TDB=$ROOT/gs_experiment/local_runs/tandt_db

export PATH="$ROOT/.venv-gsplat/bin:/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_HOME=/usr/local/cuda-12.3
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"

# scene | images_dir | source
SCENES=("bonsai|images_2|$RAW/bonsai" "garden|images_4|$RAW/garden" "playroom|images|$TDB/db/playroom")
# higher threshold -> fewer splats. 0.0002 is the 3DGS default.
THRESHOLDS=(0.0002 0.0008 0.0032 0.0128)

stage () { local n="$1" wd="$2"; shift 2
  ( cd "$wd" && PYTHONPATH=. /usr/bin/time -f "peak_RSS=%M kB elapsed=%E" "$@" ) > "$LOGD/$n.log" 2>&1
  local rc=$?
  echo "  [$n] rc=$rc $(grep -ao 'peak_RSS=[0-9]* kB elapsed=[0-9:.]*' "$LOGD/$n.log"|tail -1)"
  [ $rc -eq 0 ] || { echo "    tail:"; tr '\r' '\n' < "$LOGD/$n.log" | grep -vE '^$' | tail -5 | sed 's/^/      /'; }
  return $rc
}

for spec in "${SCENES[@]}"; do
  L="${spec%%|*}"; rest="${spec#*|}"; IMG="${rest%%|*}"; SRC="${rest#*|}"
  for T in "${THRESHOLDS[@]}"; do
    TAG="${L}_t${T#0.}"
    M=$ROOT/gs_experiment/local_runs/cap_$TAG
    LOGD=$ROOT/gs_experiment/local_runs/pipeline_logs/${TAG}_cap; mkdir -p "$LOGD"
    echo "######## $L  densify_grad_threshold=$T ########"; date
    if [ ! -f "$M/point_cloud/iteration_30000/point_cloud.ply" ]; then
      rm -rf "$M"
      stage train "$GSU" $PY train.py -s $SRC -i $IMG -m $M --eval --iterations 30000 \
          --densify_grad_threshold $T --quiet || continue
      rm -rf "$M/point_cloud/iteration_7000"
    fi
    # splat count straight off the ply header -- the quantity being swept
    N=$($PY -c "
from plyfile import PlyData
print(len(PlyData.read('$M/point_cloud/iteration_30000/point_cloud.ply')['vertex']))" 2>/dev/null)
    echo "  splats=$N"
    stage fit_theirs "$GSU" $PY train_errors.py -m $M -s $SRC -i $IMG --eval --load_iteration 30000 || continue
    stage fit_ours "$ROOT" $PY gs_experiment/scripts/our_uncertainty_for_3dgs_model.py \
        -m $M -s $SRC -i $IMG --prior population || continue
    stage reduce "$ROOT" $PY gs_experiment/scripts/reduce_per_view.py "$M" || continue
    for F in error_masks error_masks_ours; do
      stage score_$F "$GSU" $PY uncertainty_metrics.py -i $M -f $F --split test
      cp "$M/renders/eval/test/uncertainty_metrics__$F.json" \
         "$ROOT/gs_experiment/results/per_view/cap_${TAG}__$F.json" 2>/dev/null
    done
    echo "{\"scene\":\"$L\",\"threshold\":$T,\"splats\":$N}" \
        > "$ROOT/gs_experiment/results/per_view/cap_${TAG}__capacity.json"
    rm -rf "$M/renders/test" "$M/point_cloud"
    echo "  DONE $TAG"; df -h /home/michael | tail -1
  done
done
echo "===== CAPACITY SWEEP COMPLETE ====="; date
