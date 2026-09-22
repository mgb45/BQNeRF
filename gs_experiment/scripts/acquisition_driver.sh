#!/bin/bash
# The SLAM-shaped experiment on the two scenes where a coverage gap was shown
# to matter: truck (FINDINGS s28, inverts completely under an arc hold-out)
# and drjohnson (the Deep Blending win, and a walked-through interior, which
# is the capture style an agent actually produces).
set -u
ROOT=/home/michael/Documents/Projects/BQNeRF
PY=$ROOT/.venv-gsplat/bin/python
TDB=$ROOT/gs_experiment/local_runs/tandt_db
export PATH="$ROOT/.venv-gsplat/bin:/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin"
export CUDA_HOME=/usr/local/cuda-12.3
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions"
cd $ROOT
for spec in "truck|$TDB/tandt/truck" "drjohnson|$TDB/db/drjohnson"; do
  L="${spec%%|*}"; SRC="${spec#*|}"
  echo "######## $L ########"; date
  PYTHONPATH=. $PY gs_experiment/scripts/run_acquisition_loop.py "$L" \
      --source "$SRC" -i images --image_dirs images \
      --rounds 4 --batch 8 --reach 20 \
      --cheap_iters 3000 --cheap_thresh 0.0008 --full_iters 30000 \
      --random_seeds 0 1 2 \
      --out $ROOT/gs_experiment/local_runs/acq_$L || echo "### $L FAILED"
  cp $ROOT/gs_experiment/local_runs/acq_$L/acquisition.json \
     $ROOT/gs_experiment/results/acquisition_$L.json 2>/dev/null
  rm -rf $ROOT/gs_experiment/local_runs/acq_$L/final_*_src $ROOT/gs_experiment/local_runs/acq_$L/*/round*
  df -h /home/michael | tail -1
done
echo "===== ACQUISITION COMPLETE ====="; date
