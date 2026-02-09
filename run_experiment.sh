#!/bin/bash

TRAIN_SCRIPT="train.py"

run_model() {
  local SLUG="$1"
  local DISPLAY="$2"
  local LAYERS="$3"
  local SEQ_LEN="$4"
  local BATCH="$5"
  local STRATEGY="$6"
  local LAGS="$7"
  local PRE_NORM="$8"

  echo "===================================================="
  echo ">>> RUNNING: $DISPLAY"
  echo "===================================================="

  local cmd=(
    python "$TRAIN_SCRIPT"
    --name "$SLUG"
    --num-layers "$LAYERS"
    --max-context-len "$SEQ_LEN"
    --batch-size "$BATCH"
    --tensor-lifting-strategy "$STRATEGY"
  )

  if [[ -n "$LAGS" ]]; then
    local lags_array=()
    read -r -a lags_array <<<"$LAGS"
    cmd+=(--lags "${lags_array[@]}")
  fi

  if [[ "$PRE_NORM" == "true" ]]; then
    cmd+=(--pre-norm)
  fi

  "${cmd[@]}"
  echo -e "Done.\n"
}

case "$1" in
attn-6l-128)
  run_model "attn_6l_128" "Attention | 6 Layers | Seq 128" 6 128 32 "attention"
  ;;
grass-6l-128)
  run_model "grass_6l_128" "Grassmann | 6 Layers | Seq 128" 6 128 32 "grassmann" "1 2 4 8 12 16"
  ;;
6l-128)
  "$0" attn-6l-128
  "$0" grass-6l-128
  ;;

attn-12l-256)
  run_model "attn_12l_256" "Attention | 12 Layers | Seq 256" 12 256 16 "attention"
  ;;
grass-12l-256)
  run_model "grass_12l_256" "Grassmann | 12 Layers | Seq 256" 12 256 16 "grassmann" "1 1 2 2 4 4 8 8 12 12 16 16"
  ;;
12l-256)
  "$0" attn-12l-256
  "$0" grass-12l-256
  ;;

attn-6l-128-prenorm)
  run_model "attn_6l_128_prenorm" "Attention (Pre-Norm) | 6 Layers | Seq 128" 6 128 32 "attention" "" "true"
  ;;
grass-6l-128-prenorm)
  run_model "grass_6l_128_prenorm" "Grassmann (Pre-Norm) | 6 Layers | Seq 128" 6 128 32 "grassmann" "1 2 4 8 12 16" "true"
  ;;
6l-128-prenorm)
  "$0" attn-6l-128-prenorm
  "$0" grass-6l-128-prenorm
  ;;

attn-12l-256-prenorm)
  run_model "attn_12l_256_prenorm" "Attention (Pre-Norm) | 12 Layers | Seq 256" 12 256 16 "attention" "" "true"
  ;;
grass-12l-256-prenorm)
  run_model "grass_12l_256_prenorm" "Grassmann (Pre-Norm) | 12 Layers | Seq 256" 12 256 16 "grassmann" "1 1 2 2 4 4 8 8 12 12 16 16" "true"
  ;;
12l-256-prenorm)
  "$0" attn-12l-256-prenorm
  "$0" grass-12l-256-prenorm
  ;;

all)
  "$0" 6l-128
  "$0" 12l-256
  ;;

all-prenorm)
  "$0" 6l-128-prenorm
  "$0" 12l-256-prenorm
  ;;

*)
  echo "Usage: $0 <experiment>"
  echo
  echo "Single runs (Post-Norm):"
  echo "  attn-6l-128      Attention, 6 layers, seq 128"
  echo "  grass-6l-128     Grassmann, 6 layers, seq 128"
  echo "  attn-12l-256     Attention, 12 layers, seq 256"
  echo "  grass-12l-256    Grassmann, 12 layers, seq 256"
  echo
  echo "Single runs (Pre-Norm):"
  echo "  attn-6l-128-prenorm      Attention (Pre-Norm), 6 layers, seq 128"
  echo "  grass-6l-128-prenorm     Grassmann (Pre-Norm), 6 layers, seq 128"
  echo "  attn-12l-256-prenorm     Attention (Pre-Norm), 12 layers, seq 256"
  echo "  grass-12l-256-prenorm    Grassmann (Pre-Norm), 12 layers, seq 256"
  echo
  echo "Grouped runs:"
  echo "  6l-128           Both models @ 6 layers, seq 128 (Post-Norm)"
  echo "  12l-256          Both models @ 12 layers, seq 256 (Post-Norm)"
  echo "  6l-128-prenorm   Both models @ 6 layers, seq 128 (Pre-Norm)"
  echo "  12l-256-prenorm  Both models @ 12 layers, seq 256 (Pre-Norm)"
  echo "  all              Run everything (Post-Norm)"
  echo "  all-prenorm      Run everything (Pre-Norm)"
  exit 1
  ;;
esac
