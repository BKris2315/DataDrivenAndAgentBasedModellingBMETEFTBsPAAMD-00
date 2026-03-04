#!/bin/bash

# ----- parameters -----
ISING_EXEC=./ising
INPUT=network/lattice32.dat
OUTPUT_DIR=./field_sweep

T0=6.0
Tf=0.1
TSTEP=0.1

THRM=2000
MEAS=50

H_MIN=-1.0
H_MAX=1.0
H_STEP=0.01

mkdir -p $OUTPUT_DIR

echo "Starting increasing field sweep..."

# ---- increasing field ----
h=$H_MIN
while (( $(echo "$h <= $H_MAX" | bc -l) ))
do
    out=${OUTPUT_DIR}/h_up_${h}

    echo "Running h = $h"

    $ISING_EXEC \
        -input_file=$INPUT \
        -T0=$T0 \
        -Tf=$Tf \
        -T_step=$TSTEP \
        -thrm_steps=$THRM \
        -meas_steps=$MEAS \
        -ext_field=$h \
        -output=$out

    h=$(echo "$h + $H_STEP" | bc)
done

echo "Starting decreasing field sweep..."

# ---- decreasing field ----
h=$H_MAX
while (( $(echo "$h >= $H_MIN" | bc -l) ))
do
    out=${OUTPUT_DIR}/h_down_${h}

    echo "Running h = $h"

    $ISING_EXEC \
        -input_file=$INPUT \
        -T0=$T0 \
        -Tf=$Tf \
        -T_step=$TSTEP \
        -thrm_steps=$THRM \
        -meas_steps=$MEAS \
        -ext_field=$h \
        -output=$out

    h=$(echo "$h - $H_STEP" | bc)
done

echo "Field sweep completed."