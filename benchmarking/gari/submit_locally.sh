#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status (-e),
# treat unset variables as an error (-u), and catch errors in pipes (-o pipefail).
set -euo pipefail

mkdir -p out

TESSERACT_BIN=./bazel-bin/src/tesseract
# Create a timestamp in nanoseconds for when the script starts
STARTTIME=$(($(date +%s%N)))

COUNTER=0
SHOTS=50000

for num in 0; do
  for p_err in 0.001; do
    for circuit in testdata/bivariatebicyclecodes/r=6,*p=$p_err,noise=si1000,c=bivariate_bicycle_Z,*.stim; do
      echo "$circuit"
    done
    # for circuit in testdata/colorcodes/r=7,*p=$p_err,noise=si1000,c=superdense_color_code_Z,*.stim; do
    #   echo "$circuit"
    # done
    # for circuit in testdata/surfacecodes/r=7,*p=$p_err,noise=si1000,c=surface_code_Z,*.stim; do
    #   echo "$circuit"
    # done
  done
done | shuf | while read circuit; do
  circuit_dir=$(dirname "$circuit")
  circuit_name=$(basename "$circuit" .stim)
  mapping_file="$circuit_dir/gari/${circuit_name}_mapping.json"

  echo "========================================="
  echo "Running benchmark for circuit: $circuit_name"
  # Submit also one baseline job
   $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-1det.json
   $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-21det.json
   $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --no-revisit-dets --beam 5 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-5beam.json
   $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --no-revisit-dets --beam 5 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --stats-out out/${STARTTIME}-${COUNTER}-baseline-5beam.json

   #run with gari
  #  for mode in modeA modeB modeC modeF modeG modeH modeI modeJ modeK modeL modeM modeN; do
  #  for mode in modeF modeG modeH modeI modeJ modeK modeA modeN; do
   for mode in modeA modeI modeN; do


     dem_file="$circuit_dir/gari/${circuit_name}_${mode}.dem"
     echo "Running GARI mode: $dem_file"
     for order in order2 order4 order7 order7a order7b order7c order9 order10 all; do
       echo "  Running order: $order"
       $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --beam 5 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${mode}-${order}-revisit_beam5.json
       $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --beam 10 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${mode}-${order}-revisit_beam10.json

      #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --beam 20 --beam-climbing --num-det-orders 1 --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --custom-order "$order" --stats-out out/${STARTTIME}-${COUNTER}-gari-${mode}-${order}-revisit.json

     done
   done
  #  $TESSERACT_BIN --circuit "$circuit" --sample-num-shots $SHOTS --sample-seed 0 --max-errors 100 --threads 32 --beam 20 --beam-climbing --num-det-orders 21 --det-order-index --pqlimit 1000000 --dem "$dem_file" --det-mapping-file "$mapping_file" --stats-out out/${STARTTIME}-${COUNTER}-gari-revisit-21det.json

#    $TESSERACT_BIN --circuit "$circuit" --sample-num-shots 1000 --sample_seed 0 --max-errors 10 --threads 32 --no-revisit-dets --beam 20 --beam-climbing --num-det-orders 1 --det-order-index --pqlimit 1000000 --dem "$dem_file" —det-mapping-file "$mapping_file" —stats-out out/${STARTTIME}-${COUNTER}.json

  # Increment counter for every single job so JSON files don't get overwritten
  COUNTER=$((COUNTER + 1))
done
