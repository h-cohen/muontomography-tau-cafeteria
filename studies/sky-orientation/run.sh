# Sky swap: ingest-raw and opacity of pos0 and pos1 against each roof orientation
# (SKY_A, SKY_B) and the whole roof run. runs/skyswap/config.yaml is
# configs/cafeteria.yaml with raw.exposures and data_dir pointing at this machine's data.
# Then: compare.py runs/skyswap, and sectors.py.
set -e
for s in SKY_A SKY_B SKY; do
  arg=""
  if [ "$s" != SKY ]; then arg="--sky $s"; fi
  uv run cafetomo ingest-raw --config runs/skyswap/config.yaml --out runs/skyswap/ingest_$s --cache runs/skyswap/cache --jobs 12 $arg
  uv run cafetomo opacity --config runs/skyswap/config.yaml --pose results/pose.json --ingest runs/skyswap/ingest_$s --out runs/skyswap/opacity_$s
  echo "done $s"
done
