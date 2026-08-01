#!/usr/bin/env bash
set -euo pipefail

project_root=/home/tlclab/Structure_building
cd "$project_root"

bash training/run_catalyst_supplement_r4.sh
bash training/run_catalyst_supplement_evaluation.sh r4 r3
