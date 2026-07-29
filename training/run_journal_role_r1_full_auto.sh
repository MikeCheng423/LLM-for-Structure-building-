#!/usr/bin/env bash
set -euo pipefail

cd /home/tlclab/Structure_building
bash training/run_journal_role_r1.sh
bash training/run_journal_role_evaluation.sh
