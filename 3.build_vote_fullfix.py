# -*- coding: utf-8 -*-
"""Read case_plan.csv and execute vote for each enabled case."""
from pathlib import Path
import sys
import multiprocessing
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) in sys.path:
    sys.path.remove(str(_PROJECT_DIR))
sys.path.insert(0, str(_PROJECT_DIR))
# Spyder can retain modules from another project directory.
for _name in list(sys.modules):
    if _name in ('case_manager', 'config_loader', 'hsr_pipeline', 'feasibility_core') or _name == 'hsrgen' or _name.startswith('hsrgen.') or _name == 'trainset_tools' or _name.startswith('trainset_tools.'):
        sys.modules.pop(_name, None)
from case_manager import run_entry

if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(run_entry('vote'))
