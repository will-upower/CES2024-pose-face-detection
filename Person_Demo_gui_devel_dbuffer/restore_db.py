#! /usr/bin/env python3

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--db_dir', default=Path('./db'), type=Path,
                    help='directory to store database files')
args = parser.parse_args()

if __name__ == "__main__":
    for f in args.db_dir.glob('*.bak'):
        f = str(f)
        shutil.copy(f, f.replace('.bak', '', 1))
