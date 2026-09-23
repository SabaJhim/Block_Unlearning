"""Download / clean datasets into data/<name>.csv.

    python scripts/prepare_data.py adult
    python scripts/prepare_data.py credit_default --src path/to/default_of_credit_card_clients.xls
    python scripts/prepare_data.py bank_marketing --src path/to/bank-additional-full.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bvu.data import prepare_adult, prepare_bank_marketing, prepare_credit_default  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["adult", "credit_default", "bank_marketing"])
    ap.add_argument("--src", help="path to the manually downloaded raw file (credit_default, bank_marketing)")
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    if args.dataset == "adult":
        prepare_adult()
    else:
        if not args.src:
            sys.exit(f"--src is required for {args.dataset}; see README 'Datasets' for where to get it")
        {"credit_default": prepare_credit_default, "bank_marketing": prepare_bank_marketing}[args.dataset](args.src)


if __name__ == "__main__":
    main()
