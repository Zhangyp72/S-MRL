

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import GKSD_CE_JE  


def parse_args() -> argparse.Namespace:  
  
    p = argparse.ArgumentParser(description="Run the multimodal spike-train pipeline")
    p.add_argument("--data", type=str, default="./data", help="Dataset root folder")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p.add_argument("--hidden", type=int, default=16, help="Hidden neuron count")
    p.add_argument("--synapses", type=int, default=4, help="Synapses per connection")
    p.add_argument("--skip-train", action="store_true", help="Skip network training phase")
    return p.parse_args()

def main() -> None:    
    args = parse_args()
    data_root = Path(args.data)
    if not data_root.exists():
        print(f"[ERROR] Data root {data_root} does not exist", file=sys.stderr)
        sys.exit(1)   
    dataset = GKSD_CE_JE.build_dataset(data_root)
    if not dataset:
        print("[ERROR] No data found – please populate the data/ folder.", file=sys.stderr)
        sys.exit(1)
    labels = {d["label"] for d in dataset}
    print(f"Loaded {len(dataset)} samples across {len(labels)} categories: {sorted(labels)}")  
    if not args.skip_train:
        GKSD_CE_JE.train_network(dataset, hidden=args.hidden, synapses=args.synapses, epochs=args.epochs)   
    acc =GKSD_CE_JE.nearest_neighbor_accuracy(dataset)
    print(f"\nNearest-neighbor leave-one-out accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
