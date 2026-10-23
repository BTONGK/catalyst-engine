"""Pull adsorbate-on-oxide pairs out of OC22's IS2RE-Total LMDBs: each adsorbate+slab system in the validation
splits with its bare slab, DFT-relaxed positions and DFT total energies (PBE+U, spin-polarized).

Download into cache/oc22/ (git-ignored; CC-BY 4.0, Tran et al. 2023), then run in a throwaway environment:

    cd experiments/oxide_check/cache/oc22
    curl -O https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz
    curl -O https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl
    tar -xzf is2res_total_train_val_test_lmdbs.tar.gz
    cd ../..
    uv run --no-project --python 3.12 --with lmdb --with torch --with numpy python -I extract_oc22.py

Each LMDB record is a pickled torch_geometric Data object. They load through a restricted unpickler that resolves
only the four names those records use (an OrderedDict, torch's tensor rebuild, torch's storage loader through
torch.load(weights_only=True), and Data as a plain stand-in), so torch_geometric stays uninstalled and nothing else
in the files can run. Writes cache/oc22_pairs.json.
"""

from __future__ import annotations

import collections
import glob
import io
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OC22 = HERE / "cache" / "oc22"
LMDBS = OC22 / "is2res_total_train_val_test_lmdbs" / "data" / "oc22" / "is2re-total"


class Data:
    """Stand-in for torch_geometric's Data: keeps the attribute dict it was pickled with."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


def _storage(b: bytes):
    return torch.load(io.BytesIO(b), weights_only=True)


ALLOWED = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch.storage", "_load_from_bytes"): _storage,
    ("torch_geometric.data.data", "Data"): Data,
}


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def load(b: bytes):
    return Restricted(io.BytesIO(b)).load()


def records(split: str):
    for path in sorted(glob.glob(str(LMDBS / split / "*.lmdb"))):
        env = lmdb.open(path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                if key == b"length":
                    continue
                yield load(value)
        env.close()


def as_row(d: Data, split: str) -> dict | None:
    if getattr(d, "y_relaxed", None) is None or getattr(d, "pos_relaxed", None) is None:
        return None
    arr = lambda t: np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)
    return {"sid": int(arr(d.sid).ravel()[0]) if hasattr(d, "sid") else None, "split": split,
            "numbers": arr(d.atomic_numbers).astype(int).ravel().tolist(),
            "pos": arr(d.pos_relaxed).reshape(-1, 3).round(5).tolist(),
            "cell": arr(d.cell).reshape(3, 3).round(5).tolist(),
            "fixed": arr(d.fixed).astype(int).ravel().tolist() if hasattr(d, "fixed") else None,
            "tags": arr(d.tags).astype(int).ravel().tolist() if hasattr(d, "tags") else None,
            "energy": float(arr(d.y_relaxed).ravel()[0])}


def main() -> None:
    meta = Restricted(open(OC22 / "oc22_metadata.pkl", "rb")).load()
    by_sid: dict[int, dict] = {}
    for split in ("val_id", "val_ood"):
        for d in records(split):
            row = as_row(d, split)
            if row and row["sid"] is not None:
                by_sid[row["sid"]] = row
    ads = {sid: r for sid, r in by_sid.items() if meta.get(sid, {}).get("nads", 0) >= 1}
    wanted = {meta[sid]["slab_sid"] for sid in ads if meta[sid].get("slab_sid") is not None} - set(by_sid)
    if wanted:  # bare slabs that sit in another split
        for split in ("train", "test_id", "test_ood"):
            for d in records(split):
                row = as_row(d, split)
                if row and row["sid"] in wanted:
                    by_sid[row["sid"]] = row
    pairs = []
    for sid, r in ads.items():
        m = meta[sid]
        slab = by_sid.get(m.get("slab_sid"))
        if slab is None:
            continue
        pairs.append({"sid": sid, "split": r["split"], "bulk_id": m["bulk_id"], "bulk": m["bulk_symbols"],
                      "miller": list(m["miller_index"]), "adsorbate": m["ads_symbols"], "nads": m["nads"],
                      "adslab": r, "slab": slab})
    out = HERE / "cache" / "oc22_pairs.json"
    out.write_text(json.dumps(pairs))
    counts = collections.Counter(p["adsorbate"] for p in pairs)
    print(f"{len(by_sid)} systems with relaxed energies; {len(pairs)} adsorbate+slab pairs with their slab")
    print("adsorbates:", dict(counts.most_common()))
    print("splits:", dict(collections.Counter(p["split"] for p in pairs)))


if __name__ == "__main__":
    sys.exit(main())
