"""Build the COMPLETE final 217-image annotated dataset bundle (deliverable/archival).
Read-only w.r.t. production annotations. Does NOT touch training code or the 150-image
training bundle. Includes: images, merged binary masks, per-plane masks, full metadata,
split.json (holdout FIXED at 13), manifest.json (sha256 of every file)."""
import os, io, sys, json, hashlib
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("/app/backend/.env")
from pathlib import Path
from datetime import datetime
from PIL import Image
import numpy as np
from bson import ObjectId
import storage, config, pymongo

db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
FROZEN13 = ["0016","0023","0031","0032","0037","0044","0051","0054","0058","0064","0085","0120","0138"]
OUT = Path("/app/gpu_run/dataset_bundle_217")
for sub in ["images", "masks_merged", "masks_planes"]:
    (OUT/sub).mkdir(parents=True, exist_ok=True)

def jsonable(v):
    if isinstance(v, ObjectId): return str(v)
    if isinstance(v, datetime): return v.isoformat()
    if isinstance(v, dict): return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, list): return [jsonable(x) for x in v]
    return v

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    appr = sorted(db.images.find({"status": "approved"}), key=lambda d: d["dataset_id"])
    by_id = {d["dataset_id"]: d for d in appr}
    assert all(f in by_id for f in FROZEN13), "a frozen-13 holdout id is missing!"
    hold_groups = set(by_id[f].get("house_group_id") for f in FROZEN13)

    manifest = {"files": {}, "counts": {}}
    metadata = {}
    n_planes_total = 0; n_parapet_total = 0

    for d in appr:
        did = d["dataset_id"]
        # image
        raw,_ = storage.get_object(d["storage_path"])
        Image.open(io.BytesIO(raw)).convert("RGB").save(OUT/"images"/f"{did}.png")
        manifest["files"][f"images/{did}.png"] = sha(OUT/"images"/f"{did}.png")
        # merged binary mask
        md,_ = storage.get_object(config.merged_path(did))
        mm = (np.array(Image.open(io.BytesIO(md)).convert("L")) > 127).astype(np.uint8)*255
        Image.fromarray(mm, "L").save(OUT/"masks_merged"/f"{did}.png")
        manifest["files"][f"masks_merged/{did}.png"] = sha(OUT/"masks_merged"/f"{did}.png")
        # per-plane masks
        ann = db.annotations.find_one({"dataset_id": did})
        (OUT/"masks_planes"/did).mkdir(exist_ok=True)
        plane_meta = []
        for p in ann["generated"]["planes"]:
            name = p.split("/")[-1]
            pd,_ = storage.get_object(p)
            pm = (np.array(Image.open(io.BytesIO(pd)).convert("L")) > 127).astype(np.uint8)*255
            Image.fromarray(pm, "L").save(OUT/"masks_planes"/did/name)
            rel = f"masks_planes/{did}/{name}"
            manifest["files"][rel] = sha(OUT/rel)
            is_par = name.endswith("-parapet.png")
            plane_meta.append({"file": rel, "name": name, "is_parapet": is_par})
            n_planes_total += 1; n_parapet_total += int(is_par)
        # metadata (image doc + annotation doc, JSON-safe)
        metadata[did] = {
            "image": jsonable({k: d[k] for k in d if k != "_id"}),
            "annotation": jsonable({k: ann[k] for k in ann if k not in ("_id",)}),
            "planes": plane_meta,
        }

    # split (holdout FIXED at 13; group-aware pool)
    holdout = list(FROZEN13)
    pool, excluded = [], []
    for d in appr:
        did = d["dataset_id"]
        if did in FROZEN13: continue
        (excluded if d.get("house_group_id") in hold_groups else pool).append(did)
    split = {"holdout": holdout, "pool": sorted(pool), "excluded_group_overlap": sorted(excluded),
             "house_groups": {d["dataset_id"]: d.get("house_group_id") for d in appr},
             "note": "Final 217-image annotated dataset. Holdout FIXED at 13. Pool excludes any image "
                     "sharing a house_group with the holdout (group-aware, no leakage)."}
    (OUT/"split.json").write_text(json.dumps(split, indent=2))
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2))
    manifest["files"]["split.json"] = sha(OUT/"split.json")
    manifest["files"]["metadata.json"] = sha(OUT/"metadata.json")
    manifest["counts"] = {"approved_images": len(appr), "holdout": len(holdout),
                          "pool_train": len(pool), "excluded_group_overlap": len(excluded),
                          "per_plane_masks_total": n_planes_total, "parapet_planes_total": n_parapet_total,
                          "files_total": len(manifest["files"])}
    manifest["bundle_sha256"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2))

    print("counts:", json.dumps(manifest["counts"]))
    print("excluded_group_overlap:", excluded)
    print("bundle_sha256:", manifest["bundle_sha256"])
    print("bundle at:", OUT)

if __name__ == "__main__":
    main()
