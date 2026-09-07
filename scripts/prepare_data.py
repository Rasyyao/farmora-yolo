from roboflow import Roboflow
import shutil, os, yaml
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = SCRIPT_DIR.parent                        
DATASETS_DIR = ROOT_DIR / "datasets" / "merged"

load_dotenv()  
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

WORKSPACE = "money-detection-2wuos"

os.chdir(SCRIPT_DIR)

proj1 = rf.workspace(WORKSPACE).project("cyperus-rotundus-k8rjw-zlvru")
ds1 = proj1.version(1).download("yolov8", location="raw_rumput_teki")

proj2 = rf.workspace(WORKSPACE).project("amaranthus-spinosus-ovnza")
ds2 = proj2.version(1).download("yolov8", location="raw_bayam_duri")


def get_class_names(dataset_dir):
    with open(Path(dataset_dir) / "data.yaml") as f:
        data = yaml.safe_load(f)
    return data["names"]


def remap_and_copy(src_dir, dst_root, target_class_names, new_class_id, split):
    src_dir = Path(src_dir)
    dst_root = Path(dst_root)
    src_names = get_class_names(src_dir)

    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    labels_dir = src_dir / split / "labels"
    if not labels_dir.exists():
        print(f"  [{src_dir.name}/{split}] folder labels tidak ada, skip")
        return 0

    copied = 0
    for lbl_file in labels_dir.iterdir():
        with open(lbl_file) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            orig_class_id = int(parts[0])
            orig_class_name = src_names[orig_class_id]
            if orig_class_name in target_class_names:
                parts[0] = str(new_class_id)
                new_lines.append(" ".join(parts))

        if new_lines:
            img_base = lbl_file.stem
            src_images_dir = src_dir / split / "images"
            found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = src_images_dir / f"{img_base}{ext}"
                if img_path.exists():
                    shutil.copy(img_path, dst_images / f"{img_base}{ext}")
                    found = True
                    break
            if found:
                with open(dst_labels / lbl_file.name, "w") as f:
                    f.write("\n".join(new_lines))
                copied += 1

    return copied


print("Nama class Cyperus:", get_class_names(SCRIPT_DIR / "raw_rumput_teki"))
print("Nama class Amaranthus:", get_class_names(SCRIPT_DIR / "raw_bayam_duri"))
print()

total = 0
for split in ["train", "valid", "test"]:
    n1 = remap_and_copy(SCRIPT_DIR / "raw_rumput_teki", DATASETS_DIR,
                          target_class_names=["cyperus"], new_class_id=0, split=split)
    n2 = remap_and_copy(SCRIPT_DIR / "raw_bayam_duri", DATASETS_DIR,
                          target_class_names=["Amaranthus-spinosus"], new_class_id=1, split=split)
    print(f"[{split}] rumput_teki: {n1} images, bayam_duri: {n2} images")
    total += n1 + n2

print(f"\nTotal images merged: {total}")
print(f"Output location: {DATASETS_DIR}")