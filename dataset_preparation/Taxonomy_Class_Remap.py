# Remaping based on the Excel File.
import os
import re
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd
import yaml
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_YAML = Path(r"D:/detection/dataset/data.yaml")
EXCEL_FILE = Path(r"D:/detection/dataset/taxonomy_classification_CVAT.xlsx")
DATASET_ROOT = Path(r"D:/detection/dataset")
OUTPUT_ROOT = Path(r"D:/detection/dataset/dataset_phylum")

SPLITS = ["train", "val", "test"]

# Non-biological object class: Plastic
NON_BIOLOGICAL_LABEL = "Non-biological"

# ============================================================
# 2. NAME ALIASES
# YAML NAME -> NAME USED IN EXCEL
# ============================================================

ALIASES = {
    "Nattastoma melanura": "Nettastoma melanurum",
    "Muriceides/Villogorgia complex":
        "Muriceides/Villogorgiacomplex",
    "gorgonia pic bianca": "White gorgonian",

    # Genus-level names, where applicable
    "Leptometra sp.": "Leptometra",
    "Munida sp.": "Munida",
    "Nezumia spp.": "Nezumia",
    "Plesionika sp.": "Plesionika",
    "Scorpaena sp.": "Scorpaena",
    "Trachurus sp.": "Trachurus",
}

# Plastic has no biological phylum.
# This is a dataset category, not a biological taxon.
MANUAL_PHYLUM_OVERRIDES = {
    "Plastic": "plastic",
    "Cidaridae": "Echinodermata",
    "Filograna- salmacina complex": "Annelida",
}

# ============================================================
# 3. HELPERS
# ============================================================

def normalize_name(value):
    """Normalize names to allow punctuation/spacing differences."""
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]", "", value)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def link_or_copy(source, destination):
    """Create a hard link when possible; otherwise copy the image."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


# ============================================================
# 4. LOAD DATA.YAML
# ============================================================

if not DATA_YAML.exists():
    raise FileNotFoundError(f"YAML not found: {DATA_YAML}")

data = load_yaml(DATA_YAML)

if "names" not in data or "nc" not in data:
    raise ValueError("data.yaml must contain 'names' and 'nc'.")

original_names = {
    int(class_id): str(name).strip()
    for class_id, name in data["names"].items()
}

if len(original_names) != int(data["nc"]):
    raise ValueError(
        f"YAML declares {data['nc']} classes, but "
        f"{len(original_names)} names were found."
    )

expected_ids = set(range(int(data["nc"])))

if set(original_names) != expected_ids:
    raise ValueError(
        "Original YAML class IDs must be consecutive, starting at 0."
    )

print(f"Loaded {len(original_names)} original YOLO classes.")


# ============================================================
# 5. LOAD EXCEL TAXONOMY
# ============================================================

if not EXCEL_FILE.exists():
    raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

taxonomy = pd.read_excel(EXCEL_FILE)

required_columns = {"query_name", "valid_name", "phylum"}

missing = required_columns - set(taxonomy.columns)

if missing:
    raise ValueError(
        f"Missing required Excel columns: {sorted(missing)}"
    )

# Build a name -> phylum dictionary.
# Use query_name and valid_name; genus is used as a fallback
# for labels representing an entire genus.

name_to_phylum = {}

for _, row in taxonomy.iterrows():
    phylum_value = row["phylum"]

    if pd.isna(phylum_value) or not str(phylum_value).strip():
        continue

    phylum = str(phylum_value).strip()

    candidate_names = [
        row["query_name"],
        row["valid_name"],
    ]

    if "genus" in taxonomy.columns:
        candidate_names.append(row["genus"])

    for candidate in candidate_names:
        if pd.isna(candidate) or not str(candidate).strip():
            continue

        key = normalize_name(candidate)

        if key in name_to_phylum:
            existing = name_to_phylum[key]

            if existing != phylum:
                raise ValueError(
                    f"Conflicting phylum assignments for "
                    f"'{candidate}': '{existing}' vs '{phylum}'. "
                    "Resolve this conflict in the taxonomy first."
                )

        name_to_phylum[key] = phylum


# ============================================================
# 6. MAP ORIGINAL CLASSES TO PHYLUM
# ============================================================

id_to_phylum = {}
unresolved = []

for old_id, original_name in original_names.items():

    # Special dataset category: Plastic
    if original_name in MANUAL_PHYLUM_OVERRIDES:
        phylum = MANUAL_PHYLUM_OVERRIDES[original_name]

    else:
        # First try the original YAML name, then its Excel alias.
        lookup_names = [
            original_name,
            ALIASES.get(original_name, original_name),
        ]

        phylum = None

        for lookup_name in lookup_names:
            key = normalize_name(lookup_name)

            if key in name_to_phylum:
                phylum = name_to_phylum[key]
                break

        if phylum is None:
            unresolved.append((old_id, original_name))
            continue

    id_to_phylum[old_id] = phylum


# Stop before writing labels if any class is unresolved.
if unresolved:
    print("\nERROR: These classes could not be mapped:")

    for class_id, class_name in unresolved:
        print(f"  ID {class_id}: {class_name}")

    raise ValueError(
        "Remapping stopped. Add the correct Excel name to ALIASES "
        "or investigate the missing taxonomy before rerunning."
    )


# ============================================================
# 7. CREATE NEW CLASS IDS
# ============================================================

# Deterministic IDs: alphabetical order of the phylum names.
phylum_names = sorted(set(id_to_phylum.values()))

phylum_to_id = {
    phylum: new_id
    for new_id, phylum in enumerate(phylum_names)
}

old_id_to_new_id = {
    old_id: phylum_to_id[phylum]
    for old_id, phylum in id_to_phylum.items()
}

print("\n" + "=" * 65)
print("ORIGINAL CLASS -> PHYLUM MAPPING")
print("=" * 65)

for old_id in sorted(original_names):
    print(
        f"{old_id:2d}: {original_names[old_id]}"
        f" -> {id_to_phylum[old_id]}"
        f" (new ID: {old_id_to_new_id[old_id]})"
    )

print(f"\nOriginal classes: {len(original_names)}")
print(f"Phylum-level classes: {len(phylum_names)}")

print("\nNEW CLASS NAMES")

for new_id, phylum in enumerate(phylum_names):
    print(f"{new_id}: {phylum}")


# ============================================================
# 8. REMAP YOLO ANNOTATIONS
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

bad_lines = []
empty_files = []
processed_files = 0
processed_boxes = 0
class_box_counts = Counter()

for split in SPLITS:

    source_labels = DATASET_ROOT / "labels" / split
    output_labels = OUTPUT_ROOT / "labels" / split

    if not source_labels.exists():
        raise FileNotFoundError(
            f"Label directory not found for split '{split}': "
            f"{source_labels}"
        )

    output_labels.mkdir(parents=True, exist_ok=True)

    label_files = sorted(source_labels.glob("*.txt"))

    print(
        f"\nProcessing {split}: "
        f"{len(label_files)} label files"
    )

    for label_file in tqdm(label_files, desc=f"Labels {split}"):

        output_file = output_labels / label_file.name
        new_lines = []

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_number, line in enumerate(lines, start=1):

            parts = line.strip().split()

            # Empty lines are ignored.
            if not parts:
                continue

            # Standard YOLO detection format:
            # class_id x_center y_center width height
            if len(parts) != 5:
                bad_lines.append(
                    (
                        split,
                        label_file.name,
                        line_number,
                        f"Expected 5 fields, found {len(parts)}",
                    )
                )
                continue

            try:
                class_float = float(parts[0])
                old_id = int(class_float)

                if class_float != old_id:
                    raise ValueError("Class ID is not an integer")

                coordinates = [float(x) for x in parts[1:]]

                if not all(
                    0.0 <= value <= 1.0
                    for value in coordinates
                ):
                    raise ValueError(
                        "Bounding-box coordinates must be between 0 and 1"
                    )

                if coordinates[2] <= 0 or coordinates[3] <= 0:
                    raise ValueError(
                        "Bounding-box width and height must be positive"
                    )

            except ValueError as exc:
                bad_lines.append(
                    (
                        split,
                        label_file.name,
                        line_number,
                        str(exc),
                    )
                )
                continue

            if old_id not in old_id_to_new_id:
                bad_lines.append(
                    (
                        split,
                        label_file.name,
                        line_number,
                        f"Unknown original class ID: {old_id}",
                    )
                )
                continue

            new_id = old_id_to_new_id[old_id]

            # Only replace the class ID.
            # Bounding-box coordinates remain unchanged.
            parts[0] = str(new_id)

            new_lines.append(" ".join(parts))

            processed_boxes += 1
            class_box_counts[new_id] += 1

        with open(output_file, "w", encoding="utf-8") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")

        processed_files += 1

        # Empty output files are valid when an image has no objects.
        if not new_lines:
            empty_files.append(f"{split}/{label_file.name}")


# ============================================================
# 9. COPY/LINK IMAGES
# ============================================================

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

for split in SPLITS:

    source_images = DATASET_ROOT / "images" / split
    output_images = OUTPUT_ROOT / "images" / split

    if not source_images.exists():
        raise FileNotFoundError(
            f"Image directory not found: {source_images}"
        )

    output_images.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in source_images.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    print(f"\nPreparing {split} images: {len(image_files)}")

    for image_path in tqdm(image_files, desc=f"Images {split}"):

        # Preserve relative subdirectory structure.
        relative_path = image_path.relative_to(source_images)
        destination = output_images / relative_path

        link_or_copy(image_path, destination)


# ============================================================
# 10. WRITE NEW DATA YAML
# ============================================================

new_data = {
    "path": str(OUTPUT_ROOT).replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(phylum_names),
    "names": {
        new_id: phylum
        for new_id, phylum in enumerate(phylum_names)
    },
}

output_yaml = OUTPUT_ROOT / "data_phylum.yaml"

with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        new_data,
        f,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# 11. SAVE MAPPING TABLE
# ============================================================

mapping_rows = []

for old_id in sorted(original_names):
    mapping_rows.append({
        "original_id": old_id,
        "original_name": original_names[old_id],
        "phylum": id_to_phylum[old_id],
        "new_id": old_id_to_new_id[old_id],
    })

mapping_csv = OUTPUT_ROOT / "class_mapping.csv"

pd.DataFrame(mapping_rows).to_csv(
    mapping_csv,
    index=False,
    encoding="utf-8",
)


# ============================================================
# 12. SAVE INVALID-LABEL REPORT
# ============================================================

if bad_lines:
    report_file = OUTPUT_ROOT / "invalid_labels_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        for item in bad_lines:
            f.write(f"{item}\n")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("REMAPPING COMPLETED")
print("=" * 65)

print(f"Original classes: {len(original_names)}")
print(f"New phylum-level classes: {len(phylum_names)}")
print(f"Label files processed: {processed_files}")
print(f"Bounding boxes remapped: {processed_boxes}")
print(f"Empty annotation files: {len(empty_files)}")
print(f"Invalid annotation lines: {len(bad_lines)}")

print("\nBounding boxes per new class:")

for new_id, phylum in enumerate(phylum_names):
    print(
        f"  {new_id}: {phylum} "
        f"-> {class_box_counts[new_id]} boxes"
    )

print(f"\nOutput dataset: {OUTPUT_ROOT}")
print(f"New YAML: {output_yaml}")
print(f"Mapping CSV: {mapping_csv}")

if bad_lines:
    print(
        "\nWARNING: Invalid annotation lines were skipped. "
        "Review invalid_labels_report.txt before training."
    )
else:
    print("\nNo invalid annotation lines were detected.")

print("\nUse data_phylum.yaml for phylum-level YOLO training.")
