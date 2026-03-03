# mikado-blender

Synthetic training data generator for [mikado-judge](https://github.com/spyszkowski/mikado-judge).

Uses Blender's rigid-body physics to simulate Mikado sticks falling into a pile,
renders from above, and auto-exports YOLO-OBB labels.

## How it works

1. Sticks are created as cylinders with coloured tip caps
2. Rigid-body simulation drops them onto a flat surface
3. Camera looks straight down, renders the settled pile
4. Label file is written with the projected OBB of each stick

## Usage

### Google Colab (recommended)

Open `notebooks/generate.ipynb` in Colab and run all cells.
Blender is installed automatically via `apt-get`.

### Local (requires Blender installed)

```bash
blender --background --python scripts/generate.py -- \
    --count 200 \
    --output output/ \
    --config configs/
```

## Configuration

| File | Purpose |
|------|---------|
| `configs/sticks.yaml` | Stick dimensions, colours, counts per class |
| `configs/render.yaml` | Image size, camera, lighting, physics settings |

## Output

```
output/
├── images/
│   ├── synthetic_00000.png
│   ├── synthetic_00001.png
│   └── ...
└── labels/
    ├── synthetic_00000.txt
    ├── synthetic_00001.txt
    └── ...
```

Label format is identical to mikado-judge YOLO-OBB format:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4   (normalised 0-1)
```

## Integration with mikado-judge

After generating, merge the synthetic images/labels with your real dataset
and re-run `prepare_dataset.py` to rebuild the training split.
