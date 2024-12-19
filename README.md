# representations

## usage

- Prereq: `uv`

- create and activate venv

```bash
uv venv
source .venv/bin/activate
uv sync
```

- run training:

```bash
python scripts/run.py --data_path /home/howard/representations/data/imagenette2-320/ --labeled_ratio 0.1,0.25,0.5 --batch-size 128
```
