import json
from pathlib import Path

root = Path(__file__).resolve().parent
for path in sorted(root.glob("runs/*/seed_*/fold_*/models.json")):
    print(f"== {path.relative_to(root)}")
    for config in json.loads(path.read_text(encoding="utf-8")):
        print(f"  {config['model']:<11} param={config['parameter']:<7} epochs={config['epochs']:<3} "
              f"fit={config['fit_seconds']:7.1f}s infer={config['inference_seconds']:6.2f}s "
              f"cells/s={config['cells_per_second']:9.0f} peak={config['peak_cuda_allocated_bytes']}")
    run = path.parents[2] / "run_config.json"
    if run.exists():
        args = json.loads(run.read_text(encoding="utf-8"))
        print(f"  n_genes={args['arguments']['n_genes']} batch={args['arguments']['batch_size']} "
              f"max_epochs={args['arguments']['max_epochs']} device={args['cuda_device']}")
