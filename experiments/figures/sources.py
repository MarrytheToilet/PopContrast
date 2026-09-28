"""Read frozen inputs from a checkout or a standalone manuscript bundle."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def source_path(path):
    path = Path(path)
    bundled = ROOT / 'plot_data' / path.relative_to(ROOT)
    return bundled if bundled.is_file() else path
