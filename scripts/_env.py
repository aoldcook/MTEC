"""Load API credentials from the repo-root .env file (untracked).

Keys must never be hard-coded in tracked source. Copy .env.example to .env and
fill it in; both this loader and the shell scripts read from there.
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=None):
    path = path or os.path.join(_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def require(name):
    """Return env var `name`, or raise with a pointer to .env.example."""
    load_env()
    val = os.environ.get(name, "")
    if not val:
        raise SystemExit(
            "%s is not set. Copy .env.example to .env and fill in your key." % name
        )
    return val


load_env()
