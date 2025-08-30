from pathlib import Path
from dotenv import load_dotenv


_loaded = False


def ensure_env_loaded() -> None:
    global _loaded
    if _loaded:
        return
    
    cwd = Path(__file__).resolve().parents[1]
    candidates = [cwd / ".env", cwd / ".env.local"]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p)
    _loaded = True
