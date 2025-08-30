from pathlib import Path
import runpy

pkg_script = Path(__file__).resolve().parent / "rag" / "chat.py"
if not pkg_script.exists():
	raise SystemExit("rag/chat.py not found")
runpy.run_path(str(pkg_script), run_name="__main__")
