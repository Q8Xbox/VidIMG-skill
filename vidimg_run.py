"""Run VidIMG from any folder (the Claude Code plugin calls it by its full path):

    python vidimg_run.py setup [--output DIR] [--yes] [--check]   # install / repair / verify
    python vidimg_run.py doctor | analyze ... | restore ... | restore-video ...   # the vidimg CLI
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    if sys.argv[1:2] == ["setup"]:
        from vidimg.setup import main as setup_main
        sys.exit(setup_main(sys.argv[2:]))
    from vidimg.cli import main as cli_main
    sys.exit(cli_main(sys.argv[1:]))
