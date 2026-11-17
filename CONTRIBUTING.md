# Contributing to VidIMG

Contributions are welcome: bug reports, new samples that break the guard, fixes, translations and docs.
By submitting a pull request you agree that your contribution is licensed under the Apache License 2.0
(see LICENSE), like the rest of the project.

## Ground rules (they are the point of the tool)

1. **Faithful, not generative.** No face restorers, diffusion or other generative models in the default
   path. A change must never alter who people are, or the colours and lighting, unless the person asks.
2. **Free tools only.** Every dependency must be free and downloadable from an official release; pin new
   downloads with a SHA-256 in the setup code or a manifest/lock file under `tools/`.
3. **The guard decides.** Never deliver a result that fails the fidelity guard; new processing steps need
   a guard check (or an existing gate) that would catch them going wrong.
4. **Originals are untouched.** Never write next to or over an input file.
5. **Privacy.** Never send people's photos or videos to an external service, and never commit personal
   media: tests use synthetic images or skip when a private sample is missing.

## Development

```bash
git clone https://github.com/Q8Xbox/VidIMG-skill.git
cd VidIMG-skill
python vidimg_run.py setup --output D:/VidIMG
python -m pytest -q
```

Open the folder in Claude Code to use the skill and commands without the `vidimg:` prefix.
Keep pull requests focused, add a unit test for each fix, and describe what you measured (which file,
which gate values) when a change affects quality.
