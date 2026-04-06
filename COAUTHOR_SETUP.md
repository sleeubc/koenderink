# Coauthor Setup Notes

Do not share a single `venv/` across machines. Python virtual environments contain
machine-specific absolute paths, so Dropbox syncing can break the interpreter and
entry points for a coauthor.

This project uses a per-user virtual environment inside the shared folder:

- your env: `.venv-$USER`
- coauthor env: `.venv-$USER`

## Setup

```bash
cd /Users/sanghoonlee/Dropbox/_proj/watersheds/tasks/koenderink_app
chmod +x setup_env.sh run_app.sh
./setup_env.sh
```

If you need a specific Python executable:

```bash
PYTHON_BIN=/opt/homebrew/bin/python3 ./setup_env.sh
```

## Run

```bash
./run_app.sh
```

Or activate manually:

```bash
source .venv-$USER/bin/activate
streamlit run app.py
```

## Existing broken `venv/`

The old shared `venv/` is not reliable and should not be used. Its scripts may point
to another machine's absolute path.

Each collaborator should ignore `venv/` completely and create their own local env with:

```bash
./setup_env.sh
```

Then always start the app with:

```bash
./run_app.sh
```

Do not run `source venv/bin/activate` or `streamlit run app.py` from the shared
`venv/` directory.
