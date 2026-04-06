# Koenderink Scale-Space Explorer

Interactive Streamlit app for visualizing Gaussian scale space through a synthetic
landscape inspired by Koenderink's scale-space framework.

The app shows:

- a 3D smoothed surface
- a 2D structural view of peaks, saddles, and watershed boundaries
- interactive scale control with animation playback

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

This app is ready for Streamlit Community Cloud deployment.

1. Push the repository to GitHub.
2. Go to `https://share.streamlit.io/`.
3. Create a new app.
4. Select the repository, branch, and `app.py` as the entrypoint.
5. Choose the Python version in advanced settings.
6. Deploy.

If available in the deploy dialog, use Python `3.14` to match the local development
environment used most recently for this project.

## Repository Notes

- Main app: `app.py`
- Python dependencies: `requirements.txt`
- Coauthor/local shared-folder setup notes: `COAUTHOR_SETUP.md`

## Reference

This project is a visualization-oriented interpretation of ideas from Jan J.
Koenderink's work on scale space and image structure.
