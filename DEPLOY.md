# Deploying the prototype

Goal: a URL you can send to a client, for free.

**Recommended: Streamlit Community Cloud.** It is free, purpose-built for this
kind of app, gives a permanent `https://...streamlit.app` link, and redeploys
itself every time you push. The whole thing takes about 10 minutes.

---

## Before you start: one decision

The repository contains the mill's 499-shade dictionary (as
`data/shade_master.csv`) and the generated job population. The raw client
workbooks, PDFs and photos are **excluded** by `.gitignore` and will not be
published — but the shade names themselves are in the committed data, because
the app needs them to work.

If that is commercially sensitive, **make the GitHub repository private.**
Streamlit Community Cloud deploys from private repositories on the free tier,
and the resulting app can still be shared by link. Repository private, app
link public is the combination you want.

---

## Route 1 — Streamlit Community Cloud (recommended)

### Step 1. Put the code on GitHub

The repository is already initialised and committed locally. Create an empty
repository on GitHub (no README, no .gitignore — this repo has both), then:

```bash
cd "C:/Users/Bhawanthi Pabasara/OneDrive/Desktop/pathum"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

If GitHub asks for a password, it wants a **Personal Access Token**, not your
account password: GitHub → Settings → Developer settings → Personal access
tokens → Tokens (classic) → Generate new token, tick `repo`, and paste the
token as the password.

### Step 2. Deploy

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**
   (if your repo is private, authorise the extra permission when prompted).
3. Fill in:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick something like `dye-scaleup-platform`
4. Click **Deploy**.

The first build installs the dependencies (2–4 minutes), then the app trains
its four models on first load (~30 seconds, once). After that it is instant.

### Step 3. Share it

Your link will be:

```
https://<whatever-you-chose>.streamlit.app
```

In the app's **⋮ → Settings → Sharing**, confirm it is set so anyone with the
link can view. Send that URL to the client. Nothing to install on their side.

### Keeping it updated

```bash
git add -A
git commit -m "describe the change"
git push
```

Streamlit Cloud redeploys automatically within a minute.

---

## Route 2 — Hugging Face Spaces (free alternative)

Useful if you would rather not use GitHub, since you can upload files through
the browser.

1. Create an account at **https://huggingface.co**.
2. **New** → **Space**. Name it, choose **Streamlit** as the SDK, hardware
   **CPU basic (free)**, visibility **Public**.
3. Upload the repository contents (**Files** → **Add file** → **Upload
   files**), or push with git as the Space page instructs.
4. Rename `app.py` to `streamlit_app.py`, **or** add a one-line `README.md`
   front-matter block with `app_file: app.py`. Spaces looks for
   `streamlit_app.py` by default.

Link: `https://huggingface.co/spaces/<user>/<space>`

---

## Route 3 — a temporary link, right now, with no accounts

If you only need the client to look at it today and the app is already running
locally, tunnel it. The link dies when you close the terminal.

**Cloudflare Tunnel** (no signup):

```bash
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8501
```

It prints a `https://something.trycloudflare.com` URL. That is your link.

**ngrok** (free account, more stable):

```bash
winget install ngrok.ngrok
ngrok config add-authtoken <token-from-ngrok.com>
ngrok http 8501
```

Keep the laptop awake and the terminal open — the link only works while both
the tunnel and the Streamlit server are running. This is fine for a live demo
call, and the wrong tool for "have a look whenever you get a chance".

---

## Why the trained models are not in the repository

`.gitignore` excludes `*.joblib`. That is deliberate, and it is the single
most important thing to leave alone:

1. **Size.** The stage models are 44 MB and the champion is another 11 MB.
2. **Pickle fragility — the real reason.** A joblib file is only guaranteed to
   load under the library versions that wrote it. This machine runs
   scikit-learn 1.9 / NumPy 2.5 / pandas 3.0; a host will install whatever pip
   resolves on the day. A mismatched unpickle fails at import, and the app dies
   on startup with an error the client sees.

Instead, `requirements.txt` gives version **ranges**, and the app trains its
own models from the committed 3.1 MB gzipped dataset the first time it starts.
Training is seeded, so the deployed app reproduces the exact figures quoted in
the paper — verified: 2.84 / 2.97 / 3.12 / 3.54 % MAPE across the four stages.

If a stored `stage_models.joblib` *is* present and loadable, the app uses it
and skips training. The fallback is a safety net, not a detour.

---

## Two dependency files

| File | Use |
|---|---|
| `requirements.txt` | What the **deployed app** needs. Lean on purpose — fewer packages, faster cold start, fewer ways for the build to fail. |
| `requirements-research.txt` | What **`run_all.py`** needs, adding LightGBM, CatBoost and SHAP for the full benchmark. |

Hosts read `requirements.txt`, so the benchmark-only libraries never slow down
a deploy. The app's champion is XGBoost, which is in both.

---

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| Build fails installing packages | A pinned version is unavailable for the host's Python. Loosen it in `requirements.txt` — ranges, not pins. |
| "Dataset is missing" error page | `data/textile_dyeing_scaleup_master.csv.gz` did not get committed. Check with `git ls-files data/`. |
| App stuck on "Please wait..." | First load is training the models (~30 s). Only the first visitor after a restart sees this. |
| Resource-limit error | Community Cloud allows ~1 GB RAM. Lower `N_RECORDS` in `src/config.py` and regenerate. |
| Client sees a login wall | The app is private. App menu → **Settings** → **Sharing** → allow anyone with the link. |
| App "has gone to sleep" | Free Community Cloud apps sleep after ~7 days idle. Anyone can wake it with the button; it takes ~30 s. Open the link yourself before a client demo. |
