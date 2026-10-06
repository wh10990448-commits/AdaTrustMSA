# Publishing AdaTrustMSA on GitHub

## Recommended Repository Settings

**Repository name**

```text
AdaTrustMSA
```

**Description**

```text
A PyTorch framework for multimodal sentiment analysis on CMU-MOSEI, featuring reliability-aware text-audio-visual fusion, uncertainty estimation, robustness to noisy or missing modalities, and adaptive temporal early exit.
```

**Suggested topics**

```text
multimodal-learning
multimodal-sentiment-analysis
sentiment-analysis
cmu-mosei
cmu-mosi
pytorch
uncertainty-estimation
early-exit
robustness
deep-learning
```

## GitHub Creation Form

The provided package already contains a detailed `README.md`, `.gitignore`, and
MIT `LICENSE`. When creating the empty repository on GitHub:

- choose **Public** or **Private** according to your needs;
- leave **Add a README file** unchecked;
- choose **None** for `.gitignore`;
- choose **None** for license.

Initializing these files again on GitHub creates an unrelated initial commit and
may cause an avoidable merge conflict on the first push.

## Files That Must Not Be Uploaded

The included `.gitignore` excludes datasets, BERT weights, checkpoints, run
directories, generated results, logs, virtual environments, and Python caches.
Before the first commit, still verify that none of the following are staged:

```text
*.pkl
*.npz
*.csd
*.pt
*.pth
*.ckpt
*.safetensors
.venv/
models/
data/
runs/
results/
```

Never commit server passwords, API tokens, private SSH keys, or personal access
tokens.

## Initialize and Push

Create an empty repository on GitHub, extract the package, and run:

```bash
cd AdaTrustMSA
git init
git add .
git status
git commit -m "Initial AdaTrustMSA release"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/AdaTrustMSA.git
git push -u origin main
```

Replace `YOUR_NAME` with your GitHub username or organization. If the remote is
not empty, inspect its existing history before pulling or merging. Do not force
push unless you intentionally want to replace the remote history.

## Optional Release Checklist

- Confirm that the installation commands work in a clean environment.
- Confirm that `python -m pytest -q` passes.
- Check every result table against the JSON files under `docs/`.
- State that datasets and checkpoints are not included.
- Distinguish the standard MMSA protocol from the full CSD project protocol.
- State that the final Pareto result is a two-checkpoint ensemble.
- Avoid claiming a complete E-MoE or QA-MoE reproduction.
- Avoid claiming that all E-MoE metrics are exceeded.

A suitable first release tag is:

```text
v1.0.0
```
