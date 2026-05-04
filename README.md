# 🎙️ Audio Deepfake Detection with LCNN

A deep learning system that detects AI-generated audio and attributes it to specific generative models, using a Light CNN (LCNN) with LFCC features.

---

## How It Works

- **Binary Detection** — Classifies audio as *Real* or *Fake*
- **Source Attribution** — Identifies *which AI model* generated the fake audio
- **Grad-CAM** — Highlights suspicious time-frequency regions for explainability

The backbone uses **Max-Feature-Map (MFM)** activations, which are effective at catching vocoder artifacts introduced by TTS systems.

---

## Dataset

| Source     | Label  |
|------------|--------|
| LJSpeech   | Real   |
| Fake Audio (multi-model) | Per-model |

Split: 80% train / 10% val / 10% test, stratified and balanced.

---

## Training

| Parameter   | Value |
|-------------|-------|
| Epochs      | 30    |
| Batch Size  | 64    |
| Optimizer   | Adam + Cosine Annealing |
| Loss        | Weighted BCE + Cross Entropy |

---

## Quick Start

```bash
pip install librosa soundfile torch pytorch-grad-cam scikit-learn seaborn grad-cam
```

```python
model = LCNN(n_classes=12).to(DEVICE)
model.load_state_dict(torch.load('best_lcnn.pth', map_location=DEVICE))
model.eval()

lfcc = extract_lfcc('audio.wav')
feat = torch.tensor(lfcc).unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    bin_logit, src_logit = model(feat)

print(f"Fake probability: {torch.sigmoid(bin_logit).item():.2%}")
print(f"Attributed to: {idx2model[src_logit.argmax(1).item()]}")
```

---

## Tech Stack

PyTorch · Librosa · SciPy · Grad-CAM · scikit-learn · Matplotlib
