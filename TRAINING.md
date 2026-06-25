# Model Training Guide

This guide covers the complete process of training a custom RVC voice model for use with Gandhigiri AI — from raw audio collection through to the final trained artifacts.

---

## Prerequisites

Before beginning, ensure you have the following:

- A Google account with access to [Google Colab](https://colab.research.google.com) and [Google Drive](https://drive.google.com)
- At least **1 GB of free Google Drive storage**
- A collection of raw audio recordings of your target voice (see Section 1)
- [Audacity](https://www.audacityteam.org/) or any audio editing software capable of exporting `.wav` files

---

## Section 1: Audio Collection

The quality and quantity of your training data is the single most important factor in determining the accuracy of the trained voice model.

### Minimum Requirements

| Property | Requirement |
|---|---|
| Total duration | 5–15 minutes of clean speech |
| Format | `.wav` or `.flac` (lossless) |
| Speaker | Single speaker only — no overlapping voices |
| Background noise | None, or minimal |
| Music/score | None |

### Guidelines

- **More data is better**, up to approximately 15–20 minutes. Beyond this, quality improvements are marginal.
- Fragmented clips (individual words or short phrases) are acceptable and can be combined with longer continuous recordings. RVC processes audio in small chunks regardless of source continuity.
- Prioritize **phonetic variety** — recordings that cover a wide range of sounds, emotions, and cadences will produce a more generalizable model than a single long monotone recording.
- Avoid recordings where the target voice overlaps with other speakers, even briefly. Any contamination from a second speaker will degrade the model.

---

## Section 2: Audio Cleaning

Raw recordings — particularly those extracted from film, television, or other produced media — almost always contain background music, ambient sound, or reverb that must be removed before training. Training on unclean audio is the most common cause of poor voice model quality.

### Step 1: Separate Voice from Background Music

Use **Ultimate Vocal Remover (UVR5)** to isolate the vocal track from any background music or score.

1. Download UVR5 from: [https://github.com/Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
2. Load your audio files into UVR5
3. Select the **MDX-Net** or **Demucs** model for best results
4. Run separation and export the isolated vocal track as `.wav`

### Step 2: Remove Ambient Noise

If the isolated vocal still contains background hiss, room noise, or light ambient sound:

1. Open the vocal track in **Audacity**
2. Select a short section of pure background noise (no speech)
3. Go to **Effect → Noise Reduction**
4. Click **Get Noise Profile**, then select the entire track and apply **Noise Reduction**
5. Use conservative settings to avoid introducing artifacts into the speech

### Step 3: Manual Review and Trimming

This step is critical and cannot be automated reliably.

1. Listen through the entire cleaned audio
2. Cut out any segments that contain:
   - A second speaker's voice, even for a single word
   - Audible music or score bleed-through that UVR5 did not fully remove
   - Coughing, laughter, or non-speech sounds
   - Sections with heavy reverb or echo
3. Keep only clean, clear speech segments from the target speaker
4. Export all kept segments as individual `.wav` files, or concatenate them into a single file

### Step 4: Verify Final Audio

Before proceeding to training, confirm the following:

- All files are in `.wav` or `.flac` format
- Total combined duration is at least 5 minutes
- Listening through the audio, you can hear only the target speaker with no background noise or music
- The audio is not clipped or distorted

---

## Section 3: Training with Applio on Google Colab

Applio is the recommended training platform for this project. It is an actively maintained fork of RVC with a cleaner dependency stack and a well-supported Colab notebook.

### Step 1: Open the Applio Colab Notebook

Navigate to the official Applio Colab notebook:

```
https://colab.research.google.com/github/IAHispano/Applio/blob/main/assets/Applio.ipynb
```

### Step 2: Set the Runtime to GPU

1. In Colab, go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**

### Step 3: Run the Installation Cells

Run the following cells in order:

1. **Install Applio** — installs all required dependencies
2. **Sync with Google Drive** — connects your Drive so that trained models are saved automatically. This creates a folder called `ApplioBackup` in your Drive root.

> This Drive sync is important. Colab sessions are temporary and will disconnect after a period of inactivity. Saving to Drive ensures your training progress is not lost.

### Step 4: Launch the Interface

Run the launch cell. Colab will output a URL — open it in your browser. When prompted for a password, enter the IP address displayed in the Colab cell output.

### Step 5: Upload Your Dataset

1. Upload your cleaned `.wav` files to a folder in your Google Drive (e.g., `MyDrive/gandhi_dataset/`)
2. In the Applio interface, navigate to the **Train** tab
3. Set the **Dataset Path** to the Google Drive path of your audio folder

### Step 6: Preprocess the Dataset

In the **Train** tab, fill in the following fields and click **Preprocess Dataset**:

| Field | Recommended Value |
|---|---|
| Model Name | A short identifier, e.g. `my-project` |
| Sample Rate | `40000` (40k) |
| Dataset Path | Path to your cleaned audio folder on Drive |

Wait for preprocessing to complete. This step slices your audio into training segments automatically.

### Step 7: Extract Features

Still in the **Train** tab, configure feature extraction:

| Field | Recommended Value |
|---|---|
| Pitch Extraction Method | `rmvpe` |

Click **Extract Features** and wait for completion.

### Step 8: Train the Model

Configure training parameters:

| Field | Recommended Value | Notes |
|---|---|---|
| Total Epochs | `200` | Increase to 300 if dataset is under 5 minutes |
| Batch Size | `8` | Reduce to 4 if you encounter out-of-memory errors |
| Save Frequency | Every `25` epochs | Allows you to compare intermediate checkpoints |
| Save Only Latest | `True` | Saves Drive storage |
| Pretrained Model | Default (enabled) | Significantly speeds up convergence — do not disable |

Click **Start Training**. For 5–10 minutes of training data at 200 epochs, expect approximately 1–3 hours on a free T4 GPU.

> **Note on overfitting:** More epochs is not always better, especially with smaller datasets. A model trained for too many epochs may memorize the training clips rather than generalizing the voice characteristics. Test intermediate checkpoints (e.g., at epoch 100, 150, and 200) and use the one that sounds most natural.

### Step 9: Train the Index File

After training completes, click **Train Feature Index** in the Train tab. This generates the `.index` file, which improves voice timbre accuracy during inference.

### Step 10: Export Your Model

1. Go to the **Train** tab → **Export Model** sub-tab
2. Click **Refresh** to list your trained checkpoints
3. Select the `.pth` file corresponding to your best epoch (e.g., `my-project_200e_6000s.pth`)
4. Select the `.index` file (named `added_*.index`)
5. Click **Upload** — both files will be saved to `ApplioExported/` in your Google Drive

---

## Section 4: Identifying the Correct Model Files

After training, your Google Drive `ApplioExported` folder will contain several files. Here is how to identify which ones to use:

| File | Description | Use? |
|---|---|---|
| `my-project_200e_6000s.pth` | Generator checkpoint at 200 epochs — **this is your voice model** | ✅ Yes |
| `my-project_100e_3000s.pth` | Intermediate checkpoint at 100 epochs | ⚠️ Only if 200e sounds worse |
| `D_<number>.pth` | Discriminator checkpoint — used only during training | ❌ No |
| `G_<number>.pth` | Raw generator checkpoint — superseded by the named export | ❌ No |
| `added_*.index` | Feature index file — required for inference | ✅ Yes |

Download the two files marked **Yes** and place them in the `models/` directory of this project.

---

## Section 5: Placing Model Files in the Project

Once downloaded from Google Drive, place your model files as follows:

```
gandhigiri-ai/
└── models/
    ├── my-project_200e_6000s.pth    ← your trained generator
    └── added_my-project.index       ← your trained index
```

Then update the model paths in `scripts/voice_changer.py`:

```python
MODEL_PATH = BASE_DIR / "models" / "my-project_200e_6000s.pth"
INDEX_PATH = BASE_DIR / "models" / "added_my-project.index"
```

---

## Section 6: Testing and Pitch Tuning

Before integrating the model into the full pipeline, verify it works correctly and find the optimal pitch setting.

### Test Basic Inference

```bash
python scripts/voice_changer.py audio_output/any_tts_output.wav
```

Listen to the output in `audio_output/gandhi_output.wav`.

### Tune the Pitch

The base TTS voice and the trained voice model may sit at different natural pitches. Run the pitch tuner to find the best offset:

```bash
python scripts/voice_changer.py audio_output/any_tts_output.wav --tune-pitch
```

This generates five files in `audio_output/` at pitch offsets of -4, -2, 0, +2, and +4 semitones. Listen to each and identify which sounds most accurate. Then update `PITCH_SHIFT` in `scripts/voice_changer.py`:

```python
PITCH_SHIFT = -2   # or whichever value sounded best
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Voice sounds robotic or metallic | Background noise in training data | Re-clean audio with UVR5 and retrain |
| Wrong voice timbre leaking through | Second speaker in training data | Manually review and trim audio, retrain |
| Output pitch sounds unnatural | Pitch mismatch between TTS and model | Run `--tune-pitch` and adjust `PITCH_SHIFT` |
| `KeyError: 'config'` during inference | Model trained with incompatible RVC version | Retrain using the Applio Colab notebook |
| Poor quality with correct voice | Too few training epochs or too little data | Increase epochs or collect more audio |
| Overfitted — sounds exactly like training clips | Too many epochs for dataset size | Use an earlier checkpoint (e.g., 100 or 150 epochs) |
