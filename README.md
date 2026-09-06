<h1 align="center">OpenAI Whisper GUI (Refactored Fork)</h1>

###

<p align="center">Modern GUI application that transcribes and translates audio/video files using OpenAI Whisper.</p>

###

![demo](https://github.com/rudymohammadbali/OpenAI-Whisper-GUI/assets/63475761/96b3e69e-067c-461e-b212-e1884f15a2a3)

###

> **Note on this Fork:** This version refactors the core program workflow and execution pipeline beyond basic bug fixes to optimize transcription processing and UI responsiveness.

###

<h2 align="left">Version: 1.1.0 (Custom Refactor)</h2>

- **Workflow & Core Refactoring**
1. Refactored application execution flow and pipeline for Whisper models.
2. Streamlined UI controls and interaction behavior.
3. Improved file handling and processing state feedback.

- **Base Features**
1. Config handler: Save, load, and reset config
2. Automatic GPU detection and dynamic model selection
3. Modern UI with Light/Dark mode themes
4. Export transcribed text & add subtitles to video

###

<h2 align="left">Installation & Environment Notes</h2>

###

<h4 align="left">Requirements</h4>

<p>
* <strong>Python version 3.10 or 3.11 recommended</strong> (See CUDA troubleshooting below)<br>
* Torch with CUDA support (GPU recommended; CPU supported)<br>
* ffmpeg
</p>

###

<h4 align="left">Setup</h4>

git clone https://github.com/ersu007/OpenAI-Whisper-GUI.git
cd OpenAI-Whisper-GUI
python setup.py
python main.py

###

<h2 align="left">Troubleshooting & CUDA Notes</h2>

###

> **Important:** If PyTorch defaults to CPU mode or fails to detect your GPU, verify your Python version and PyTorch CUDA installation.

* **Python Version:** Python 3.10 or 3.11 is strongly recommended for stable CUDA wheel support. Newer Python releases often lack pre-compiled PyTorch CUDA wheels, causing a silent fallback to CPU mode.
* **Fixing CPU-Only PyTorch:** If your GPU is not detected, reinstall PyTorch with explicit CUDA 12.1 support:

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

* **Verify GPU Acceleration:**

python -c "import torch; print(torch.cuda.is_available())"

*(Should print True if CUDA is configured correctly)*

###

<h2 align="left">Credits & Support</h2>

###

<p align="left">
Originally created by <a href="https://github.com/rudymohammadbali/OpenAI-Whisper-GUI" target="_blank">rudymohammadbali</a>.<br>
For more information visit <a href="https://github.com/openai/whisper" target="_blank">OpenAI Whisper GitHub</a>.
</p>

###

<p align="left">Support the original creator's ongoing efforts via PayPal:</p>

<div align="center">
  <a href="https://www.paypal.com/paypalme/iamironman0" target="_blank">
    <img src="https://img.shields.io/static/v1?message=PayPal&logo=paypal&label=&color=00457C&logoColor=white&labelColor=&style=flat" height="40" alt="paypal logo" />
  </a>
</div>