
> **Note on this Fork:** This version refactors the core program workflow and execution pipeline beyond basic bug fixes to optimize transcription processing and UI responsiveness.

* **Workflow & Core Refactoring**

1. Refactored application execution flow and background GPU pipeline for Whisper models.


2. Streamlined UI controls and interaction behavior.


3. Removed blocking GUI loader overlays (`CTkLoader`) during live text streaming for full visual visibility.
4. Thread-safe message processing via queue polling for smooth real-time transcription updates.
5. Improved file handling and processing state feedback.



* **Base Features**

1. Config handler: Save, load, and reset config


2. Automatic GPU detection and dynamic model selection


3. Modern UI with Light/Dark mode themes


4. Export transcribed text & add subtitles to video



git clone [https://github.com/ersu007/OpenAI-Whisper-GUI.git](https://github.com/ersu007/OpenAI-Whisper-GUI.git?utm_source=gemini)


cd OpenAI-Whisper-GUI
python setup.py
python main.py

*To launch the app without a background console window:*

pythonw.exe main.py

> **Important:** If PyTorch defaults to CPU mode or fails to detect your GPU, verify your Python version and PyTorch CUDA installation.
> 
> 

* **Python Version:** Python 3.10, 3.11, or 3.12 is strongly recommended for stable CUDA wheel support. **Python 3.14 is currently incompatible with CUDA acceleration** for `faster-whisper` and `torch` because pre-built GPU binaries (`wheels`) do not exist yet, causing a fallback to CPU mode or missing module errors.
* **Fixing CPU-Only PyTorch:** If your GPU is not detected, reinstall PyTorch with explicit CUDA 12.1 support:



pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121?utm_source=gemini)

* **Verify GPU Acceleration:**


python -c "import torch; print(torch.cuda.is_available())"

*(Should print True if CUDA is configured correctly)*