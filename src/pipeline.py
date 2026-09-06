# src/pipeline.py
import os
import site

# Register NVIDIA CUDA & cuDNN DLL paths for Windows CTranslate2
try:
    for sp in site.getsitepackages():
        cudnn_path = os.path.join(sp, "nvidia", "cudnn", "bin")
        cublas_path = os.path.join(sp, "nvidia", "cublas", "bin")
        if os.path.exists(cudnn_path):
            os.add_dll_directory(cudnn_path)
        if os.path.exists(cublas_path):
            os.add_dll_directory(cublas_path)
except Exception as e:
    print(f"Warning loading CUDA DLLs: {e}", flush=True)

from faster_whisper import WhisperModel
from src.audio_enhancer import transcribe_with_adaptive_retry

def run_gpu_pipeline(options: dict, gui_queue=None):
    """
    Initializes faster-whisper on CUDA and executes 
    adaptive retries with live queue streaming.
    """
    model_size = options.get("model", "large-v3")
    device = options.get("device", "cuda")
    compute_type = options.get("compute_type", "float16")

    # Initialize model on GPU
    load_model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

    # Class wrapper matching transcriber interface
    class TranscriberWrapper:
        def __init__(self, audio_file, model, language, task):
            self.audio_file = audio_file
            self.load_model = model
            self.language = None if language == "Auto" else language.lower()
            self.task = task
            self.result = None

    transcriber = TranscriberWrapper(
        audio_file=options["audio"],
        model=load_model,
        language=options.get("language", "Auto"),
        task=options.get("task", "transcribe")
    )

    # Execute retry pipeline
    return transcribe_with_adaptive_retry(
        transcriber,
        enable_retry=options.get("enable_retry", True),
        target_score=options.get("target_score", -1.0),
        max_retries=options.get("max_retries", 2),
        secondary_language=options.get("secondary_language", "English"),
        gui_queue=gui_queue
    )