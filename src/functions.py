import json
import os
import webbrowser
import sys
from pathlib import Path
from threading import Thread

import customtkinter as ctk
import pynvml
import whisper  # Retained for helper functions & subtitle writers

sys.path.append(str(Path(__file__).resolve().parent))

from faster_whisper import WhisperModel
from audio_enhancer import transcribe_with_adaptive_retry
from PIL import Image
from pydub import AudioSegment
from whisper.utils import get_writer

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DOWNLOAD_DIR = Path().home() / "Downloads"
APP_VERSION = "Version 1.0.0"
ICONS = {
    "settings": ctk.CTkImage(light_image=Image.open(f"{CURRENT_PATH}\\icons\\settings.png"),
                             dark_image=Image.open(f"{CURRENT_PATH}\\icons\\settings.png"), size=(25, 25)),
    "folder": ctk.CTkImage(light_image=Image.open(f"{CURRENT_PATH}\\icons\\folder.png"),
                           dark_image=Image.open(f"{CURRENT_PATH}\\icons\\folder.png"), size=(18, 18))
}

LANGUAGE_VALUES = [
    "Auto",
    "English",
    "Turkish",
    "Polish",
    "Catalan",
    "Dutch",
    "Arabic",
    "Swedish",
    "Italian",
    "Indonesian",
    "Hindi",
    "Finnish",
    "Vietnamese",
    "Hebrew",
    "Ukrainian",
    "Greek",
    "Malay",
    "Czech",
    "Romanian",
    "Danish",
    "Hungarian",
    "Tamil",
    "Norwegian",
    "Thai",
    "Urdu",
    "Croatian",
    "Bulgarian",
    "Lithuanian",
    "Latin",
    "Maori",
    "Malayalam",
    "Welsh",
    "Slovak",
    "Telugu",
    "Persian",
    "Latvian",
    "Bengali",
    "Serbian",
    "Azerbaijani",
    "Slovenian",
    "Kannada",
    "Estonian",
    "Macedonian",
    "Breton",
    "Basque",
    "Icelandic",
    "Armenian",
    "Nepali",
    "Mongolian",
    "Bosnian",
    "Kazakh",
    "Albanian",
    "Swahili",
    "Galician",
    "Marathi",
    "Punjabi",
    "Sinhala",
    "Khmer",
    "Shona",
    "Yoruba",
    "Somali",
    "Afrikaans",
    "Occitan",
    "Georgian",
    "Belarusian",
    "Tajik",
    "Sindhi",
    "Gujarati",
    "Amharic",
    "Yiddish",
    "Lao",
    "Uzbek",
    "Faroese",
    "Haitian creole",
    "Pashto",
    "Turkmen",
    "Nynorsk",
    "Maltese",
    "Sanskrit",
    "Luxembourgish",
    "Myanmar",
    "Tibetan",
    "Tagalog",
    "Malagasy",
    "Assamese",
    "Tatar",
    "Hawaiian",
    "Lingala",
    "Hausa",
    "Bashkir",
    "Javanese",
    "Sundanese"
]

FONTS = {
    "title": ("Inter", 22, "normal"),
    "subtitle": ("Inter", 18, "normal"),
    "btn": ("Inter", 14, "normal"),
    "normal": ("Inter", 14, "normal"),
    "small": ("Inter", 12, "normal"),
}

DROPDOWN = {
    "font": FONTS["small"],
    "corner_radius": 2,
    "alpha": 1.0,
    "frame_corner_radius": 5,
    "x": 0,
    "justify": "center"
}

OPTION = {
    "width": 160,
    "height": 28,
    "corner_radius": 3,
    "font": FONTS["small"]
}

BUTTONS = {
    "height": 30,
    "corner_radius": 3,
    "font": FONTS["btn"]
}


def change_theme(new_theme):
    ctk.set_appearance_mode(new_theme)


def help_page() -> None:
    webbrowser.open("https://github.com/rudymohammadbali")


def check_gpu() -> dict:
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        cuda = device_count > 0

        if cuda:
            device = pynvml.nvmlDeviceGetHandleByIndex(0)
            total_mem = pynvml.nvmlDeviceGetMemoryInfo(device).total
            total_mem_gb = round(total_mem / (1024 ** 3))
        else:
            total_mem_gb = 0

        pynvml.nvmlShutdown()
    except Exception:
        total_mem_gb = 0
        cuda = False

    model_req = {
        10: ["large", "large-v1", "large-v2", "large-v3"],
        5: ["medium", "medium.en"],
        2: ["small", "small.en"],
        1: ["tiny", "base", "tiny.en", "base.en"]
    }

    models_list = [model for req, models in model_req.items() if total_mem_gb >= req for model in models]
    return {"models": models_list, "cuda": cuda}


def merge_dicts(d1, d2) -> dict:
    for k, v in d1.items():
        if k in d2 and isinstance(v, dict) and isinstance(d2[k], dict):
            d2[k] = merge_dicts(v, d2[k])
    return {**d1, **d2}


def save_config(settings: dict, filename: str) -> bool:
    try:
        existing_settings = load_config(filename)
        merged_settings = merge_dicts(existing_settings, settings)

        with open(filename, 'w') as f:
            json.dump(merged_settings, f, indent=4)

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def load_config(filename: str) -> dict:
    try:
        if not os.path.isfile(filename):
            return {}

        with open(filename, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Error: {e}")
        return {}


def reset_config(filename: str):
    if os.path.exists(filename):
        os.remove(filename)

    gpu_info = check_gpu()
    default_settings = {
        "download_path": str(DOWNLOAD_DIR),
        "models": gpu_info["models"],
        "cuda": gpu_info["cuda"],
        "theme": "system",
        "model": "base",
        "language": "auto",
        "task": "transcribe",
        "device": "cpu"
    }

    save_config(default_settings, filename)


def transcriber_task(options: dict = None, callback: any = None) -> None:
    transcriber = WhisperTranscriber(audio_file=options["audio"], model_size=options["model"],
                                     device=options["device"], language=options["language"], task=options["task"])
    get_result = transcriber.transcribe()
    callback(get_result)


def subtitles_writer(options: dict = None, callback: any = None) -> None:
    file_path = options["output_dir"]
    result = options["result"]
    audio_file = options["audio_file"]

    selected_extension = os.path.splitext(file_path)
    file_extension = selected_extension[1]
    dir_name, get_file_name = os.path.split(file_path)

    default_options = {
        'max_line_width': None,
        'max_line_count': None,
        'highlight_words': False
    }

    if file_extension == ".srt":
        writer = get_writer("srt", dir_name)
        writer(result, audio_file, default_options)
    elif file_extension == ".txt":
        txt_writer = get_writer("txt", dir_name)
        txt_writer(result, audio_file, default_options)
    elif file_extension == ".vtt":
        vtt_writer = get_writer("vtt", dir_name)
        vtt_writer(result, audio_file, default_options)
    elif file_extension == ".tsv":
        tsv_writer = get_writer("tsv", dir_name)
        tsv_writer(result, audio_file, default_options)
    elif file_extension == ".json":
        json_writer = get_writer("json", dir_name)
        json_writer(result, audio_file, default_options)
    elif file_extension == ".all":
        all_writer = get_writer("all", dir_name)
        all_writer(result, audio_file, default_options)

    callback(f"File exported as: {file_path}")


def subtitle_to_video(options: dict, callback: any) -> None:
    result = options["result"]
    audio = options["audio"]
    output = options["output"]
    lang = options["lang"]
    device = options["device"]
    if result and audio:
        file_name = os.path.basename(output)
        dir_name = os.path.dirname(output)
        file_extension = os.path.splitext(output)

        temp_file_name = "temp_srt.srt"
        writer = get_writer("srt", ".")
        writer(result, temp_file_name, {"highlight_words": True, "max_line_count": 50, "max_line_width": 3})

        if file_extension[1] == ".mkv":
            os.system(
                "ffmpeg -i {} -i {} -map 0 -map 1 -c copy -disposition:s:0 default -metadata:s:s:0 language={} {} -y".format(
                    '"' + audio + '"',
                    temp_file_name,
                    lang,
                    os.path.join('"' + dir_name, file_name + '"'),
                )
            )
        else:
            if device == "cuda":
                os.system(
                    "ffmpeg -i {} -c:v h264_nvenc -vf subtitles={} {} -y".format(
                        '"' + audio + '"',
                        temp_file_name,
                        os.path.join('"' + dir_name, file_name + '"'),
                    )
                )
            else:
                os.system(
                    "ffmpeg -i {} -vf subtitles={} {} -y".format(
                        '"' + audio + '"',
                        temp_file_name,
                        os.path.join('"' + dir_name, file_name + '"'),
                    )
                )

        os.remove(os.path.join(".", temp_file_name))

        callback(output)


def start_transcriber(options: dict = None, callback: any = None) -> None:
    Thread(target=transcriber_task, args=(options, callback), daemon=True).start()


def start_writer(options: dict = None, callback: any = None) -> None:
    Thread(target=subtitles_writer, args=(options, callback), daemon=True).start()


def start_subtitle(options: dict = None, callback: any = None) -> None:
    Thread(target=subtitle_to_video, args=(options, callback), daemon=True).start()


class WhisperTranscriber:
    def __init__(self, audio_file, model_size, device, language, task):
        if not audio_file:
            raise ValueError("[!] Audio file not provided!")
        if not self.validate_file(audio_file):
            raise ValueError("Error, file is not valid")

        self.audio_file = audio_file
        self.available_models = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large"]
        
        # Clean model size input without appending .en automatically (ensures Turkish support works)
        self.model_size = model_size if model_size in self.available_models else "base"
        self.device = "cpu"
        
        # Format language mapping
        lang_lower = str(language).lower().strip()
        if lang_lower in ["auto", "default", "", "none"]:
            self.language = None
        elif lang_lower in ["turkish", "tr"]:
            self.language = "tr"
        elif lang_lower in ["english", "en"]:
            self.language = "en"
        else:
            self.language = lang_lower

        self.task = task if task in ["transcribe", "translate"] else "transcribe"
        self.prompt = {}
        self.result = None

        # Load optimized INT8 CTranslate2 model for high-speed CPU execution
        self.load_model = WhisperModel(
            self.model_size, 
            device="cpu", 
            compute_type="int8"
        )

    def transcribe(self) -> dict:
        # Hand off execution to the standalone audio enhancer module
        return transcribe_with_adaptive_retry(self, target_score=-0.8, max_retries=2)

    @staticmethod
    def get_valid_prompts(prompts) -> dict:
        if prompts is None:
            return {}
        transcribe_params = {
            "verbose": bool,
            "temperature": float,
            "compression_ratio_threshold": float,
            "logprob_threshold": float,
            "no_speech_threshold": float,
            "condition_on_previous_text": bool,
            "initial_prompt": str,
            "word_timestamps": bool,
            "prepend_punctuations": str,
            "append_punctuations": str
        }

        valid_prompts = {}

        for prompt_name, value in prompts.items():
            if prompt_name in transcribe_params and isinstance(value, transcribe_params[prompt_name]):
                valid_prompts[prompt_name] = value

        return valid_prompts

    @staticmethod
    def validate_file(file_path: str) -> bool:
        if not os.path.isfile(file_path):
            return False

        try:
            AudioSegment.from_file(file_path)
            return True
        except Exception as e:
            print(e)
            return False