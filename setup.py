import os
import subprocess
import sys
import time


def make_setup():
    if not sys.platform.startswith("win"):
        print("This app only supports Windows systems!")
        sys.exit()

    else:
        # Checking required folders
        folders = ["src"]
        missing_folder = [folder for folder in folders if not os.path.exists(folder)]
        if missing_folder:
            print(f"These folder(s) are missing: {missing_folder}")
            print("Please ensure your project structure includes all required directories.")
            sys.exit()
        else:
            print("All required folders detected!")

        # Upgraded to fetch the latest whisper package alongside modern dependencies
        required_modules = [
            "Pillow>=10.0.0",
            "customtkinter>=5.2.0",
            "packaging>=23.0",
            "openai-whisper",  # Pulls the latest release automatically
            "pynvml",
            "pydub"
        ]

        # Target PyTorch with CUDA 12.1 for current GPU drivers
        pytorch_win = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

        print("Installing/updating core dependencies to latest versions...")
        for module in required_modules:
            try:
                subprocess.call(f'"{sys.executable}" -m pip install -U {module}', shell=True)
            except Exception as e:
                print(f"Error installing {module}: {e}")

        print("Ensuring PyTorch with CUDA 12.1 is installed...")
        try:
            subprocess.call(f'"{sys.executable}" -m pip install {pytorch_win}', shell=True)
        except Exception as e:
            print(f"Error installing PyTorch: {e}")
            print("Please manually run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

        print("\nSetup Complete!")
        time.sleep(3)


if __name__ == "__main__":
    make_setup()