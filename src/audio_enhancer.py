import os
import re
from pydub import AudioSegment
import pydub.effects as effects

def sanitize_filename(filename: str) -> str:
    """Removes invalid filename characters for filesystem safety."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

def process_audio_chunk(audio_file_path: str, start_sec: float, end_sec: float, level: int) -> str:
    """
    Extracts and enhances low-confidence audio chunks and saves them in a
    'low_confidence_clips' folder inside the SAME directory as the original audio.
    """
    try:
        audio = AudioSegment.from_file(audio_file_path)
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        
        # 200ms buffer to catch whole words
        chunk = audio[max(0, start_ms - 200): min(len(audio), end_ms + 200)]
        
        if level == 1:
            chunk = effects.normalize(chunk)
        elif level == 2:
            chunk = effects.normalize(chunk).high_pass_filter(150)
            
        # Target the SAME folder as the original audio file
        original_dir = os.path.dirname(os.path.abspath(audio_file_path))
        output_dir = os.path.join(original_dir, "low_confidence_clips")
        os.makedirs(output_dir, exist_ok=True)
        
        # Build file name: originalname_01m23s_to_01m28s_lvl1.mp3
        start_fmt = f"{int(start_sec // 60):02d}m{int(start_sec % 60):02d}s"
        end_fmt = f"{int(end_sec // 60):02d}m{int(end_sec % 60):02d}s"
        
        base_name = sanitize_filename(os.path.splitext(os.path.basename(audio_file_path))[0])
        out_path = os.path.join(output_dir, f"{base_name}_{start_fmt}_to_{end_fmt}_lvl{level}.mp3")
        
        chunk.export(out_path, format="mp3")
        return out_path

    except Exception as e:
        print(f"Audio processing error: {e}")
        return ""


def transcribe_with_adaptive_retry(transcriber, target_score: float = -0.8, max_retries: int = 2) -> dict:
    """
    Transcribes audio with faster-whisper and automatically executes a targeted
    enhancement loop for any segment falling below the target confidence score.
    """
    print("\n--- Starting Adaptive Multi-Pass Transcription ---")
    
    segments, info = transcriber.load_model.transcribe(
        transcriber.audio_file,
        language=transcriber.language,
        task=transcriber.task,
        beam_size=1
    )
    
    segments_list = []
    
    for segment in segments:
        start_time = f"{int(segment.start // 60):02d}:{int(segment.start % 60):02d}"
        end_time = f"{int(segment.end // 60):02d}:{int(segment.end % 60):02d}"
        time_str = f"[{start_time} -> {end_time}]"

        current_text = segment.text
        current_score = segment.avg_logprob
        attempt = 0
        
        while current_score < target_score and attempt < max_retries:
            attempt += 1
            print(f"{time_str} ⚠️ Low confidence ({round(current_score, 2)}). Retry attempt {attempt}/{max_retries}...")
            
            chunk_file = process_audio_chunk(
                transcriber.audio_file, 
                segment.start, 
                segment.end, 
                level=attempt
            )
            
            if chunk_file and os.path.exists(chunk_file):
                print(f"  ├─ Saved audio snippet: {chunk_file}")
                
                retry_segments, _ = transcriber.load_model.transcribe(
                    chunk_file,
                    language=transcriber.language,
                    task=transcriber.task,
                    beam_size=5
                )
                
                retry_list = list(retry_segments)
                if retry_list:
                    new_text = "".join([s.text for s in retry_list]).strip()
                    new_score = retry_list[0].avg_logprob
                    
                    if new_score > current_score and new_text:
                        print(f"  └─ Score improved from {round(current_score, 2)} to {round(new_score, 2)}")
                        current_text = new_text
                        current_score = new_score

        print(f"{time_str} {current_text} (Final Score: {round(current_score, 2)})")
        
        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": current_text,
            "score": current_score
        })

    full_text = "".join([s["text"] for s in segments_list])
    
    result = {
        "text": full_text,
        "segments": segments_list,
        "language": info.language
    }
    
    transcriber.result = result
    return result