import os
from typing import Tuple
from datetime import datetime
from config import MEETINGS_DIR, WHISPER_MODEL

def transcribe_audio(audio_path: str) -> Tuple[str, str, str]:
    """
    Returns: (meeting_id, transcript_text, transcript_path)
    """
    import whisper
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path)
    text = (result.get("text") or "").strip()

    meeting_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(MEETINGS_DIR, meeting_id)
    os.makedirs(out_dir, exist_ok=True)

    tpath = os.path.join(out_dir, "transcript.txt")
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(text)

    return meeting_id, text, tpath
