import os, io, json, tempfile, base64
import runpod
import yt_dlp
import torch
import whisperx

# ---- helpers ----
def _get_video_url(inp: dict) -> str:
    for k in ["videoUrl", "url", "audio_url", "youtube_url"]:
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _get_language(inp: dict, fallback=None):
    lang = inp.get("language") or inp.get("lang")
    if isinstance(lang, str) and lang.strip():
        return lang.strip()
    return fallback

def _dl_audio_to_wav(video_url: str, out_dir: str) -> str:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        # final file has .wav after postproc
        base = info.get("id")
        # Some templates download original ext then convert; find the .wav
        for fn in os.listdir(out_dir):
            if fn.startswith(base) and fn.lower().endswith(".wav"):
                return os.path.join(out_dir, fn)
    raise RuntimeError("Failed to download/convert audio")

def _simplify_segments(segments):
    out = []
    for s in segments:
        try:
            out.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": (s.get("text") or "").strip()
            })
        except Exception:
            pass
    return out

def _flatten_words(segments):
    words = []
    for s in segments:
        for w in s.get("words", []) or []:
            word_txt = w.get("word") or w.get("text") or ""
            words.append({
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "word": str(word_txt)
            })
    return words

# ---- handler ----
def handler(event):
    """
    Expects event['input'] like:
    {
      "videoUrl": "https://www.youtube.com/watch?v=XXXX",
      "language": "en"   // optional
    }
    Returns:
      { "segments": [{start,end,text}], "words": [{start,end,word}] }
    """
    inp = event.get("input") or {}
    url = _get_video_url(inp)
    if not url:
        return {"error": "videoUrl required"}

    lang = _get_language(inp, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    with tempfile.TemporaryDirectory() as td:
        # 1) Download to WAV
        wav_path = _dl_audio_to_wav(url, td)

        # 2) Transcribe
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, batch_size=16, language=lang)

        # 3) Align
        use_lang = lang or result.get("language") or "en"
        model_a, metadata = whisperx.load_align_model(language_code=use_lang, device=device)
        aligned = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                 return_char_alignments=False)

        segs_simple = _simplify_segments(aligned.get("segments") or result.get("segments") or [])
        words = _flatten_words(aligned.get("segments") or [])

        return {"segments": segs_simple, "words": words, "language": use_lang}

runpod.serverless.start({"handler": handler})
