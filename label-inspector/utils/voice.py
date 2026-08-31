"""Spoken prompts and a fault tone, for the operator standing at the machine.

The operator's hands are on the coil, not on the keyboard, and the screen is
across the machine from them. When the line stops they need to be told what
happened and what to do about it without walking over to read it -- which is
the whole point of the rewind: they are already at the coil, winding it back,
and the console is telling them whether it worked.

Two things make that work in a factory rather than on a desk.

Voices. espeak is a formant synthesiser from the 1990s and sounds like one;
nobody wants to be shouted at by it all shift. The default here is Microsoft's
en-IN neural voice through edge-tts -- a real Indian-English speaker, female
(Neerja) or male (Prabhat) -- with espeak kept only as the fallback for a
machine with no network and nothing cached yet.

Cache. Those voices are synthesised over the network, which is both too slow
to do while a fault is being raised and no use at all on a line with no
internet. But the vocabulary here is tiny and almost entirely fixed, so every
phrase is rendered to a .wav on disk the first time it is needed and simply
replayed forever after. The fixed phrases are pre-rendered in the background
at start-up, so the first fault of the day speaks instantly, and a machine
that has run once keeps its voice with the network unplugged.

Everything goes out on a worker thread behind a queue, so neither synthesis
nor playback can stall the capture loop. An urgent line clears whatever is
queued and jumps the front: when the machine has just stopped, "rotate the
coil back" matters and the running commentary behind it does not.
"""

import hashlib
import math
import os
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import wave

# The Indian-English neural voices edge-tts can reach. Neerja is the default
# because a female voice carries better over machine noise.
EDGE_VOICES = {
    "female": "en-IN-NeerjaNeural",
    "male": "en-IN-PrabhatNeural",
    "expressive": "en-IN-NeerjaExpressiveNeural",
}
DEFAULT_EDGE_VOICE = EDGE_VOICES["female"]

# espeak has no Indian English, so the fallback asks for the nearest thing it
# does have and is only ever reached when the good voices cannot be.
ESPEAK_FALLBACK_VOICE = "en-gb"

# edge-tts pads every clip with about a second of silence, which is a second
# between the alert tone and hearing what is wrong. Trim it from both ends:
# reverse, strip the lead, reverse back, strip the other lead.
_TRIM = ("silenceremove=start_periods=1:start_threshold=-45dB:"
         "start_silence=0.03:detection=peak,areverse,"
         "silenceremove=start_periods=1:start_threshold=-45dB:"
         "start_silence=0.05:detection=peak,areverse")

CACHE_ROOT = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "label-inspector", "voice")


def _which(*names):
    for name in names:
        if shutil.which(name):
            return name
    return None


def _have(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class Voice:
    """Say things out loud, without ever blocking the caller.

    `say(text, key=...)` drops a line whose key matches the last one spoken,
    which is what stops a message raised inside the capture loop from being
    repeated sixty times a second. Pass no key for a line that should always
    be said.

    `alert(text, lead=...)` is the fault form: the tone, then a short fixed
    phrase that is always pre-rendered and so always instant, then the detail
    -- which carries a row number, cannot be pre-rendered for every row, and
    is synthesised while the lead is still playing.
    """

    def __init__(self, enabled=True, engine="auto", name=None, rate=0,
                 tone=True, warm=()):
        self.enabled = bool(enabled)
        self.engine = None
        self.voice = None
        self.rate = int(rate)
        self.player = None
        self.tone_path = None
        self._last_key = None
        self._queue = queue.Queue(maxsize=24)
        self._thread = None
        self._cache = None
        # One lock per phrase, not one for the whole cache: the warm-up thread
        # is synthesising the vocabulary while the line runs, and a fault must
        # not have to wait behind whatever phrase it happens to be on.
        self._locks = {}
        self._locks_lock = threading.Lock()

        if not self.enabled:
            return

        self.player = _which("aplay", "paplay", "afplay")
        self.ffmpeg = _which("ffmpeg")
        self._pick_engine(engine, name)
        if self.engine is None or self.player is None:
            self.enabled = False
            print("[voice] no way to speak found - pip install edge-tts, or "
                  "apt install espeak-ng alsa-utils (running silent)")
            return
        if tone:
            self.tone_path = self._make_tone()

        print(f"[voice] {self.voice} via {self.engine}, played with "
              f"{self.player}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if warm:
            # Off the main thread: rendering the whole vocabulary the first
            # time takes a few seconds, and the camera should not wait for it.
            threading.Thread(target=self._warm, args=(list(warm),),
                             daemon=True).start()

    # -- which engine ------------------------------------------------------
    def _open_cache(self):
        """One folder per voice, so changing voice does not replay the old
        one out of cache."""
        self._cache = os.path.join(
            CACHE_ROOT, f"{self.engine}-{self.voice}-{self.rate:+d}")
        os.makedirs(self._cache, exist_ok=True)

    def _pick_engine(self, engine, name):
        want = (engine or "auto").lower()
        if want in ("auto", "edge") and _have("edge_tts") and self.ffmpeg:
            self.engine = "edge"
            self.voice = EDGE_VOICES.get((name or "").lower(),
                                         name or DEFAULT_EDGE_VOICE)
            self._open_cache()
            # One probe phrase decides it: if the voice can be reached, or was
            # reached on an earlier run and is still cached, use it.
            if want == "edge" or self._render("Ready.") is not None:
                return
            # asked for auto, but edge cannot reach the network and has
            # nothing cached, so fall through to something that always works
            print("[voice] edge-tts could not be reached - falling back to "
                  "espeak until it can")
        if want in ("auto", "edge", "espeak") and _which("espeak-ng", "espeak"):
            self.engine = "espeak"
            self.voice = name if (name and want == "espeak") \
                else ESPEAK_FALLBACK_VOICE
            self._open_cache()
            return
        self.engine = None
        self._cache = None

    # -- rendering ---------------------------------------------------------
    def _lock_for(self, path):
        with self._locks_lock:
            lock = self._locks.get(path)
            if lock is None:
                lock = self._locks[path] = threading.Lock()
            return lock

    def _path_for(self, text):
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]
        return os.path.join(self._cache, digest + ".wav")

    def _render(self, text):
        """The .wav for `text`, synthesising and caching it if need be.

        Returns None when it cannot be made -- no network for a voice that
        needs one, and nothing cached -- which the caller treats as "say it
        some other way" rather than as an error.
        """
        if not text:
            return None
        path = self._path_for(text)
        if os.path.exists(path):
            return path
        with self._lock_for(path):        # one synthesis of a phrase, not two
            if os.path.exists(path):
                return path
            tmp = path + f".{os.getpid()}.part"
            try:
                ok = (self._render_edge(text, tmp) if self.engine == "edge"
                      else self._render_espeak(text, tmp))
                if not ok or not os.path.exists(tmp) \
                        or os.path.getsize(tmp) < 128:
                    return None
                os.replace(tmp, path)     # never leave a half-written cache
                return path
            except Exception:
                return None
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def _render_edge(self, text, out):
        import asyncio
        import edge_tts

        mp3 = out + ".mp3"
        rate = f"{self.rate:+d}%" if self.rate else "+0%"

        async def go():
            await edge_tts.Communicate(text, self.voice, rate=rate).save(mp3)

        def convert(filters):
            return subprocess.run(
                [self.ffmpeg, "-v", "error", "-y", "-i", mp3, *filters,
                 "-ar", "22050", "-ac", "1", "-f", "wav", out],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30).returncode == 0

        try:
            asyncio.run(go())
            # Plain conversion if this ffmpeg has no silenceremove: a clip
            # with a second of dead air still beats no clip at all.
            return convert(["-af", _TRIM]) or convert([])
        finally:
            if os.path.exists(mp3):
                try:
                    os.remove(mp3)
                except OSError:
                    pass

    def _render_espeak(self, text, out):
        return self._render_espeak_to(text, out, self.voice)

    def _render_espeak_to(self, text, out, voice):
        engine = _which("espeak-ng", "espeak")
        # --voice-rate is a percentage either way; espeak wants words a minute.
        wpm = int(165 * (1.0 + self.rate / 100.0))
        return subprocess.run(
            [engine, "-v", voice, "-s", str(max(80, wpm)), "-w", out, text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30).returncode == 0

    def _warm(self, phrases):
        made = 0
        for text in phrases:
            if self._render(text) is not None:
                made += 1
        if made:
            print(f"[voice] {made} phrase(s) ready in {self._cache}")

    # -- the fault tone ----------------------------------------------------
    def _make_tone(self):
        """A two-note alert, written once to a temp .wav.

        Synthesised rather than shipped so there is no binary asset to keep
        beside the code, and so it is there whatever the machine has.
        """
        rate = 22050
        frames = bytearray()
        for freq, secs in ((880.0, 0.16), (620.0, 0.26)):
            n = int(rate * secs)
            edge = max(rate // 100, 1)
            for i in range(n):
                # A short fade at each end, or the speaker clicks on every beep.
                fade = min(i, n - i, edge) / float(edge)
                value = math.sin(2 * math.pi * freq * i / rate) * 0.45 * fade
                frames += struct.pack("<h", int(value * 32767))
        path = os.path.join(tempfile.gettempdir(), "label-inspector-alert.wav")
        try:
            with wave.open(path, "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(rate)
                fh.writeframes(bytes(frames))
        except OSError:
            return None
        return path

    # -- the worker --------------------------------------------------------
    def _play(self, path):
        if not path:
            return
        try:
            subprocess.run([self.player, path] if self.player != "paplay"
                           else ["paplay", path],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass

    def _speak(self, text):
        """Say one line, whatever it takes.

        The good voice first. If it cannot be made -- the network went away
        mid-shift and this particular phrase was never cached -- fall back to
        espeak for this line only, into a scratch file rather than the cache,
        so the machine still says something and the cache is not poisoned
        with the voice nobody wanted.
        """
        path = self._render(text)
        if path is not None:
            self._play(path)
            return
        if self.engine == "espeak" or not _which("espeak-ng", "espeak"):
            return
        tmp = os.path.join(tempfile.gettempdir(),
                           f"label-inspector-say-{os.getpid()}.wav")
        try:
            if self._render_espeak_to(text, tmp, ESPEAK_FALLBACK_VOICE):
                self._play(tmp)
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            tone, lines = item
            if tone and self.tone_path:
                self._play(self.tone_path)
            for text in lines:
                self._speak(text)

    # -- the interface -----------------------------------------------------
    def _put(self, tone, lines, urgent):
        if urgent:
            while True:                  # a stale queue is worse than silence
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        try:
            self._queue.put_nowait((tone, [t for t in lines if t]))
        except queue.Full:
            pass

    def say(self, text, key=None, urgent=False):
        """Speak `text`, unless `key` matches the line spoken last."""
        if not self.enabled:
            return
        if key is not None and key == self._last_key:
            return
        self._last_key = key
        self._put(False, [text], urgent)

    def alert(self, text=None, lead=None, key=None):
        """The tone, then `lead`, then `text` -- jumping anything queued.

        `lead` is the phrase to keep short and fixed. It is pre-rendered, so
        it starts the moment the tone ends, and `text` -- which carries the
        row number and has to be synthesised the first time that row comes up
        -- is rendered while the lead is still being spoken.
        """
        if not self.enabled:
            return
        if key is not None and key == self._last_key:
            return
        self._last_key = key
        self._put(True, [lead, text], True)

    def close(self):
        if self._thread is not None:
            self._put(False, [], True)
            self._queue.put(None)
