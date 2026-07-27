from enum import Enum
import threading
import time
import queue
import numpy as np

from sonorium.obs import logger
import av

LOG_THRESHOLD = 500


class ExclusionGroupCoordinator:
    """
    Coordinates exclusive playback for tracks in a mutual exclusion group.

    When multiple tracks are marked as 'exclusive', only one can play at a time.
    Other exclusive tracks must wait until the playing track finishes AND a
    cooldown period has passed before they can play.

    Key behaviors:
    - On stream start, no exclusive track plays immediately (initial delay)
    - Only one exclusive track can play at a time
    - After a track finishes, there's a mandatory gap before any exclusive track plays
    - Avoids playing the same track twice in a row (unless it's the only exclusive track)

    This is shared across all streams in a ThemeStream.
    """

    # Minimum gap after an exclusive track finishes before another can start (seconds)
    # Increased from 30s to 120s to prevent exclusive tracks playing back-to-back
    MIN_GAP_AFTER_EXCLUSIVE = 120.0
    # Initial delay before any exclusive track can play on stream start
    INITIAL_DELAY = 60.0

    def __init__(self):
        self._lock = threading.Lock()
        self._playing_track: str | None = None  # Name of currently playing exclusive track
        self._play_end_time: float = 0  # When current track will finish
        self._last_played_track: str | None = None  # Track that played most recently
        self._cooldown_until: float = 0  # No exclusive track can play until this time
        self._registered_tracks: set[str] = set()  # All registered exclusive tracks
        self._start_time: float = time.time()  # When the coordinator was created

    def register_track(self, track_name: str):
        """Register an exclusive track with the coordinator."""
        with self._lock:
            self._registered_tracks.add(track_name)
            logger.debug(f'ExclusionGroup: Registered track "{track_name}" ({len(self._registered_tracks)} total)')

    def try_start_playing(self, track_name: str, duration_seconds: float) -> bool:
        """
        Attempt to start playing an exclusive track.

        Returns True if allowed to play, False otherwise.

        Conditions to play:
        1. Initial delay has passed since stream start
        2. No exclusive track is currently playing
        3. Cooldown period has passed since last track finished
        4. This is not the same track that just played (unless it's the only track)
        """
        with self._lock:
            now = time.time()

            # Check initial delay on stream start
            if now < self._start_time + self.INITIAL_DELAY:
                return False

            # Check if current playing track has finished
            if self._playing_track is not None:
                if now >= self._play_end_time:
                    # Track finished - start cooldown
                    self._last_played_track = self._playing_track
                    self._playing_track = None
                    self._cooldown_until = now + self.MIN_GAP_AFTER_EXCLUSIVE
                    logger.debug(f'ExclusionGroup: "{self._last_played_track}" finished, cooldown until +{self.MIN_GAP_AFTER_EXCLUSIVE}s')
                else:
                    # Track still playing
                    return False

            # Check cooldown period
            if now < self._cooldown_until:
                return False

            # Don't play same track twice in a row (unless only one exclusive track)
            if (self._last_played_track == track_name and
                len(self._registered_tracks) > 1):
                return False

            # All checks passed - start playing
            self._playing_track = track_name
            self._play_end_time = now + duration_seconds
            logger.debug(f'ExclusionGroup: "{track_name}" starting playback (duration: {duration_seconds:.1f}s)')
            return True

    def finish_playing(self, track_name: str):
        """Mark that an exclusive track has finished playing."""
        with self._lock:
            if self._playing_track == track_name:
                now = time.time()
                self._last_played_track = track_name
                self._playing_track = None
                self._play_end_time = 0
                self._cooldown_until = now + self.MIN_GAP_AFTER_EXCLUSIVE
                logger.debug(f'ExclusionGroup: "{track_name}" finished, cooldown until +{self.MIN_GAP_AFTER_EXCLUSIVE}s')

    def is_blocked(self, track_name: str) -> bool:
        """Check if a track is blocked from playing."""
        with self._lock:
            now = time.time()

            # Initial delay check
            if now < self._start_time + self.INITIAL_DELAY:
                return True

            # Check if current track has finished
            if self._playing_track is not None and now >= self._play_end_time:
                self._last_played_track = self._playing_track
                self._playing_track = None
                self._cooldown_until = now + self.MIN_GAP_AFTER_EXCLUSIVE

            # Blocked if another track is playing
            if self._playing_track is not None and self._playing_track != track_name:
                return True

            # Blocked during cooldown
            if now < self._cooldown_until:
                return True

            # Blocked if same track just played (and there are other options)
            if (self._last_played_track == track_name and
                len(self._registered_tracks) > 1):
                return True

            return False

    def get_wait_time(self) -> float:
        """Get seconds until this coordinator might allow a play."""
        with self._lock:
            now = time.time()

            # Initial delay
            if now < self._start_time + self.INITIAL_DELAY:
                return (self._start_time + self.INITIAL_DELAY) - now

            # Currently playing
            if self._playing_track is not None:
                remaining = self._play_end_time - now
                if remaining > 0:
                    return remaining + self.MIN_GAP_AFTER_EXCLUSIVE

            # In cooldown
            if now < self._cooldown_until:
                return self._cooldown_until - now

            return 0

    def get_track_count(self) -> int:
        """Get number of registered exclusive tracks."""
        with self._lock:
            return len(self._registered_tracks)


class PlaybackMode(str, Enum):
    """Playback mode for tracks.

    - CONTINUOUS: Track loops continuously with crossfade (default for long files)
    - SPARSE: Track plays once at full volume, then silence for an interval before repeating
    - PRESENCE: Track fades in/out of the mix based on presence value
    - AUTO: Automatically choose based on file length and presence setting
    """
    CONTINUOUS = "continuous"
    SPARSE = "sparse"
    PRESENCE = "presence"
    AUTO = "auto"

# Threshold for "short" audio files that get sparse playback
SHORT_FILE_THRESHOLD_SECONDS = 15.0
# Sparse playback interval range (seconds between plays)
SPARSE_MIN_INTERVAL = 180.0   # 3 minutes at 100% presence
SPARSE_MAX_INTERVAL = 1800.0  # 30 minutes at ~0% presence
# Variance applied to intervals (±30%)
SPARSE_INTERVAL_VARIANCE = 0.30

# Crossfade duration in seconds for loop transitions
LOOP_CROSSFADE_DURATION = 8
# Fade duration for tracks fading in/out of the mix
TRACK_FADE_DURATION = 6.0
# Sample rate
SAMPLE_RATE = 44100
# Calculated sample counts
CROSSFADE_SAMPLES = int(LOOP_CROSSFADE_DURATION * SAMPLE_RATE)
TRACK_FADE_SAMPLES = int(TRACK_FADE_DURATION * SAMPLE_RATE)


class RecordingMetadata:
    """
    Represents file, metadata, etc. The non-state stuff, on disk. One per file. Immutable
    """

    def __init__(self, path):
        self.path = path
        self._duration_samples = None
        self._cached_audio = None  

    def get_instance(self, theme=None):
        return RecordingThemeInstance(self, theme=theme)

    @property
    def name(self):
        return self.path.stem
    
    def get_audio_data(self) -> np.ndarray:
        """Decodes, resamples, and caches the entire audio file into RAM once."""
        if self._cached_audio is not None:
            return self._cached_audio

        logger.info(f"Caching audio into RAM for {self.name}...")
        resampler = av.AudioResampler(format='s16', layout='mono', rate=SAMPLE_RATE)
        
        try:
            container = av.open(self.path)
            if len(container.streams.audio) == 0:
                container.close()
                self._cached_audio = np.zeros(0, dtype=np.float32)
                return self._cached_audio

            stream = next(iter(container.streams.audio))
            samples = []

            for frame_orig in container.decode(stream):
                for frame_resamp in resampler.resample(frame_orig):
                    data = frame_resamp.to_ndarray()
                    data = data.mean(axis=0).astype(np.float32)
                    samples.append(data.flatten())

            container.close()

            if not samples:
                self._cached_audio = np.zeros(0, dtype=np.float32)
            else:
                self._cached_audio = np.concatenate(samples)
                self._duration_samples = len(self._cached_audio)
                
        except Exception as e:
            logger.error(f"Failed to decode {self.path}: {e}")
            self._cached_audio = np.zeros(0, dtype=np.float32)
            
        return self._cached_audio

    @property
    def duration_samples(self):
        """Get total duration in samples"""
        if self._duration_samples is None:
            # Getting the audio data will automatically set the exact sample count
            self.get_audio_data()
        return self._duration_samples

    @property
    def duration_seconds(self):
        """Get total duration in seconds"""
        return self.duration_samples / SAMPLE_RATE

    def is_short_file(self, threshold: float = SHORT_FILE_THRESHOLD_SECONDS) -> bool:
        """Check if this is a short audio file that should use sparse playback"""
        return self.duration_seconds < threshold


class BufferedAudioStream:
    """
    Wraps any synchronous audio stream with a background producer thread
    and a thread-safe queue to absorb CPU jitter and prevent audio gaps.

    Many low-power Home Assistant devices reuse the same numpy buffer object
    for streamed chunks, so we copy before enqueueing to avoid outdated
    buffer contents being read by the consumer.
    """
    CHUNK_SIZE = 4096
    DEFAULT_QUEUE_SIZE = 64
    QUEUE_GET_TIMEOUT = 5.0

    def __init__(self, inner_stream, max_queue_size: int | None = None):
        self.inner_stream = inner_stream
        self.queue = queue.Queue(maxsize=max_queue_size or self.DEFAULT_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._producer_loop, daemon=True)
        self._worker.start()

    def _producer_loop(self):
        try:
            for chunk in self.inner_stream:
                if self._stop_event.is_set():
                    break
                while not self._stop_event.is_set():
                    try:
                        self.queue.put(chunk.copy(), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop_event.is_set():
                    break
        except Exception as e:
            logger.error(f"BufferedAudioStream error in producer thread: {e}")

    def close(self):
        self._stop_event.set()
        try:
            if self._worker.is_alive():
                self._worker.join(timeout=1.0)
        except Exception:
            pass
        if hasattr(self.inner_stream, 'close'):
            try:
                self.inner_stream.close()
            except Exception:
                pass

    @property
    def instance(self):
        return getattr(self.inner_stream, 'instance', None)

    def __getattr__(self, name):
        return getattr(self.inner_stream, name)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.queue.get(timeout=self.QUEUE_GET_TIMEOUT)
        except queue.Empty:
            # Fallback silence chunk to avoid dropping audio output entirely during heavy stalls
            fallback = np.zeros((1, self.CHUNK_SIZE), dtype=np.int16)
            logger.warning("BufferedAudioStream queue starved, emitting safety silence chunk")
            return fallback


class RecordingThemeInstance:
    """
    Wraps the metadata, but with some extra state, to represent how that recording is set up within a given theme.
    Every theme gets one of these for each recording.
    """

    def __init__(self, meta: RecordingMetadata, theme=None):
        self.meta = meta
        self.theme = theme  # Reference to parent ThemeDefinition for threshold access
        self.volume = 1.0  # Amplitude multiplier (keep at 1.0 for now)
        self.presence = 1.0  # How often this track plays: 1.0 = always, 0.5 = half the time, 0 = never
        self.is_enabled = True  # Master enable/disable (mute)
        self.crossfade_enabled = True  # Enable crossfade looping by default
        self.playback_mode = PlaybackMode.AUTO  # How playback/looping is handled
        self.exclusive = False  # If True, only one exclusive track can play at a time

    @property
    def short_file_threshold(self) -> float:
        """Get the short file threshold from theme, or use default"""
        if self.theme is not None:
            return self.theme.short_file_threshold
        return SHORT_FILE_THRESHOLD_SECONDS

    def _resolve_playback_mode(self) -> PlaybackMode:
        """Resolve AUTO mode to an actual playback mode based on file characteristics."""
        if self.playback_mode != PlaybackMode.AUTO:
            return self.playback_mode

        # AUTO logic: short files use sparse, long files use presence (if < 1.0) or continuous
        if self.meta.is_short_file(self.short_file_threshold):
            return PlaybackMode.SPARSE if self.presence < 1.0 else PlaybackMode.CONTINUOUS
        else:
            return PlaybackMode.PRESENCE if self.presence < 1.0 else PlaybackMode.CONTINUOUS

    def get_stream(self, exclusion_coordinator: ExclusionGroupCoordinator = None):
        mode = self._resolve_playback_mode()

        # SPARSE: Play once at full volume, then silence for interval
        if mode == PlaybackMode.SPARSE:
            stream = SparsePlaybackStream(self, exclusion_coordinator)
        else:
            # CONTINUOUS or PRESENCE: Start with base looping stream
            if self.crossfade_enabled:
                base_stream = CrossfadeRecordingStream(self)
            else:
                base_stream = RecordingThemeStream(self)

            # PRESENCE: Wrap with fade in/out based on presence value
            if mode == PlaybackMode.PRESENCE and self.presence < 1.0:
                stream = PresenceMixingStream(base_stream, self)
            else:
                stream = base_stream

        # Wrap the stream in a background buffered queue to decouple generation from consumption
        return BufferedAudioStream(stream)

    @property
    def name(self):
        return self.meta.name


class RecordingThemeStream:
    """
    Basic recording stream without crossfade - loops with hard cut from RAM cache.
    Optimized with a pre-allocated output buffer for zero-copy yielding.
    """
    CHUNK_SIZE = 4096

    def __init__(self, instance: RecordingThemeInstance):
        self.instance = instance
        # Pre-allocate the yield buffer once
        self.out_buffer = np.empty((1, self.CHUNK_SIZE), dtype=np.int16)
        self.gen = self._gen()

    def _gen(self):
        audio_data = self.instance.meta.get_audio_data()
        
        if len(audio_data) == 0:
            self.out_buffer.fill(0)
            while True:
                yield self.out_buffer
                
        # Pre-apply volume and convert to int16 immediately
        audio_data = audio_data * self.instance.volume
        audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
        
        pos = 0
        total_samples = len(audio_data)
        i = 0
        
        while True:
            chunk_end = pos + self.CHUNK_SIZE
            
            if chunk_end <= total_samples:
                # Copy directly into the pre-allocated buffer
                self.out_buffer[0, :] = audio_data[pos:chunk_end]
                pos += self.CHUNK_SIZE
            else:
                rem = total_samples - pos
                self.out_buffer[0, :rem] = audio_data[pos:]
                self.out_buffer[0, rem:] = audio_data[:chunk_end - total_samples]
                pos = chunk_end - total_samples
            
            yield self.out_buffer
            
            if i % LOG_THRESHOLD == 0:
                logger.debug(f'{self.__class__.__name__} Yielding chunk #{i}')
            i += 1

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def close(self):
        if self.gen is not None:
            try:
                self.gen.close()
            except Exception:
                pass
            self.gen = None


class CrossfadeRecordingStream:
    """
    Recording stream with crossfade looping - seamlessly blends end of track into beginning using zero-copy in-place math arrays.
    """
    CHUNK_SIZE = 4096

    def __init__(self, instance: RecordingThemeInstance):
        self.instance = instance
        # Pre-allocate math buffers to avoid runtime garbage collection
        self.mix_buffer = np.empty(self.CHUNK_SIZE, dtype=np.float32)
        self.temp_buffer = np.empty(self.CHUNK_SIZE, dtype=np.float32)
        self.out_buffer = np.empty((1, self.CHUNK_SIZE), dtype=np.int16)
        self.gen = self._gen()

    def _gen(self):
        audio_data = self.instance.meta.get_audio_data()
        
        if len(audio_data) == 0:
            self.out_buffer.fill(0)
            while True:
                yield self.out_buffer
        
        # Pre-apply volume to the float cache 
        audio_data = (audio_data * self.instance.volume).astype(np.float32)
        track_duration = len(audio_data)
        
        actual_crossfade_samples = min(CROSSFADE_SAMPLES, track_duration // 2)
        crossfade_start = track_duration - actual_crossfade_samples
        
        logger.info(f'CrossfadeStream: {self.instance.name} duration={track_duration} samples, crossfade at {crossfade_start}')
        
        fade_out = np.cos(np.linspace(0, np.pi/2, actual_crossfade_samples)).astype(np.float32)
        fade_in = np.sin(np.linspace(0, np.pi/2, actual_crossfade_samples)).astype(np.float32)
        
        pos = 0
        chunk_count = 0
        
        while True:
            chunk_end = pos + self.CHUNK_SIZE
            
            # Case 1: Entire chunk is before the crossfade
            if chunk_end <= crossfade_start:
                self.mix_buffer[:] = audio_data[pos:chunk_end]
                pos += self.CHUNK_SIZE
                
            # Case 2: Entire chunk is inside the crossfade
            elif pos >= crossfade_start:
                fade_start_idx = pos - crossfade_start
                fade_end_idx = fade_start_idx + self.CHUNK_SIZE
                
                if fade_end_idx <= actual_crossfade_samples:
                    # In-place multiply and add
                    np.multiply(audio_data[pos:pos+self.CHUNK_SIZE], fade_out[fade_start_idx:fade_end_idx], out=self.mix_buffer)
                    np.multiply(audio_data[fade_start_idx:fade_end_idx], fade_in[fade_start_idx:fade_end_idx], out=self.temp_buffer)
                    np.add(self.mix_buffer, self.temp_buffer, out=self.mix_buffer)
                    
                    pos += self.CHUNK_SIZE
                    if pos >= track_duration:
                        pos = actual_crossfade_samples
                else:
                    # Chunk splits exactly across the end of the file
                    rem_samples = track_duration - pos
                    
                    np.multiply(audio_data[pos:track_duration], fade_out[fade_start_idx:actual_crossfade_samples], out=self.temp_buffer[:rem_samples])
                    np.multiply(audio_data[fade_start_idx:actual_crossfade_samples], fade_in[fade_start_idx:actual_crossfade_samples], out=self.mix_buffer[:rem_samples])
                    np.add(self.temp_buffer[:rem_samples], self.mix_buffer[:rem_samples], out=self.mix_buffer[:rem_samples])
                    
                    pos = actual_crossfade_samples
                    remaining_chunk = self.CHUNK_SIZE - rem_samples
                    self.mix_buffer[rem_samples:] = audio_data[pos:pos+remaining_chunk]
                    pos += remaining_chunk
                    
            # Case 3: Chunk starts normal, but ends inside the crossfade
            else:
                normal_samples = crossfade_start - pos
                fade_samples = self.CHUNK_SIZE - normal_samples
                
                self.mix_buffer[:normal_samples] = audio_data[pos:crossfade_start]
                
                np.multiply(audio_data[crossfade_start:crossfade_start+fade_samples], fade_out[:fade_samples], out=self.temp_buffer[:fade_samples])
                np.multiply(audio_data[:fade_samples], fade_in[:fade_samples], out=self.mix_buffer[normal_samples:])
                np.add(self.temp_buffer[:fade_samples], self.mix_buffer[normal_samples:], out=self.mix_buffer[normal_samples:])
                
                pos += self.CHUNK_SIZE

            # In-place clipping and integer casting
            np.clip(self.mix_buffer, -32768, 32767, out=self.mix_buffer)
            self.out_buffer[0, :] = self.mix_buffer
            
            chunk_count += 1
            if chunk_count % LOG_THRESHOLD == 0:
                logger.debug(f'CrossfadeStream: chunk #{chunk_count}, samples={pos}')
                
            yield self.out_buffer

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def close(self):
        if self.gen is not None:
            try:
                self.gen.close()
            except Exception:
                pass
            self.gen = None


class SparsePlaybackStream:
    """
    Stream for short audio files (< 15 seconds) that plays the file once,
    then outputs silence for a randomized interval before playing again.

    This prevents short sounds (like a horse whinny) from looping repeatedly.
    The interval between plays is randomized based on presence:
    - presence=1.0: Plays continuously (not sparse - use regular stream)
    - presence=0.5: Interval is middle of range (~165 seconds average)
    - presence=0.1: Interval is near max (~270 seconds average)

    The file plays once with fade in/out, then silence until next play.

    If the track is marked as 'exclusive' and an ExclusionGroupCoordinator is
    provided, only one exclusive track can play at a time. Other exclusive
    tracks output silence until the playing track finishes.
    """
    CHUNK_SIZE = 4096

    def __init__(self, instance: RecordingThemeInstance, exclusion_coordinator=None):
        self.instance = instance
        self.exclusion_coordinator = exclusion_coordinator
        # Pre-allocate math buffers
        self.mix_buffer = np.empty(self.CHUNK_SIZE, dtype=np.float32)
        self.out_buffer = np.empty((1, self.CHUNK_SIZE), dtype=np.int16)
        
        if self.instance.exclusive and self.exclusion_coordinator is not None:
            self.exclusion_coordinator.register_track(self.instance.name)

        self.gen = self._gen()

    def _gen(self):
        import random

        presence = self.instance.presence
        file_duration_seconds = self.instance.meta.duration_seconds

        logger.info(f'SparsePlaybackStream: {self.instance.name} - short file ({file_duration_seconds:.1f}s)')

        fade_duration = min(TRACK_FADE_DURATION, file_duration_seconds / 3)
        fade_samples = int(fade_duration * SAMPLE_RATE)
        fade_in_curve = np.sin(np.linspace(0, np.pi/2, fade_samples)).astype(np.float32)
        fade_out_curve = np.cos(np.linspace(0, np.pi/2, fade_samples)).astype(np.float32)

        def get_silent_interval():
            factor = 1.0 - presence
            base_interval = SPARSE_MIN_INTERVAL + (SPARSE_MAX_INTERVAL - SPARSE_MIN_INTERVAL) * factor
            variance_min = 1.0 - SPARSE_INTERVAL_VARIANCE
            variance_max = 1.0 + SPARSE_INTERVAL_VARIANCE
            final_interval = base_interval * random.uniform(variance_min, variance_max)
            return int(final_interval * SAMPLE_RATE)

        def is_blocked_exclusive():
            if not self.instance.exclusive or self.exclusion_coordinator is None: return False
            return self.exclusion_coordinator.is_blocked(self.instance.name)

        def try_start_exclusive():
            if not self.instance.exclusive or self.exclusion_coordinator is None: return True
            return self.exclusion_coordinator.try_start_playing(self.instance.name, file_duration_seconds)

        def finish_exclusive():
            if self.instance.exclusive and self.exclusion_coordinator is not None:
                self.exclusion_coordinator.finish_playing(self.instance.name)

        def get_block_wait_chunks():
            if self.exclusion_coordinator is not None:
                wait_time = self.exclusion_coordinator.get_wait_time()
                if wait_time > 0:
                    wait_time += random.uniform(0.5, 3.0)
                    return int(wait_time * SAMPLE_RATE / self.CHUNK_SIZE)
            return int(random.uniform(1.0, 3.0) * SAMPLE_RATE / self.CHUNK_SIZE)

        chunk_count = 0
        silence_chunk = np.zeros((1, self.CHUNK_SIZE), dtype=np.int16)
        first_play = True

        while True:
            presence = self.instance.presence

            if first_play:
                first_play = False
                initial_delay_samples = int(get_silent_interval() * random.uniform(0.0, 1.0))
                initial_delay_chunks = initial_delay_samples // self.CHUNK_SIZE
                if initial_delay_chunks > 0:
                    for _ in range(initial_delay_chunks):
                        chunk_count += 1
                        yield silence_chunk

            if is_blocked_exclusive() or not try_start_exclusive():
                wait_chunks = get_block_wait_chunks()
                for _ in range(wait_chunks):
                    chunk_count += 1
                    yield silence_chunk
                continue

            # Instead of copying the whole array, reference it dynamically
            audio_data = self.instance.meta.get_audio_data()

            if len(audio_data) > 0:
                pos = 0
                while pos < len(audio_data):
                    chunk_end = min(pos + self.CHUNK_SIZE, len(audio_data))
                    chunk_len = chunk_end - pos

                    # Pull in chunk and apply volume dynamically
                    np.multiply(audio_data[pos:chunk_end], self.instance.volume, out=self.mix_buffer[:chunk_len])

                    # In-place fade-in
                    if pos < fade_samples:
                        fade_end = min(chunk_len, fade_samples - pos)
                        np.multiply(self.mix_buffer[:fade_end], fade_in_curve[pos:pos+fade_end], out=self.mix_buffer[:fade_end])

                    # In-place fade-out
                    fade_out_start_pos = len(audio_data) - fade_samples
                    if chunk_end > fade_out_start_pos:
                        buf_start = max(0, fade_out_start_pos - pos)
                        curve_start = pos + buf_start - fade_out_start_pos
                        overlap_len = chunk_len - buf_start
                        np.multiply(self.mix_buffer[buf_start:chunk_len], fade_out_curve[curve_start:curve_start+overlap_len], out=self.mix_buffer[buf_start:chunk_len])

                    # Pad end if chunk is smaller than CHUNK_SIZE
                    if chunk_len < self.CHUNK_SIZE:
                        self.mix_buffer[chunk_len:].fill(0)

                    # In-place clip and assign
                    np.clip(self.mix_buffer, -32768, 32767, out=self.mix_buffer)
                    self.out_buffer[0, :] = self.mix_buffer
                    
                    chunk_count += 1
                    yield self.out_buffer
                    pos += self.CHUNK_SIZE

            finish_exclusive()

            silent_samples = get_silent_interval()
            silent_chunks = silent_samples // self.CHUNK_SIZE

            for _ in range(silent_chunks):
                chunk_count += 1
                yield silence_chunk

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def close(self):
        if self.gen is not None:
            try:
                self.gen.close()
            except Exception:
                pass
            self.gen = None


class PresenceMixingStream:
    """
    Wrapper stream that controls track presence in the mix.

    Instead of controlling amplitude, the 'presence' value (0.0-1.0) controls
    how often this track is audible in the mix:
    - presence=1.0: Track plays continuously (always in mix)
    - presence=0.5: Track plays ~50% of the time, fading in/out
    - presence=0.0: Track never plays (always silent)

    Uses randomized timing so tracks don't all fade in/out together.
    """
    CHUNK_SIZE = 4096

    def __init__(self, base_stream, instance: RecordingThemeInstance):
        self.base_stream = base_stream
        self.instance = instance
        # Pre-allocate math buffers
        self.float_buffer = np.empty((1, self.CHUNK_SIZE), dtype=np.float32)
        self.out_buffer = np.empty((1, self.CHUNK_SIZE), dtype=np.int16)
        self.gen = self._gen()

    def _gen(self):
        import random

        is_active = True  
        current_gain = 1.0 if self.instance.presence >= 1.0 else 0.0
        target_gain = 1.0 if self.instance.presence >= 1.0 else 0.0
        fade_position = 0
        samples_until_change = 0

        min_active_duration = int(30 * SAMPLE_RATE)  
        max_active_duration = int(120 * SAMPLE_RATE)  
        min_inactive_duration = int(20 * SAMPLE_RATE)  
        max_inactive_duration = int(90 * SAMPLE_RATE)  

        def get_next_duration(presence, is_active):
            if presence >= 1.0: return float('inf')  
            if presence <= 0.0: return float('inf')  

            if is_active:
                base_duration = min_active_duration + (max_active_duration - min_active_duration) * presence
                return int(base_duration * random.uniform(0.7, 1.3))
            else:
                base_duration = max_inactive_duration - (max_inactive_duration - min_inactive_duration) * presence
                return int(base_duration * random.uniform(0.7, 1.3))

        presence = self.instance.presence
        if presence >= 1.0:
            is_active = True
            current_gain = 1.0
            target_gain = 1.0
        elif presence <= 0.0:
            is_active = False
            current_gain = 0.0
            target_gain = 0.0
        else:
            is_active = random.random() < presence
            current_gain = 1.0 if is_active else 0.0
            target_gain = current_gain

        samples_until_change = get_next_duration(presence, is_active)
        chunk_count = 0

        while True:
            try:
                chunk = next(self.base_stream)
            except StopIteration:
                return

            new_presence = self.instance.presence
            if new_presence != presence:
                presence = new_presence
                if presence >= 1.0 and target_gain < 1.0:
                    target_gain = 1.0
                    fade_position = 0
                elif presence <= 0.0 and target_gain > 0.0:
                    target_gain = 0.0
                    fade_position = 0

            samples_until_change -= self.CHUNK_SIZE
            if samples_until_change <= 0 and 0 < presence < 1.0:
                is_active = not is_active
                target_gain = 1.0 if is_active else 0.0
                fade_position = 0
                samples_until_change = get_next_duration(presence, is_active)

            if current_gain != target_gain:
                fade_progress = min(1.0, fade_position / TRACK_FADE_SAMPLES)
                
                if target_gain > current_gain:
                    applied_gain = np.sin(fade_progress * np.pi / 2)
                else:
                    applied_gain = np.cos(fade_progress * np.pi / 2)

                fade_position += self.CHUNK_SIZE

                if fade_progress >= 1.0:
                    current_gain = target_gain
                    applied_gain = target_gain
            else:
                applied_gain = current_gain

            # Perform high-performance bypass or zero-copy math
            if applied_gain == 1.0:
                yield chunk  
            elif applied_gain == 0.0:
                self.out_buffer.fill(0)
                yield self.out_buffer
            else:
                # Copy into float buffer, multiply, clip, and recast—all in-place without creating new arrays
                np.copyto(self.float_buffer, chunk)
                np.multiply(self.float_buffer, applied_gain, out=self.float_buffer)
                np.clip(self.float_buffer, -32768, 32767, out=self.float_buffer)
                
                self.out_buffer[:] = self.float_buffer
                yield self.out_buffer

            chunk_count += 1

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def close(self):
        if self.base_stream is not None and hasattr(self.base_stream, 'close'):
            try:
                self.base_stream.close()
            except Exception:
                pass
        if self.gen is not None:
            try:
                self.gen.close()
            except Exception:
                pass
            self.gen = None