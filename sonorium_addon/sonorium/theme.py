import re
import time
from functools import cached_property

import av
import numpy as np

from sonorium.obs import logger
from sonorium.recording import LOG_THRESHOLD, ExclusionGroupCoordinator, RecordingThemeStream
from sonorium.utils import IndexList


def sanitize(text: str) -> str:
    """Sanitize a string to be safe for use as an ID/filename."""
    # Replace spaces and special chars with underscores
    text = re.sub(r'[^\w\-]', '_', text.lower())
    # Remove consecutive underscores
    text = re.sub(r'_+', '_', text)
    # Strip leading/trailing underscores
    return text.strip('_')

# Default output gain multiplier (now controlled via device.master_volume)
DEFAULT_OUTPUT_GAIN = 6.0

# Default threshold for short file detection (seconds)
DEFAULT_SHORT_FILE_THRESHOLD = 15.0


class ThemeDefinition:
    """

    Run-time only. A ephemeral mix defined by the user.

    ThemeDefinition: What recordings are involved, volumes. User defines these via the UI, then selects a media player entity to stream from it.
    ThemeStream: One instance per client/connection. Has a RecordingStream for each recording in the ThemeDefinition.

    When a user selectes a media player for this theme, then clicks play, HA tells the player to play URL /theme/name.
     - On the API side, the ThemeDefinition with ID "name" is selected, and a new ThemeStream initialized.

    When a user modifies a themeDefinition, like change recording volume, all live ThemeStreams are updated.

    Every ThemeDefinition needs an inited RecordingStream for each recording. That way we can have per-theme, per-recording state (volume, playing, etc).
    Not really. Cos each connection needs its own Stream.


    recording (immutable, one per-path) -> recording_instance (mutable, contains addition vol, is_enabled, etc) -> recording_stream (one per-connection)

    """

    def __init__(self, sonorium, name, theme_id: str = None):
        self.sonorium = sonorium
        self.name = name
        # Use provided UUID, or fall back to sanitized folder name for backwards compatibility
        self._theme_id = theme_id

        # Short file threshold (seconds) - files shorter than this use sparse playback
        # Can be customized per theme via metadata.json
        self.short_file_threshold = DEFAULT_SHORT_FILE_THRESHOLD

        # Use theme-specific recordings instead of all recordings
        if name in self.sonorium.theme_metas:
            theme_metas = self.sonorium.theme_metas[name]
        else:
            # Fallback to all recordings for backwards compatibility
            theme_metas = self.sonorium.metas

        # Pass theme reference to instances so they can access threshold
        self.instances = IndexList(meta.get_instance(theme=self) for meta in theme_metas)

        self.streams: list[ThemeStream] = []

    @cached_property
    def url(self) -> str:
        from sonorium.settings import settings
        return f'{settings.stream_url}/stream/{self.id}'

    @property
    def id(self):
        """Return the theme UUID from metadata.json, or sanitized name as fallback."""
        return self._theme_id if self._theme_id else sanitize(self.name)


    def get_stream(self):
        theme = ThemeStream(self)
        self.streams.append(theme)
        logger.info(f'ThemeDefinition {self.name}: Created new ThemeStream (total: {len(self.streams)} streams)')
        return theme

    def close_all_streams(self):
        for stream in list(self.streams):
            try:
                stream.close()
            except Exception:
                pass
        self.streams.clear()


class ThemeStream:
    """

    Run-time only. A ephemeral mix defined by the user.

    ThemeDefinition: What recordings are involved, volumes. User defines these via the UI, then selects a media player entity to stream from it.
    ThemeStream: One instance per client/connection. Has a RecordingStream for each recording in the ThemeDefinition.

    When a user selectes a media player for this theme, then clicks play, HA tells the player to play URL /theme/name.
     - On the API side, the ThemeDefinition with ID "name" is selected, and a new ThemeStream initialized.

    When a user modifies a themeDefinition, like change recording volume, all live ThemeStreams are updated.

    """

    def __init__(self, theme_def: ThemeDefinition):
        self.theme_def = theme_def

        # Create shared exclusion coordinator for tracks marked as exclusive
        self.exclusion_coordinator = ExclusionGroupCoordinator()

        # Create streams, passing the exclusion coordinator
        self.recording_streams = [
            instance.get_stream(exclusion_coordinator=self.exclusion_coordinator)
            for instance in theme_def.instances
        ]

    @cached_property
    def chunk_silence(self):
        data = np.zeros((1, RecordingThemeStream.CHUNK_SIZE), np.int16)
        return data

    def iter_chunks(self):
        chunk_size = RecordingThemeStream.CHUNK_SIZE
        mix_buffer = np.empty(chunk_size, dtype=np.float32)
        out_buffer = np.empty((1, chunk_size), dtype=np.int16)

        try:
            while True:
                data_recs = []
                for stream in self.recording_streams:
                    inst = getattr(stream, 'instance', None)
                    if inst is None or getattr(inst, 'is_enabled', True):
                        data_recs.append(next(stream))

                if not data_recs:
                    # logger.debug(f'Theme "{self.theme_def.name}" has no enabled recordings. Streaming silence...')
                    data_recs.append(self.chunk_silence)

                mix_buffer.fill(0.0)
                for data in data_recs:
                    mix_buffer += data[0].astype(np.float32)

                # Soft clipping / normalization to prevent distortion
                # Divide by sqrt(n) for a good balance between volume and avoiding clipping
                n_tracks = len(data_recs)
                if n_tracks > 1:
                    mix_buffer /= np.sqrt(n_tracks)

                # Apply output gain boost (use device master_volume if available)
                output_gain = getattr(self.theme_def.sonorium, 'master_volume', DEFAULT_OUTPUT_GAIN)
                mix_buffer *= output_gain

                np.clip(mix_buffer, -32768, 32767, out=mix_buffer)
                out_buffer[0, :] = mix_buffer.astype(np.int16)
                yield out_buffer
        finally:
            for stream in self.recording_streams:
                if hasattr(stream, 'close'):
                    try:
                        stream.close()
                    except Exception:
                        pass
            self.recording_streams.clear()

    def close(self):
        for stream in list(self.recording_streams):
            if hasattr(stream, 'close'):
                try:
                    stream.close()
                except Exception:
                    pass
        self.recording_streams.clear()
        if self in self.theme_def.streams:
            try:
                self.theme_def.streams.remove(self)
            except ValueError:
                pass

    def __iter__(self):
        output = av.open(file='.mp3', mode="w")
        bitrate = 128_000
        out_stream = output.add_stream(codec_name='mp3', rate=44100, bit_rate=bitrate)
        iter_chunks = self.iter_chunks()

        start_time = time.time()
        audio_time = 0.0  # total audio duration sent

        try:
            while True:
                for i, data in enumerate(iter_chunks):
                    frame = av.AudioFrame.from_ndarray(data, format='s16', layout='mono')
                    frame.rate = 44100

                    frame_duration = frame.samples / frame.rate
                    audio_time += frame_duration

                    for packet in out_stream.encode(frame):
                        packet_bytes = bytes(packet)
                        yield packet_bytes

                    # Only sleep if we are ahead of real-time
                    now = time.time()
                    ahead = audio_time - (now - start_time)
                    if ahead > 0:
                        time.sleep(ahead)

                    if i % LOG_THRESHOLD == 0:
                        logger.debug(f'Waiting {ahead:.5f} seconds to maintain real-time pacing {audio_time=}...')


        finally:
            try:
                output.close()
            except Exception:
                pass
            logger.info('Closing transcoder...')
            try:
                iter_chunks.close()
            except Exception:
                pass
            output.close()
