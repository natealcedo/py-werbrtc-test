import asyncio
import cv2
import json
import logging
import websockets
from av import VideoFrame, AudioFrame
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc import AudioStreamTrack
import pyaudio
import numpy as np
import fractions
import webrtcvad
from scipy.signal import resample


# Configure logging
logging.basicConfig(level=logging.INFO)  # Changed to DEBUG for more details
logger = logging.getLogger("webrtc_server")


class VideoCamera(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self._active_tracks = 0
        self._lock = asyncio.Lock()
        self.cap = None
        logger.info("VideoCamera instance created")

    async def ensure_camera_initialized(self):
        if self.cap is None:
            logger.info("Initializing camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                raise RuntimeError("Could not open video device")

            # Set video properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Changed to 30 FPS for better compatibility
            logger.info("Camera initialized successfully")

    async def add_track(self):
        async with self._lock:
            self._active_tracks += 1
            await self.ensure_camera_initialized()
            logger.info(f"Track added. Active tracks: {self._active_tracks}")

    async def remove_track(self):
        async with self._lock:
            self._active_tracks -= 1
            logger.info(f"Track removed. Active tracks: {self._active_tracks}")
            if self._active_tracks <= 0:
                self.stop()

    async def recv(self):
        async with self._lock:
            try:
                await self.ensure_camera_initialized()
                pts, time_base = await self.next_timestamp()

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    return None

                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create VideoFrame
                video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = pts
                video_frame.time_base = time_base

                return video_frame
            except Exception as e:
                logger.error(f"Error in recv: {str(e)}")
                return None

    def stop(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

class AudioRecorder(AudioStreamTrack):
    def __init__(self):
        super().__init__()
        self.pa = None  # PyAudio instance
        self.sample_rate = 48000  # WebRTC standard sample rate
        self.stream = None
        self.time_base = fractions.Fraction(1, self.sample_rate)
        self.pts = 0
        self._lock = asyncio.Lock()
        self.vad = webrtcvad.Vad()  # Voice Activity Detection
        self.vad.set_mode(1)  # Less aggressive noise suppression
        logger.info("AudioRecorder initialized with improved audio processing")

    def ensure_pyaudio_initialized(self):
        """Ensure that the PyAudio instance is initialized."""
        if self.pa is None:
            self.pa = pyaudio.PyAudio()
            logger.info("PyAudio instance initialized")

    def start_stream(self):
        """Ensure the audio stream is started."""
        self.ensure_pyaudio_initialized()
        if self.stream is None or not self.stream.is_active():
            self.stream = self.pa.open(
                format=pyaudio.paInt16,  # 16-bit audio
                channels=1,             # Mono audio
                rate=self.sample_rate,  # Sample rate (48 kHz for WebRTC)
                input=True,
                frames_per_buffer=960,  # Buffer size for 20 ms of audio at 48 kHz
            )
            logger.info("Audio stream started")

    def _apply_noise_reduction(self, audio_data):
        """Apply optional noise reduction and voice activity detection (VAD)."""
        try:
            # Convert raw audio to NumPy array
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)

            # Check for voice activity using VAD (16kHz required)
            downsampled_audio = resample(audio_samples, len(audio_samples) * 16000 // self.sample_rate)
            if self.vad.is_speech(downsampled_audio.astype(np.int16).tobytes(), 16000):
                # Voice detected; return original audio
                return audio_data
            else:
                # Silence detected; return zeroed audio
                return b'\x00' * len(audio_data)
        except Exception as e:
            logger.warning(f"Error during noise reduction: {e}")
            return audio_data

    async def recv(self):
        """Receive audio samples from the microphone."""
        try:
            async with self._lock:
                # Ensure the stream is initialized and active
                if self.stream is None or not self.stream.is_active():
                    self.start_stream()

                # Read raw audio data
                audio_data = self.stream.read(960, exception_on_overflow=False)

                # Apply optional noise reduction
                cleaned_audio = self._apply_noise_reduction(audio_data)

                # Convert raw audio to NumPy array (16-bit integers)
                audio_samples = np.frombuffer(cleaned_audio, dtype=np.int16)

                # Create an AV AudioFrame
                frame = AudioFrame(format="s16", layout="mono", samples=len(audio_samples))
                frame.sample_rate = self.sample_rate
                frame.planes[0].update(audio_samples.tobytes())

                # Set the PTS and time_base for the frame
                frame.pts = self.pts
                frame.time_base = self.time_base

                # Increment the PTS by the number of samples
                self.pts += len(audio_samples)

                return frame
        except Exception as e:
            logger.error(f"Error in AudioRecorder.recv: {e}")
            return None

    async def stop(self):
        """Stop the audio stream and clean up resources."""
        async with self._lock:
            if self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                logger.info("Audio stream stopped")
            if self.pa:
                self.pa.terminate()
                self.pa = None
                logger.info("PyAudio terminated")


# Add audio handling to WebRTCServer
class WebRTCServer:
    def __init__(self):
        self.pcs = set()
        self.camera = None
        self.microphone = None
        self._camera_lock = asyncio.Lock()
        self.client_count = 0
        logger.info("WebRTC server initialized")

    async def websocket_handler(self, websocket):
        self.client_count += 1
        client_id = self.client_count
        logger.info(f"New WebSocket connection. Client ID: {client_id}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message from client {client_id}: {data['type']}")

                    if data["type"] == "offer":
                        logger.info(f"Processing offer from client {client_id}")
                        pc = RTCPeerConnection(RTCConfiguration(
                            iceServers=[
                                RTCIceServer(urls="stun:stun.l.google.com:19302")
                            ]
                        ))

                        @pc.on("connectionstatechange")
                        async def on_connectionstatechange():
                            logger.info(f"Client {client_id} - Connection state: {pc.connectionState}")
                            if pc.connectionState == "failed":
                                await self.cleanup_pc(pc, client_id)

                        @pc.on("iceconnectionstatechange")
                        async def on_iceconnectionstatechange():
                            logger.info(f"Client {client_id} - ICE state: {pc.iceConnectionState}")

                        try:
                            offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                            logger.info("Setting remote description")
                            await pc.setRemoteDescription(offer)

                            # Add video track
                            async with self._camera_lock:
                                if not self.camera:
                                    self.camera = VideoCamera()
                                await self.camera.add_track()
                                pc.addTrack(self.camera)
                                logger.info(f"Video track added for client {client_id}")

                            # Add audio track
                            if not self.microphone:
                                self.microphone = AudioRecorder()
                            pc.addTrack(self.microphone)
                            logger.info(f"Audio track added for client {client_id}")

                            answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)

                            response = {
                                "sdp": pc.localDescription.sdp,
                                "type": pc.localDescription.type
                            }
                            await websocket.send(json.dumps(response))
                            self.pcs.add(pc)

                        except Exception as e:
                            logger.error(f"Error in offer handling: {str(e)}", exc_info=True)
                            if pc:
                                await self.cleanup_pc(pc, client_id)
                            raise

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client {client_id}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for client {client_id}")
        finally:
            pcs_to_cleanup = [pc for pc in self.pcs]
            for pc in pcs_to_cleanup:
                await self.cleanup_pc(pc, client_id)

    async def cleanup_pc(self, pc, client_id):
        if pc in self.pcs:
            self.pcs.discard(pc)
            await pc.close()
            if self.camera:
                await self.camera.remove_track()
            if self.microphone:
                await self.microphone.stop()
                self.microphone = None
            logger.info(f"Client {client_id} disconnected. Active connections: {len(self.pcs)}")

    async def cleanup_all(self):
        logger.info("Cleaning up all connections")
        coros = [self.cleanup_pc(pc, i) for i, pc in enumerate(self.pcs.copy(), 1)]
        await asyncio.gather(*coros)
        if self.camera:
            self.camera.stop()
        if self.microphone:
            await self.microphone.stop()



async def main():
    server = WebRTCServer()
    async with websockets.serve(
            server.websocket_handler,
            "0.0.0.0",
            8889,
            compression=None
    ):
        logger.info("WebRTC server running on ws://localhost:8889")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")