import gradio as gr
import numpy as np
import threading
import time
from queue import Queue, Empty
import cv2
from thread_safe_queue import ThreadSafeQueue
import tempfile
import os

class UniversalVideoRenderer:
    PLAY_THROUGH_MODE = 0
    RESTART_ON_GET_MODE = 1
    
    def __init__(self, vid_queue, mode, MAX_RATINGS, num_ratings, zoom=4, playback_speed=1):
        self.mode = mode
        self.vid_queue = vid_queue  # Consumes from render_q
        self.zoom_factor = zoom
        self.playback_speed = playback_speed
        self._stop_event = threading.Event()
        self.user_rating = Queue()
        self.latest_video = None
        self.lock = threading.Lock()
        self.temp_dir = tempfile.mkdtemp()  # Create temporary directory for video files
        self.rating_count = 0  # Add rating counter
        self.MAX_RATINGS = MAX_RATINGS  # Add maximum ratings constant
        self.num_ratings = num_ratings
        
        # Initialize Gradio interface components
        with gr.Blocks() as self.interface:
            with gr.Row():
                self.video_output = gr.Video(label="Video Feed", autoplay=True, loop=True)
                with gr.Column():
                    self.rating_input = gr.Radio(
                        choices=list(map(str, range(self.num_ratings))),
                        label="Rate the segment (0-"+str(self.num_ratings)+")"
                    )
                    self.submit_btn = gr.Button("Submit Rating")
                    self.stop_btn = gr.Button("Stop")  # Added Stop button
            
            # Handle rating submission
            self.submit_btn.click(
                fn=self._handle_rating,
                inputs=[self.rating_input],
                outputs=[]
            )
            
            # Handle stop button
            self.stop_btn.click(
                fn=self.stop_interface,  # New function to handle stopping
                inputs=[],
                outputs=[]
            )

            # Add periodic video update
            self.interface.load(
                fn=self.update_video,
                inputs=None,
                outputs=self.video_output,
                every=1  # Check for new video every second
            )
        
        # Start render thread
        self._render_thread = threading.Thread(target=self._render_loop)
        self._render_thread.daemon = True
        self._render_thread.start()
        
        # Launch Gradio interface in a separate thread
        self.interface_thread = threading.Thread(target=self._launch_interface)
        self.interface_thread.daemon = True
        self.interface_thread.start()
        
        print("Gradio renderer started")
    
    def update_video(self):
        """Function to be called periodically by Gradio to update the video."""
        with self.lock:
            if self.latest_video is not None and os.path.exists(self.latest_video):
                return self.latest_video
        return None
    
    def _launch_interface(self):
        """Launch the Gradio interface."""
        self.interface.launch(share=False, prevent_thread_lock=True)
    
    def _handle_rating(self, rating):
        """Handle user rating submission."""
        if rating is not None:
            try:
                rating_value = int(rating)
                self.user_rating.put(rating_value)
                self.rating_count += 1  # Increment rating counter
                if self.rating_count % 20 == 0:
                    print(f"Received user rating: {rating_value} ({self.rating_count}/{self.MAX_RATINGS})")
                if self.rating_count >= self.MAX_RATINGS:
                    print("Maximum ratings reached. Closing interface...")
                    self.stop()
            except ValueError:
                print(f"Invalid rating value received: {rating}")
        return None

    def stop_interface(self):
        """Function to handle the stop button click from Gradio."""
        print("Stop button clicked. Stopping the interface...")
        self.stop()

    def get_rating(self, timeout=None):
        """Retrieve a rating from the user."""
        try:
            return self.user_rating.get(timeout=timeout)
        except Empty:
            return None
    
    def stop(self):
        """Stop the renderer and clean up resources."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            
            # Clean up temporary directory
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except Exception as e:
                        print(f"Error removing file {file}: {e}")
                try:
                    os.rmdir(self.temp_dir)
                except Exception as e:
                    print(f"Error removing temp directory: {e}")
            
            # Close the Gradio interface
            if hasattr(self, 'interface'):
                try:
                    self.interface.close()
                except Exception as e:
                    print(f"Error closing Gradio interface: {e}")
            
            # Get the current thread
            current_thread = threading.current_thread()
            
            # Wait for interface_thread to finish if it's not the current thread
            if self.interface_thread.is_alive() and self.interface_thread != current_thread:
                try:
                    self.interface_thread.join(timeout=2)
                except RuntimeError as e:
                    print(f"Error joining interface_thread: {e}")
            
            # Wait for render_thread to finish if it's not the current thread
            if self._render_thread.is_alive() and self._render_thread != current_thread:
                try:
                    self._render_thread.join(timeout=2)
                except RuntimeError as e:
                    print(f"Error joining render_thread: {e}")
            
            print("Interface closed!")
    
    def _frames_to_video(self, frames):
        """Convert a sequence of frames to a video file using H.264 codec"""
        if not frames or len(frames) == 0:
            return None

        # Create a temporary file path
        temp_path = os.path.join(self.temp_dir, f'temp_video_{int(time.time())}.mp4')
        
        # Get frame properties
        height, width = frames[0].shape[:2]
        
        # Initialize video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try 'H264' if 'avc1' doesn't work
        fps = 30.0  # Fixed FPS for 2-second video
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        try:
            # Calculate frame spacing to create a 2-second video
            total_frames_needed = int(2.0 * fps)  # 2 seconds * 30 fps = 60 frames
            
            if len(frames) > total_frames_needed:
                # If we have too many frames, sample evenly
                frame_indices = np.linspace(0, len(frames)-1, total_frames_needed, dtype=int)
                frames = [frames[i] for i in frame_indices]
            elif len(frames) < total_frames_needed:
                # If we have too few frames, duplicate frames as needed
                frames = frames * (total_frames_needed // len(frames)) + frames[:total_frames_needed % len(frames)]
            
            # Write frames to video
            for frame in frames:
                # Convert frame to BGR if it's in RGB
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            
            out.release()
            return temp_path
            
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
        
    def _process_frame(self, frame):
        """Validate and preprocess a single frame."""
        if not isinstance(frame, np.ndarray):
            print(f"[Frame Validation] Invalid frame type: {type(frame)}")
            return None
            
        # Ensure frame is in the correct format
        if frame.size == 0 or len(frame.shape) < 2:
            print("[Frame Validation] Empty or invalid frame received")
            return None
            
        # Apply zoom if needed
        if self.zoom_factor != 1:
            height, width = frame.shape[:2]
            new_height = int(height * self.zoom_factor)
            new_width = int(width * self.zoom_factor)
            if new_height > 0 and new_width > 0:
                try:
                    frame = cv2.resize(
                        frame,
                        (new_width, new_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                except Exception as e:
                    print(f"[Resize Error] {e}")
                    return None
            else:
                print(f"[Frame Validation] Invalid dimensions after zoom: {new_width}x{new_height}")
                return None
        
        # Ensure dtype is uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Ensure data range is 0-255
        if frame.max() > 255 or frame.min() < 0:
            frame = np.clip(frame, 0, 255)
        
        return frame
    
    def _render_loop(self):
        """Continuously process frames from the queue and update the video feed."""
        try:
            while not self._stop_event.is_set():
                frames = []
                # Collect frames until we get None (end of segment)
                while not self._stop_event.is_set():
                    frame = self.vid_queue.safe_get()
                    if frame is None:
                        break
                    
                    if isinstance(frame, np.ndarray):
                        if len(frame.shape) == 4:  # Batch of frames
                            # Flatten batch into individual frames
                            for f in frame:
                                processed_frame = self._process_frame(f)
                                if processed_frame is not None:
                                    frames.append(processed_frame)
                        else:  # Single frame
                            processed_frame = self._process_frame(frame)
                            if processed_frame is not None:
                                frames.append(processed_frame)
                
                if frames:
                    # Convert frames to a 2-second video
                    video_path = self._frames_to_video(frames)
                    if video_path:
                        with self.lock:
                            # Clean up previous video if it exists
                            if self.latest_video and os.path.exists(self.latest_video):
                                try:
                                    os.remove(self.latest_video)
                                except Exception as e:
                                    print(f"Error removing old video: {str(e)}")
                            self.latest_video = video_path
                
                time.sleep(0.1)  # Small delay before next batch
                    
        except Exception as e:
            print(f"[Render Loop Error] {str(e)}")
            import traceback
            traceback.print_exc()