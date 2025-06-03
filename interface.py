#!/usr/bin/env python

"""
A CLI-based interface for querying the user about segment preferences,
with frames converted to MP4 before being sent to the Gradio interface.
"""

import logging
import time
from threading import Event, Thread
from thread_safe_queue import ThreadSafeQueue
import numpy as np
import cv2
import gradio as gr
import threading
from queue import Queue, Empty

# Import the updated UniversalVideoRenderer
from display_interface import UniversalVideoRenderer


class RateInterface:

    def __init__(self, feed_type, num_ratings, reward_model, max_num_feedback, max_segs=200):
        
        # Initialize thread-safe queues
        self.vid_q = ThreadSafeQueue(maxsize=max_segs)       # Queue for processing
        self.render_q = ThreadSafeQueue(maxsize=max_segs)    # Queue for rendering (frames)
        self.obs_q = ThreadSafeQueue(maxsize=max_segs)
        self.ratings_q = ThreadSafeQueue(maxsize=max_segs)
        self.reward_q = ThreadSafeQueue(maxsize=max_segs)

        # Create the Gradio-based renderer
        self.renderer = UniversalVideoRenderer(
            vid_queue=self.render_q,  # Pass render_q to the renderer
            mode=UniversalVideoRenderer.RESTART_ON_GET_MODE,
            MAX_RATINGS=max_num_feedback,
            num_ratings=num_ratings,
            zoom=4
        )
        
        self.feed_type = feed_type
        self.reward_model = reward_model
        self.max_segs = max_segs
        self.max_num_feedback = max_num_feedback
        
        # Thread control
        self._stop_event = Event()
        self._processing_thread = Thread(target=self._process_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()

        # Initialize other attributes
        self.feedback_count = 0
        self.ratings_obs_from_user = []
        self.ratings_from_user = []
        self.reward_from_user = []

        self.num_ratings = num_ratings

    def stop(self):
        """Stop the processing loop and renderer."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            if self.renderer:
                self.renderer.stop()
            
            # Get the current thread
            current_thread = threading.current_thread()
            
            # Wait for processing_thread to finish if it's not the current thread
            if self._processing_thread.is_alive() and self._processing_thread != current_thread:
                try:
                    self._processing_thread.join(timeout=2)
                except RuntimeError as e:
                    print(f"Error joining processing_thread: {e}")

    def _process_loop(self):
        """Main processing loop to handle video segments and user ratings."""
        while not self._stop_event.is_set():
            self.check_for_data()
            
            video_obs = self.vid_q.safe_get()
            if video_obs is not None:
                rl_obs = self.obs_q.safe_get()
                reward = self.reward_q.safe_get()
                
                if all(x is not None for x in [rl_obs, reward]):
                    rating = self.ask_user(video_obs)

                    self.ratings_obs_from_user.append(rl_obs)
                    self.ratings_from_user.append(rating)
                    self.reward_from_user.append(np.sum(np.array(reward)))

                    if len(self.ratings_from_user) >= self.reward_model.needed_feedback:
                        self._process_feedback()

                    if self.feedback_count >= self.max_num_feedback:
                        break

            time.sleep(0.1)  # Prevent busy waiting

    def _process_feedback(self):
        """Process and send collected feedback to the reward model."""
        self.reward_model.put_queries(
            np.array(self.ratings_obs_from_user),
            np.array(self.ratings_from_user).reshape(len(self.ratings_from_user), 1),
            np.array(self.reward_from_user).reshape(len(self.reward_from_user), 1)
        )
        print(f"Processed {len(self.ratings_from_user)} feedback entries.")
        self.ratings_obs_from_user = []
        self.ratings_from_user = []
        self.reward_from_user = []

    def get_feed_type(self, sa_t, render_sa_t, reward):
        """Determine which segments to feed based on the feed_type."""
        # Uniform
        if self.feed_type == 0:
            return sa_t, render_sa_t, reward
        
        # Disagreement
        elif self.feed_type == 1:
            _, disagree = self.reward_model.get_mean_and_std(sa_t)
            top_k_index = (-disagree).argsort()[:self.reward_model.mb_size]
            reward, sa_t, render_sa_t = reward[top_k_index], sa_t[top_k_index], render_sa_t[top_k_index]    
            return sa_t, render_sa_t, reward
        # Error, Default to Uniform
        else:
            print('Feed Type not defined, defaulting to Uniform.')
            return sa_t, render_sa_t, reward

    def add_data(self):
        """Add new data segments to the processing queues."""
        # Retrieve video segments from the reward model
        sa_t, render_sa_t, reward = self.reward_model.get_queries()
        
        # Select segments based on feed_type
        sa_t, render_sa_t, reward = self.get_feed_type(sa_t, render_sa_t, reward)

        # Clear existing queues to avoid backlog
        if not self.vid_q.empty():
            self.vid_q.queue.clear()
            self.obs_q.queue.clear()
            self.reward_q.queue.clear()
            self.render_q.queue.clear()  # Also clear render queue

        for i in range(len(sa_t)):
            if self.vid_q.qsize() < self.max_segs:
                # Ensure render_sa_t[i] is in the correct format
                video_segment = render_sa_t[i]
                if isinstance(video_segment, np.ndarray):
                    if len(video_segment.shape) == 4:  # Batch of frames
                        self.vid_q.safe_put(video_segment)
                    else:  # Single frame
                        self.vid_q.safe_put(video_segment)
                
                self.obs_q.safe_put(sa_t[i])
                self.reward_q.safe_put(reward[i])

    def check_for_data(self):
        """Check if new data needs to be added based on the reward model's flag."""
        if getattr(self.reward_model, 'add_data_flag', False):
            # Add the data 
            self.add_data()
            self.reward_model.add_data_flag = False
            print("New data added to queues.")

    def ask_user(self, frames):
        """
        Send frames to the renderer and wait for the user's rating.

        Args:
            frames (np.ndarray): Array of frames (H, W, C) or (N, H, W, C).

        Returns:
            int: User-provided rating.
        """
        # Send frames to the renderer
        self.render_q.safe_put(frames)

        # Signal end of segment
        self.render_q.safe_put(None)

        # Wait for rating from Gradio interface
        rating = self.renderer.get_rating(timeout=None)
        if rating is not None:
            self.feedback_count += 1
            self.reward_model.feedback_count = self.feedback_count
            if self.feedback_count >= self.max_num_feedback:
                print("Maximum ratings reached. Stopping interface...")
                self.stop()
        else:
            print("No rating received within timeout.")
        return rating

# Example usage
if __name__ == "__main__":
    # Mock implementations for reward_model and other dependencies
    class MockRewardModel:
        def __init__(self, needed_feedback=100, mb_size=10):
            self.needed_feedback = needed_feedback
            self.mb_size = mb_size
            self.feedback_count = 0
            self.add_data_flag = False
        
        def get_queries(self):
            # Return mock data
            sa_t = np.random.rand(20, 10)  # Example state-action tensors
            render_sa_t = np.random.randint(0, 256, (20, 60, 480, 640, 3), dtype=np.uint8)  # Example frames
            reward = np.random.rand(20)
            return sa_t, render_sa_t, reward
        
        def get_mean_and_std(self, sa_t):
            mean = np.mean(sa_t, axis=0)
            std = np.std(sa_t, axis=0)
            return mean, std
        
        def put_queries(self, obs, ratings, rewards):
            # Handle the feedback
            print("Reward model received feedback.")
            pass

    # Instantiate the mock reward model
    reward_model = MockRewardModel()

    # Instantiate the RateInterface
    rate_interface = RateInterface(
        feed_type=0,              # 0 for Uniform, 1 for Disagreement
        num_ratings=5,            # Ratings from 0 to 4
        reward_model=reward_model,
        max_num_feedback=100,     # Maximum number of feedback entries
        max_segs=200               # Maximum segments
    )

    try:
        while not rate_interface._stop_event.is_set():
            # Simulate setting the add_data_flag to add new data
            reward_model.add_data_flag = True
            time.sleep(5)  # Wait for some time before adding new data
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping RateInterface...")
        rate_interface.stop()
