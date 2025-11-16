"""
Cog predictor for FreeMorph - a character animation model that morphs letters and shapes.
"""

import os
import tempfile
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize model here
        # For now, this is a placeholder for the actual FreeMorph model
        # The actual model loading would happen here
        print("FreeMorph predictor initialized")

    def predict(
        self,
        source_image: Path = Input(
            description="Source image containing the character/letter to animate"
        ),
        target_image: Path = Input(
            description="Target image to morph into",
            default=None
        ),
        num_frames: int = Input(
            description="Number of frames in the animation",
            default=30,
            ge=10,
            le=120
        ),
        fps: int = Input(
            description="Frames per second for output video",
            default=15,
            ge=5,
            le=60
        ),
    ) -> Path:
        """
        Run a single prediction on the model
        
        Args:
            source_image: The source image/letter to animate
            target_image: Optional target image to morph into
            num_frames: Number of frames for the animation
            fps: Frames per second
            
        Returns:
            Path to output video file
        """
        print(f"Processing with {num_frames} frames at {fps} fps")
        
        # Load source image
        source = self._load_image(source_image)
        print(f"Loaded source image with shape: {source.shape}")
        
        # Load target image if provided
        if target_image is not None:
            target = self._load_image(target_image)
            print(f"Loaded target image with shape: {target.shape}")
        else:
            # Create a default target (e.g., rotated/scaled version of source)
            target = self._create_default_target(source)
        
        # Generate animation frames
        frames = self._generate_animation(source, target, num_frames)
        
        # Save as video
        output_path = self._save_video(frames, fps)
        
        return Path(output_path)
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess an image"""
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        return img
    
    def _create_default_target(self, source: np.ndarray) -> np.ndarray:
        """Create a default target image from source"""
        # Simple transformation: rotate 180 degrees
        target = cv2.rotate(source, cv2.ROTATE_180)
        return target
    
    def _generate_animation(
        self, 
        source: np.ndarray, 
        target: np.ndarray, 
        num_frames: int
    ) -> List[np.ndarray]:
        """
        Generate animation frames between source and target
        
        This is a placeholder implementation. The actual FreeMorph model
        would generate sophisticated morphing animations.
        """
        frames = []
        h, w = source.shape[:2]
        
        # Ensure both images have the same dimensions
        if target.shape[:2] != (h, w):
            target = cv2.resize(target, (w, h))
        
        # Simple linear interpolation between source and target
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            frame = cv2.addWeighted(source, 1 - alpha, target, alpha, 0)
            frames.append(frame)
        
        return frames
    
    def _save_video(self, frames: List[np.ndarray], fps: int) -> str:
        """Save frames as a video file"""
        # Create temporary output file
        output_fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(output_fd)
        
        # Get frame dimensions
        h, w = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved to {output_path}")
        
        return output_path
