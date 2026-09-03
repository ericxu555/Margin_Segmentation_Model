#!/usr/bin/env python3
"""
Convert segmented frames to video using OpenCV.
"""

import argparse
import cv2
from pathlib import Path
import re


def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def frames_to_video(input_dir, output_video, fps=30, pattern="*_overlay.png"):
    """
    Convert frames to video.
    
    Args:
        input_dir: Directory containing frame images
        output_video: Output video file path
        fps: Frames per second
        pattern: Glob pattern to match frame files
    """
    input_path = Path(input_dir)
    
    # Get all frame files matching pattern
    frame_files = sorted(input_path.glob(pattern), key=natural_sort_key)
    
    if not frame_files:
        print(f"Error: No frames found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(frame_files)} frames")
    print(f"Creating video at {fps} fps...")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"Error: Could not read first frame: {frame_files[0]}")
        return
    
    height, width, _ = first_frame.shape
    print(f"Frame size: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Write frames to video
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Warning: Could not read frame: {frame_file}")
            continue
        
        out.write(frame)
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(frame_files):
            print(f"Processed {i + 1}/{len(frame_files)} frames")
    
    # Release video writer
    out.release()
    print(f"\nVideo saved to: {output_video}")
    print(f"Duration: {len(frame_files) / fps:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Convert segmented frames to video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing frame images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output video file path (e.g., output.mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_overlay.png',
        help='Glob pattern to match frame files'
    )
    
    args = parser.parse_args()
    
    frames_to_video(args.input, args.output, args.fps, args.pattern)


if __name__ == '__main__':
    main()
