"""
Video Frame Extraction for GPU Benchmark
=========================================
Downloads public test videos (Big Buck Bunny) at different resolutions,
extracts frames using ffmpeg, and prepares data for GPU processing.

Usage:
    python video_frame_extract.py --download    # Download videos
    python video_frame_extract.py --extract     # Extract frames
    python video_frame_extract.py --all         # Both
"""

import subprocess
import os
import numpy as np
from pathlib import Path
import argparse

# Test videos - Big Buck Bunny at different resolutions
VIDEOS = {
    '180p': {
        'url': 'https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4',
        'filename': 'bbb_180p.mp4',
        'resolution': (320, 180)
    },
    '480p': {
        'url': 'http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_480p_30fps_normal.mp4',
        'filename': 'bbb_480p.mp4',
        'resolution': (854, 480)
    },
    '720p': {
        'url': 'http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_720p_30fps_normal.mp4',
        'filename': 'bbb_720p.mp4',
        'resolution': (1280, 720)
    },
    '1080p': {
        'url': 'http://distribution.bbb3d.renderfarming.net/video/mp4/bbb_sunflower_1080p_30fps_normal.mp4',
        'filename': 'bbb_1080p.mp4',
        'resolution': (1920, 1080)
    }
}

DATA_DIR = Path('./video_data')
FRAMES_DIR = Path('./frames')


def download_videos(resolutions=None):
    """Download test videos."""
    DATA_DIR.mkdir(exist_ok=True)
    
    if resolutions is None:
        resolutions = VIDEOS.keys()
    
    for res in resolutions:
        if res not in VIDEOS:
            print(f"Unknown resolution: {res}")
            continue
            
        video = VIDEOS[res]
        output_path = DATA_DIR / video['filename']
        
        if output_path.exists():
            print(f"{res}: already exists, skipping")
            continue
        
        print(f"{res}: downloading...")
        try:
            subprocess.run([
                'wget', '-q', '--show-progress',
                '-O', str(output_path),
                video['url']
            ], check=True)
            print(f"{res}: done")
        except subprocess.CalledProcessError as e:
            print(f"{res}: download failed - {e}")
        except FileNotFoundError:
            # wget not found, try curl
            try:
                subprocess.run([
                    'curl', '-L', '-o', str(output_path),
                    video['url']
                ], check=True)
                print(f"{res}: done")
            except Exception as e:
                print(f"{res}: download failed - {e}")


def extract_frames(video_path, output_dir, max_frames=100, fps=5):
    """
    Extract frames from video using ffmpeg.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
        fps: Frames per second to extract (lower = fewer frames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg to extract frames
    # -vf fps=N extracts N frames per second
    # -vframes limits total frames
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-vframes', str(max_frames),
        '-q:v', '2',  # Quality (2 = high)
        str(output_dir / 'frame_%04d.jpg')
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        n_frames = len(list(output_dir.glob('frame_*.jpg')))
        return n_frames
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        return 0


def extract_all_videos(max_frames_per_video=100):
    """Extract frames from all downloaded videos."""
    FRAMES_DIR.mkdir(exist_ok=True)
    
    results = {}
    
    for res, video in VIDEOS.items():
        video_path = DATA_DIR / video['filename']
        
        if not video_path.exists():
            print(f"{res}: video not found, skipping")
            continue
        
        output_dir = FRAMES_DIR / res
        print(f"{res}: extracting frames...")
        
        n_frames = extract_frames(video_path, output_dir, max_frames=max_frames_per_video)
        results[res] = {
            'n_frames': n_frames,
            'resolution': video['resolution'],
            'path': output_dir
        }
        print(f"{res}: extracted {n_frames} frames at {video['resolution']}")
    
    return results


def load_frames_as_numpy(frames_dir):
    """
    Load extracted frames as numpy arrays.
    Returns list of (height, width) varying arrays - simulates variable-size batch.
    """
    from PIL import Image
    
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob('frame_*.jpg'))
    
    frames = []
    for f in frame_files:
        img = Image.open(f).convert('L')  # Grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(arr)
    
    return frames


def create_variable_size_batch(resolutions=None, frames_per_res=500):
    """
    Create a mixed batch of different resolution frames.
    This simulates real-world variable-size workload.
    
    Returns:
        frames: List of numpy arrays (different sizes)
        metadata: List of (resolution_name, width, height) tuples
    """
    if resolutions is None:
        resolutions = ['180p', '480p', '720p', '1080p']
    
    frames = []
    metadata = []
    
    for res in resolutions:
        res_dir = FRAMES_DIR / res
        if not res_dir.exists():
            print(f"{res}: frames not found, skipping")
            continue
        
        res_frames = load_frames_as_numpy(res_dir)[:frames_per_res]
        
        for frame in res_frames:
            frames.append(frame)
            metadata.append({
                'resolution': res,
                'width': frame.shape[1],
                'height': frame.shape[0],
                'pixels': frame.shape[0] * frame.shape[1]
            })
    
    return frames, metadata


def print_batch_stats(frames, metadata):
    """Print statistics about the variable-size batch."""
    print("\n" + "=" * 60)
    print("BATCH STATISTICS")
    print("=" * 60)
    
    total_pixels = sum(m['pixels'] for m in metadata)
    sizes = [m['pixels'] for m in metadata]
    
    print(f"\nTotal frames: {len(frames)}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Min size: {min(sizes):,} pixels")
    print(f"Max size: {max(sizes):,} pixels")
    print(f"Imbalance ratio: {max(sizes) / min(sizes):.2f}x")
    
    print("\nBy resolution:")
    for res in ['180p', '480p', '720p', '1080p']:
        count = sum(1 for m in metadata if m['resolution'] == res)
        if count > 0:
            print(f"  {res}: {count} frames")
    
    return {
        'total_frames': len(frames),
        'total_pixels': total_pixels,
        'min_size': min(sizes),
        'max_size': max(sizes),
        'imbalance': max(sizes) / min(sizes)
    }


def prepare_for_gpu(frames):
    """
    Flatten frames into format ready for GPU benchmark.
    Same format as your triple_baseline_benchmark.py expects.
    
    Returns:
        images_flat: 1D array of all pixels
        offsets: Start index of each image
        sizes: Number of pixels in each image
        widths: Width of each image
        heights: Height of each image
    """
    all_pixels = []
    offsets = []
    sizes = []
    widths = []
    heights = []
    
    current_offset = 0
    
    for frame in frames:
        h, w = frame.shape
        pixels = frame.flatten()
        
        all_pixels.append(pixels)
        offsets.append(current_offset)
        sizes.append(h * w)
        widths.append(w)
        heights.append(h)
        
        current_offset += h * w
    
    images_flat = np.concatenate(all_pixels).astype(np.float32)
    offsets = np.array(offsets, dtype=np.int64)
    sizes = np.array(sizes, dtype=np.int64)
    widths = np.array(widths, dtype=np.int32)
    heights = np.array(heights, dtype=np.int32)
    
    return images_flat, offsets, sizes, widths, heights


def main():
    parser = argparse.ArgumentParser(description='Video frame extraction for GPU benchmark')
    parser.add_argument('--download', action='store_true', help='Download test videos')
    parser.add_argument('--extract', action='store_true', help='Extract frames from videos')
    parser.add_argument('--all', action='store_true', help='Download and extract')
    parser.add_argument('--stats', action='store_true', help='Show batch statistics')
    parser.add_argument('--prepare', action='store_true', help='Prepare data for GPU benchmark')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames per video')
    args = parser.parse_args()
    
    if args.all or args.download:
        print("Downloading videos...")
        download_videos()
    
    if args.all or args.extract:
        print("\nExtracting frames...")
        extract_all_videos(max_frames_per_video=args.max_frames)
    
    if args.stats:
        print("\nLoading frames...")
        frames, metadata = create_variable_size_batch()
        print_batch_stats(frames, metadata)
    
    if args.prepare:
        print("\nPreparing data for GPU...")
        frames, metadata = create_variable_size_batch()
        images_flat, offsets, sizes, widths, heights = prepare_for_gpu(frames)
        
        # Save for benchmark
        output_file = 'video_frames_data.npz'
        np.savez(output_file,
                 images_flat=images_flat,
                 offsets=offsets,
                 sizes=sizes,
                 widths=widths,
                 heights=heights)
        print(f"Saved to {output_file}")
        print(f"  Total pixels: {len(images_flat):,}")
        print(f"  Total frames: {len(sizes)}")
    
    if not any([args.download, args.extract, args.all, args.stats, args.prepare]):
        print("Usage:")
        print("  python video_frame_extract.py --all        # Download + extract")
        print("  python video_frame_extract.py --stats      # Show statistics")
        print("  python video_frame_extract.py --prepare    # Prepare for GPU benchmark")


if __name__ == "__main__":
    main()