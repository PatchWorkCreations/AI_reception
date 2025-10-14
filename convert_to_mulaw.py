#!/usr/bin/env python3
"""
Audio converter script to convert MP3 to 8kHz μ-law format
Commonly used for telephony applications like Twilio
"""

import os
import sys
from pydub import AudioSegment

def convert_to_mulaw(input_file, output_file=None):
    """
    Convert audio file to 8kHz μ-law format
    
    Args:
        input_file: Path to input audio file (MP3, WAV, etc.)
        output_file: Path to output file (optional, will auto-generate if not provided)
    
    Returns:
        Path to the converted file
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return None
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_8k_mulaw.wav"
    
    try:
        print(f"Loading audio file: {input_file}")
        # Load the audio file
        audio = AudioSegment.from_file(input_file)
        
        print("Converting to 8kHz μ-law format...")
        # Convert to mono (single channel)
        audio = audio.set_channels(1)
        
        # Convert to 8kHz sample rate
        audio = audio.set_frame_rate(8000)
        
        # Export as WAV with μ-law (PCMU) codec
        audio.export(
            output_file,
            format="wav",
            codec="pcm_mulaw",  # μ-law encoding
            parameters=["-ar", "8000"]  # Ensure 8kHz sample rate
        )
        
        print(f"✓ Successfully converted to: {output_file}")
        print(f"  - Sample rate: 8kHz")
        print(f"  - Codec: μ-law (PCMU)")
        print(f"  - Channels: Mono")
        
        # Show file sizes
        input_size = os.path.getsize(input_file) / 1024  # KB
        output_size = os.path.getsize(output_file) / 1024  # KB
        print(f"\nFile sizes:")
        print(f"  - Input:  {input_size:.2f} KB")
        print(f"  - Output: {output_size:.2f} KB")
        
        return output_file
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nMake sure you have ffmpeg installed:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - Mac: brew install ffmpeg")
        print("  - Linux: sudo apt-get install ffmpeg")
        return None


if __name__ == "__main__":
    # Default file path
    default_input = r"E:\New Downloads\medai_receptionist\myProject\myApp\static\neuromed_intro.mp3"
    
    # Check if user provided a file path as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input
    
    # Optional: custom output file path
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("Audio Converter: MP3 to 8kHz μ-law")
    print("=" * 60)
    
    result = convert_to_mulaw(input_file, output_file)
    
    if result:
        print("\n✓ Conversion completed successfully!")
    else:
        print("\n✗ Conversion failed!")
        sys.exit(1)

