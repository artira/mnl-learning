#!/usr/bin/env python3
"""
Vampire Survivors — Reward Dynamics Pipeline Runner
=====================================================
Media Neuroscience Lab, UCSB

Single entry point to run event detection on one or more
gameplay videos.

Usage:
    python run_pipeline.py video1.mp4
    python run_pipeline.py video1.mp4 video2.mp4
    python run_pipeline.py *.mp4
    python run_pipeline.py /path/to/videos/
    python run_pipeline.py video1.mp4 --sample-rate 2
    python run_pipeline.py video1.mp4 --output-dir /path/to/output

Output structure:
    pipeline_results/
      video_name/
        events/
          all_features_ocr.csv
          detected_levelups.csv
          detected_deaths.csv
          detected_results.csv
          levelup_detection_ocr.png
          event_summary.json
      batch_summary.json
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import timedelta
from glob import glob
import json


def find_videos(paths):
    """
    Resolve video paths from arguments.
    Handles individual files, glob patterns, and directories.
    """
    videos = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            videos.append(path)
        elif path.is_dir():
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                videos.extend(sorted(path.glob(ext)))
        else:
            expanded = sorted(glob(str(p)))
            for f in expanded:
                fp = Path(f)
                if fp.is_file() and fp.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    videos.append(fp)

    seen = set()
    unique = []
    for v in videos:
        resolved = v.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(v)

    return unique


def run_events(video_path, output_dir, sample_rate=4):
    """Run the OCR event detection pipeline."""
    from levelup_detector_arti_ocr_v5 import extract_features_ocr, detect_levelups
    from levelup_detector_arti_ocr_v5 import detect_events_by_type, analyze_schedule
    from levelup_detector_arti_ocr_v5 import plot_results
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [EVENTS] Running OCR event detection...")
    t0 = time.time()

    features = extract_features_ocr(str(video_path), sample_rate=sample_rate)
    features.to_csv(output_dir / 'all_features_ocr.csv', index=False)

    levelup_events = detect_levelups(features)
    levelup_events.to_csv(output_dir / 'detected_levelups.csv', index=False)

    death_events = detect_events_by_type(
        features, 'is_death', 'Death',
        config={'min_duration': 0.5, 'max_duration': 60.0, 'merge_gap': 3.0, 'min_ocr_confidence': 0.2}
    )
    death_events.to_csv(output_dir / 'detected_deaths.csv', index=False)

    results_events = detect_events_by_type(
        features, 'is_results', 'Results',
        config={'min_duration': 0.5, 'max_duration': 60.0, 'merge_gap': 3.0, 'min_ocr_confidence': 0.2}
    )
    results_events.to_csv(output_dir / 'detected_results.csv', index=False)

    schedule = analyze_schedule(levelup_events)
    plot_results(features, levelup_events, schedule,
                output_dir / 'levelup_detection_ocr.png')

    duration = features['timestamp'].max()
    elapsed = time.time() - t0

    summary = {
        'video': str(video_path),
        'duration_seconds': round(duration, 1),
        'sample_rate_hz': sample_rate,
        'event_counts': {
            'levelups': len(levelup_events),
            'deaths': len(death_events),
            'results_screens': len(results_events),
        },
        'schedule_stats': schedule,
        'processing_time': round(elapsed, 1),
    }

    with open(output_dir / 'event_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [EVENTS] Done: {len(levelup_events)} level-ups, "
          f"{len(death_events)} deaths, {len(results_events)} results "
          f"({timedelta(seconds=int(elapsed))})")

    return summary


def process_video(video_path, base_output, args):
    """Process a single video through the event detection pipeline."""
    video_path = Path(video_path)
    video_name = video_path.stem

    print(f"\n{'='*60}")
    print(f"  Processing: {video_path.name}")
    print(f"  Output: {base_output / video_name}/")
    print(f"{'='*60}")

    results = {'video': video_path.name, 'status': 'success'}
    total_start = time.time()

    try:
        event_summary = run_events(
            video_path,
            base_output / video_name / 'events',
            sample_rate=args.sample_rate,
        )
        results['events'] = event_summary.get('event_counts', {})
    except Exception as e:
        print(f"  [EVENTS] ERROR: {e}")
        results['events_error'] = str(e)
        results['status'] = 'error'

    results['total_time'] = round(time.time() - total_start, 1)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Vampire Survivors Reward Dynamics Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py video1.mp4
  python run_pipeline.py video1.mp4 video2.mp4
  python run_pipeline.py *.mp4
  python run_pipeline.py /path/to/videos/
  python run_pipeline.py video1.mp4 --sample-rate 2
        """
    )

    parser.add_argument(
        'videos', nargs='+',
        help='Video file(s), directory, or glob pattern'
    )
    parser.add_argument(
        '--output-dir', '-o', default='./pipeline_results',
        help='Base output directory (default: ./pipeline_results)'
    )
    parser.add_argument(
        '--sample-rate', '-r', type=int, default=4,
        help='Frames per second to analyze (default: 4)'
    )

    args = parser.parse_args()

    videos = find_videos(args.videos)

    if not videos:
        print("No video files found. Check your paths.")
        print("Supported formats: .mp4, .avi, .mov, .mkv")
        sys.exit(1)

    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VAMPIRE SURVIVORS — REWARD DYNAMICS PIPELINE")
    print("Media Neuroscience Lab, UCSB")
    print("=" * 60)
    print(f"  Videos to process: {len(videos)}")
    print(f"  Output directory: {base_output}")
    print(f"  Sample rate: {args.sample_rate} fps")
    print(f"  Pipeline: Event detection (OCR)")

    for v in videos:
        print(f"    {v.name}")

    batch_start = time.time()
    all_results = []

    for i, video in enumerate(videos):
        print(f"\n[Video {i+1}/{len(videos)}]")
        result = process_video(video, base_output, args)
        all_results.append(result)

    batch_time = time.time() - batch_start
    batch_summary = {
        'total_videos': len(videos),
        'successful': sum(1 for r in all_results if r['status'] == 'success'),
        'total_time_seconds': round(batch_time, 1),
        'total_time_display': str(timedelta(seconds=int(batch_time))),
        'results': all_results,
    }

    with open(base_output / 'batch_summary.json', 'w') as f:
        json.dump(batch_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"  Total time: {timedelta(seconds=int(batch_time))}")
    print(f"  Videos processed: {batch_summary['successful']}/{len(videos)}")

    for r in all_results:
        status = "OK" if r['status'] == 'success' else "FAIL"
        events_str = ""
        if 'events' in r:
            events_str = f" LU={r['events'].get('levelups', '?')}"
        print(f"  [{status}] {r['video']}{events_str} ({r.get('total_time', 0):.0f}s)")

    print(f"\n  Results: {base_output}/")
    print(f"  Summary: {base_output / 'batch_summary.json'}")


if __name__ == '__main__':
    main()
