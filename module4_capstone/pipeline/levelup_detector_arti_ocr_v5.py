"""
Vampire Survivors — Level-Up Detection via OCR
===============================================
Media Neuroscience Lab, UCSB
Author: Arti Ramanathan

Detects level-up screens by reading the actual text on screen.
Looks for "Level Up" text in the center-top region of each frame.

This approach is:
  - Accurate: directly detects the defining feature of a level-up
  - Robust: works regardless of background chaos
  - Distinguishing: separates "Level Up" from "Results" or other menus
"""

import cv2 as cv
import numpy as np
import pandas as pd
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import timedelta
import json
import time


# ============================================================
# SECTION 1: OCR-BASED FEATURE EXTRACTION
# ============================================================

def extract_features_ocr(video_path, sample_rate=4, verbose=True):
    """
    Extract text from gameplay frames using OCR.
    
    For each sampled frame, reads text from the center-top
    region where "Level Up!" appears. Also extracts basic
    visual features (center_variance, pixel_change) for
    comparison and plotting.
    
    Uses EasyOCR which is better than Tesseract for 
    stylized game fonts.
    """
    # Initialize OCR reader (downloads model on first run)
    # This takes 10-20 seconds the first time
    if verbose:
        print("  Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = max(1, int(fps / sample_rate))
    
    if verbose:
        print(f"  Video: {Path(video_path).name}")
        print(f"  FPS: {fps:.1f} | Duration: {timedelta(seconds=int(duration))}")
        print(f"  Sampling at {sample_rate} Hz (every {frame_interval} frames)")
    
    features = []
    prev_gray = None
    frame_count = 0
    analyzed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            h, w = frame.shape[:2]
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # ------------------------------------------------
            # OCR: Read text from the region where
            # "Level Up!" appears (center-top of screen)
            #
            # From the screenshot: the text appears roughly
            # at 20-35% from top, 35-70% from left
            # We crop to this region to:
            #   1. Speed up OCR (smaller image = faster)
            #   2. Avoid reading other text (stats panel, 
            #      timer, kill counter)
            # ------------------------------------------------
            text_region = frame[
                int(h * 0.10):int(h * 0.55),  # top portion
                int(w * 0.25):int(w * 0.80)   # center horizontally
            ]
            
            # Run OCR on the cropped region
            # detail=0 returns just the text strings (faster)
            # detail=1 returns text + bounding boxes + confidence
            ocr_results = reader.readtext(text_region, detail=1)
            
            # Check if any detected text contains "level up"
            found_levelup = False
            found_results = False
            found_death = False
            ocr_confidence = 0.0
            detected_text = ""
            
            # Iterate through OCR results
            # for (bbox, text, conf) in ocr_results:
            #     text_lower = text.lower().strip()
                
            #     # Check for "level up" (may be read as one or two words)
            #     if 'level' in text_lower and 'up' in text_lower:
            #         found_levelup = True
            #         ocr_confidence = max(ocr_confidence, conf)
            #         detected_text = text
            #     elif 'levelup' in text_lower.replace(' ', ''):
            #         found_levelup = True
            #         ocr_confidence = max(ocr_confidence, conf)
            #         detected_text = text
            #     elif 'level' in text_lower:
            #         # Partial match — "Level" found but "Up" might
            #         # be on a separate line or missed
            #         # Check other results for "up"
            #         for (_, other_text, other_conf) in ocr_results:
            #             if 'up' in other_text.lower().strip():
            #                 found_levelup = True
            #                 ocr_confidence = max(
            #                     ocr_confidence, 
            #                     min(conf, other_conf)
            #                 )
            #                 detected_text = f"{text} {other_text}"
            for (bbox, text, conf) in ocr_results:
                text_lower = text.lower().strip()
                
                # Check for "Level Up"
                if 'level' in text_lower and 'up' in text_lower:
                    found_levelup = True
                    ocr_confidence = max(ocr_confidence, conf)
                    detected_text = text
                elif 'levelup' in text_lower.replace(' ', ''):
                    found_levelup = True
                    ocr_confidence = max(ocr_confidence, conf)
                    detected_text = text
                elif 'level' in text_lower:
                    # Partial match — "Level" found but "Up" might
                    # be on a separate line or missed
                    # Check other results for "up"
                    for (_, other_text, other_conf) in ocr_results:
                        if 'up' in other_text.lower().strip():
                            found_levelup = True
                            ocr_confidence = max(
                                ocr_confidence, 
                                min(conf, other_conf)
                            )
                            detected_text = f"{text} {other_text}"
                # Check for "Results" screen
                if 'result' in text_lower:
                    found_results = True
                # Check for "Death" / "Game Over" screen
                if 'game' in text_lower and 'over' in text_lower:
                    found_death = True
                elif 'gameover' in text_lower.replace(' ', ''):
                    found_death = True
                elif 'death' in text_lower:
                    found_death = True
                elif 'you died' in text_lower:
                    found_death = True
            # Too much text filtering and confidence-based heuristics ended up rejecting 
            # a lot of true positives,
            # so I went back to the simplified version to just look for "level" and "up" in the detected text
                            
            # for (bbox, text, conf) in ocr_results:
            #     text_lower = text.lower().strip()
                
            #     # REJECT stat/upgrade text that contains "level" and "up"
            #     # but isn't the "Level Up!" header
            #     if any(r in text_lower for r in [':', 'by', 'damage', 'area', 'base', 'speed', 'health', 'armor', 'recovery']):
            #         continue
                
            #     # REJECT text with numbers after "level" (e.g., "level:2", "level 4")
            #     remaining = text_lower.replace('level', '').replace('up', '').replace('!', '').replace(' ', '')
            #     if any(c.isdigit() for c in remaining):
            #         continue
                
            #     # REJECT text that is too long (header is ~10 chars, stat text is 15+)
            #     if len(text.strip()) > 15:
            #         continue
                
            #     # ACCEPT clean "Level Up" matches
            #     if 'level' in text_lower and 'up' in text_lower:
            #         found_levelup = True
            #         ocr_confidence = max(ocr_confidence, conf)
            #         detected_text = text
            #         break
            #     elif 'levelup' in text_lower.replace(' ', ''):
            #         found_levelup = True
            #         ocr_confidence = max(ocr_confidence, conf)
            #         detected_text = text
            #         break
            
            # # Check for Results screen separately
            # for (bbox, text, conf) in ocr_results:
            #     if 'result' in text.lower():
            #         found_results = True
            #         if not found_levelup:
            #             detected_text = text    
            #     # Check for "Results" screen (different menu type)
            #     if 'result' in text_lower:
            #         found_results = True
            #         detected_text = text
            
            # ------------------------------------------------
            # Also extract basic visual features for plotting
            # (much cheaper than optical flow)
            # ------------------------------------------------
            if prev_gray is not None:
                diff = cv.absdiff(gray, prev_gray)
                pixel_change = float(np.mean(diff))
            else:
                pixel_change = 0.0
            
            prev_gray = gray.copy()
            
            center_region = gray[int(h*0.2):int(h*0.8), int(w*0.15):int(w*0.85)]
            center_variance = float(np.var(center_region))
            
            features.append({
                'timestamp': round(timestamp, 3),
                'is_levelup': found_levelup,
                'is_results': found_results,
                'is_death': found_death,
                'ocr_confidence': round(ocr_confidence, 4),
                'detected_text': detected_text,
                'center_variance': round(center_variance, 2),
                'pixel_change': round(pixel_change, 4),
            })
            
            analyzed += 1
            if verbose and analyzed % 50 == 0:
                pct = (frame_count / total_frames) * 100
                print(f"    {timedelta(seconds=int(timestamp))} ({pct:.0f}%)")
            
            # Log when level-up is found
            if found_levelup and verbose:
                print(f"    ✓ Level Up detected at {timedelta(seconds=int(timestamp))} "
                      f"(conf: {ocr_confidence:.2f}, text: '{detected_text}')")
        
        frame_count += 1
    
    cap.release()
    
    if verbose:
        levelup_frames = sum(1 for f in features if f['is_levelup'])
        print(f"  Done: {analyzed} frames, {levelup_frames} frames with 'Level Up' text")
    
    return pd.DataFrame(features)


# ============================================================
# SECTION 2: LEVEL-UP EVENT DETECTION
# ============================================================

def detect_levelups(features_df, config=None, verbose=True):
    """
    Group consecutive Level Up frames into events.
    
    Much simpler than the variance-based approach because
    OCR gives a binary signal: the text is there or it isn't.
    """
    if config is None:
        config = {
            'min_duration': 1.0,    # seconds
            'max_duration': 30.0,   # seconds
            'merge_gap': 1.5,       # merge events within this gap
            'max_ocr_confidence': 0.9,  # maximum OCR confidence (to filter out weird false positives)
            'min_ocr_confidence': 0.3,  # minimum OCR confidence
        }
    
    df = features_df.copy()
    
    # Filter by OCR confidence
    df['is_menu'] = (
        df['is_levelup'] & 
        (df['ocr_confidence'] >= config['min_ocr_confidence'])
    )
    
    # Find contiguous menu periods
    events = []
    in_menu = False
    menu_start = 0.0
    menu_frames = []
    
    # Iterate through frames to group into events
    for _, row in df.iterrows(): # Loop through each row in the DataFrame
        if row['is_menu'] and not in_menu:
            in_menu = True
            menu_start = row['timestamp']
            menu_frames = [row]
        elif row['is_menu'] and in_menu:
            menu_frames.append(row)
        elif not row['is_menu'] and in_menu:
            in_menu = False
            menu_end = row['timestamp']
            duration = menu_end - menu_start
            # Only keep events that are within the expected duration range
            if duration >= config['min_duration'] and duration <= config['max_duration']:
                confidences = [f['ocr_confidence'] for f in menu_frames]
                events.append({
                    'event_id': len(events) + 1,
                    'timestamp_start': round(menu_start, 2),
                    'timestamp_end': round(menu_end, 2),
                    'duration': round(duration, 2),
                    'mean_ocr_confidence': round(np.mean(confidences), 3),
                    'max_ocr_confidence': round(np.max(confidences), 3),
                    'n_frames': len(menu_frames),
                    'detected_text': menu_frames[0]['detected_text'],
                })
    
    # Handle video ending during menu
    if in_menu:
        menu_end = df['timestamp'].iloc[-1] # Get the last timestamp in the DataFrame
        duration = menu_end - menu_start # Calculate the duration of the menu event
        # Only keep events that are within the expected duration range
        if duration >= config['min_duration']:
            confidences = [f['ocr_confidence'] for f in menu_frames]
            events.append({
                'event_id': len(events) + 1,
                'timestamp_start': round(menu_start, 2),
                'timestamp_end': round(menu_end, 2),
                'duration': round(duration, 2),
                'mean_ocr_confidence': round(np.mean(confidences), 3),
                'max_ocr_confidence': round(np.max(confidences), 3),
                'n_frames': len(menu_frames),
                'detected_text': menu_frames[0]['detected_text'],
            })
    
    events_df = pd.DataFrame(events)
    
    if len(events_df) == 0:
        if verbose:
            print("  No level-up events detected.")
        return events_df
    
    # Merge close events
    merged = [events_df.iloc[0].to_dict()]
    for i in range(1, len(events_df)):
        current = events_df.iloc[i]
        prev = merged[-1]
        gap = current['timestamp_start'] - prev['timestamp_end']
        
        if gap < config['merge_gap']:
            prev['timestamp_end'] = current['timestamp_end']
            prev['duration'] = round(
                prev['timestamp_end'] - prev['timestamp_start'], 2
            )
            prev['n_frames'] = prev['n_frames'] + current['n_frames']
            prev['max_ocr_confidence'] = round(
                max(prev['max_ocr_confidence'], current['max_ocr_confidence']), 3
            )
        else:
            merged.append(current.to_dict())
    
    events_df = pd.DataFrame(merged)
    events_df['event_id'] = range(1, len(events_df) + 1)
    
    # Add display fields
    events_df['time_display'] = events_df['timestamp_start'].apply(
        lambda t: str(timedelta(seconds=int(t)))[2:]
    )
    
    def classify_speed(dur):
        if dur < 3.0: return 'quick'
        elif dur < 8.0: return 'moderate'
        else: return 'slow'
    
    events_df['decision_speed'] = events_df['duration'].apply(classify_speed)
    
    if verbose:
        print(f"  Detected {len(events_df)} level-up events via OCR")
        for _, e in events_df.iterrows():
            print(f"    #{e['event_id']:>2} at {e['time_display']}  "
                  f"dur={e['duration']:.1f}s  {e['decision_speed']:<8}  "
                  f"ocr_conf={e['mean_ocr_confidence']:.2f}  "
                  f"text='{e['detected_text']}'")
    
    return events_df
# The detect_levelups function is specific to the "Level Up" text, 
# but we can create a more generic function that detects events based on any boolean column
# (e.g., 'is_death', 'is_results'). This way, we can reuse the same logic for different types 
# of events without duplicating code.
def detect_events_by_type(features_df, event_column, event_label, config=None, verbose=True):
    """
    Generic event detector — works for any boolean column.
    Reuses the same grouping logic as detect_levelups but 
    takes the column name as a parameter.
    """
    if config is None:
        config = {
            'min_duration': 0.5,
            'max_duration': 60.0,
            'merge_gap': 2.0,
            'min_ocr_confidence': 0.2,
        }
    
    df = features_df.copy()
    
    if event_column not in df.columns:
        if verbose:
            print(f"  Column '{event_column}' not found.")
        return pd.DataFrame()
    
    df['is_event'] = df[event_column].astype(bool)
    
    events = []
    in_event = False
    event_start = 0.0
    event_frames = []
    
    for _, row in df.iterrows():
        if row['is_event'] and not in_event:
            in_event = True
            event_start = row['timestamp']
            event_frames = [row]
        elif row['is_event'] and in_event:
            event_frames.append(row)
        elif not row['is_event'] and in_event:
            in_event = False
            event_end = row['timestamp']
            duration = event_end - event_start
            
            if duration >= config['min_duration'] and duration <= config['max_duration']:
                events.append({
                    'event_id': len(events) + 1,
                    'event_type': event_label,
                    'timestamp_start': round(event_start, 2),
                    'timestamp_end': round(event_end, 2),
                    'duration': round(duration, 2),
                    'n_frames': len(event_frames),
                    'detected_text': event_frames[0].get('detected_text', ''),
                })
    
    # Handle video ending during event
    if in_event:
        event_end = df['timestamp'].iloc[-1]
        duration = event_end - event_start
        if duration >= config['min_duration']:
            events.append({
                'event_id': len(events) + 1,
                'event_type': event_label,
                'timestamp_start': round(event_start, 2),
                'timestamp_end': round(event_end, 2),
                'duration': round(duration, 2),
                'n_frames': len(event_frames),
                'detected_text': event_frames[0].get('detected_text', ''),
            })
    
    events_df = pd.DataFrame(events)
    
    if len(events_df) == 0:
        if verbose:
            print(f"  No {event_label} events detected.")
        return events_df
    
    # Merge close events
    merged = [events_df.iloc[0].to_dict()]
    for i in range(1, len(events_df)):
        current = events_df.iloc[i]
        prev = merged[-1]
        gap = current['timestamp_start'] - prev['timestamp_end']
        
        if gap < config['merge_gap']:
            prev['timestamp_end'] = current['timestamp_end']
            prev['duration'] = round(prev['timestamp_end'] - prev['timestamp_start'], 2)
            prev['n_frames'] = prev['n_frames'] + current['n_frames']
        else:
            merged.append(current.to_dict())
    
    events_df = pd.DataFrame(merged)
    events_df['event_id'] = range(1, len(events_df) + 1)
    events_df['time_display'] = events_df['timestamp_start'].apply(
        lambda t: str(timedelta(seconds=int(t)))[2:]
    )
    
    if verbose:
        print(f"  Detected {len(events_df)} {event_label} events")
        for _, e in events_df.iterrows():
            print(f"    #{e['event_id']:>2} at {e['time_display']}  "
                  f"dur={e['duration']:.1f}s  text='{e['detected_text']}'")
    
    return events_df

# ============================================================
# SECTION 3: SCHEDULE ANALYSIS (unchanged)
# ============================================================

# Analyze the temporal pattern of level-up events
def analyze_schedule(events_df, verbose=True):
    """
    Analyze the temporal pattern of level-up events.
    """
    if len(events_df) < 2: # Need at least 2 events to compute intervals
        if verbose:
            print("  Need at least 2 events for schedule analysis.")
        return {}
    
    intervals = np.diff(events_df['timestamp_start'].values)
    durations = events_df['duration'].values
    
    stats = {
        'n_events': len(events_df), # Number of detected events
        'total_time': round(float(
            events_df['timestamp_end'].iloc[-1] - 
            events_df['timestamp_start'].iloc[0]
        ), 2),
        'mean_interval': round(float(np.mean(intervals)), 2),
        'std_interval': round(float(np.std(intervals)), 2),
        'cv_interval': round(float(np.std(intervals) / np.mean(intervals)), 3),
        'min_interval': round(float(np.min(intervals)), 2),
        'max_interval': round(float(np.max(intervals)), 2),
        'median_interval': round(float(np.median(intervals)), 2),
        'mean_decision_time': round(float(np.mean(durations)), 2),
        'intervals': [round(float(i), 2) for i in intervals],
    }
    
    if stats['cv_interval'] < 0.35:
        stats['schedule_classification'] = 'FR (Fixed Ratio)'
    else:
        stats['schedule_classification'] = 'VR (Variable Ratio)'
    
    if len(intervals) >= 4:
        first_half = np.mean(intervals[:len(intervals)//2])
        second_half = np.mean(intervals[len(intervals)//2:])
        ratio = second_half / first_half if first_half > 0 else 1.0
        stats['interval_trend'] = round(ratio, 2)
    
    if verbose:
        print(f"\n  Schedule Analysis:")
        print(f"    Events: {stats['n_events']}")
        print(f"    Mean interval: {stats['mean_interval']}s ± {stats['std_interval']}s")
        print(f"    CV: {stats['cv_interval']} → {stats['schedule_classification']}")
    
    return stats


# ============================================================
# SECTION 4: VISUALIZATION
# ============================================================

def plot_results(features_df, events_df, schedule_stats, output_path=None):
    """
    Visualize OCR-based detection results.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(
        'Level-Up Detection via OCR — Vampire Survivors',
        fontsize=15, fontweight='bold', y=0.995
    )
    
    t = features_df['timestamp']
    menu_color = '#e74c3c'
    
    def shade_menus(ax):
        for _, e in events_df.iterrows():
            ax.axvspan(
                e['timestamp_start'], e['timestamp_end'],
                alpha=0.15, color=menu_color, zorder=0
            )
    
    # Panel 1: OCR Detection (binary signal)
    ax1 = axes[0]
    levelup_signal = features_df['is_levelup'].astype(float)
    ax1.fill_between(t, levelup_signal, alpha=0.6, color=menu_color, 
                     step='mid', label='"Level Up" text detected')
    ax1.fill_between(t, features_df['is_results'].astype(float), 
                     alpha=0.4, color='#3498db', step='mid',
                     label='"Results" text detected')
    shade_menus(ax1)
    ax1.set_ylabel('Text Detected', fontsize=9)
    ax1.set_title('OCR Text Detection (PRIMARY SIGNAL — reads "Level Up!" from screen)',
                  fontsize=10, fontweight='bold', loc='left')
    ax1.set_ylim(-0.1, 1.3)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['No', 'Yes'])
    ax1.legend(fontsize=8, loc='upper right')
    
    # Panel 2: OCR Confidence
    ax2 = axes[1]
    conf_values = features_df['ocr_confidence'].copy()
    conf_values[~features_df['is_levelup']] = 0
    ax2.bar(t, conf_values, width=0.4, color=menu_color, alpha=0.7)
    shade_menus(ax2)
    ax2.set_ylabel('OCR Confidence', fontsize=9)
    ax2.set_title('OCR Confidence Score (when "Level Up" text is found)',
                  fontsize=10, fontweight='bold', loc='left')
    ax2.set_ylim(0, 1.1)
    
    # Panel 3: Events with annotations
    ax3 = axes[2]
    ax3.fill_between(t, features_df['center_variance'], alpha=0.3, color='#3498db')
    ax3.plot(t, features_df['center_variance'], color='#3498db', 
             linewidth=0.5, alpha=0.7, label='Center Variance')
    for _, e in events_df.iterrows():
        ax3.annotate(
            f"#{e['event_id']} ({e['duration']:.1f}s)",
            xy=(e['timestamp_start'], 500),
            xytext=(0, 15), textcoords='offset points',
            fontsize=7, ha='center', fontweight='bold', color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fce4ec',
                     edgecolor='#c0392b', alpha=0.9)
        )
    shade_menus(ax3)
    ax3.set_ylabel('Center Variance', fontsize=9)
    ax3.set_title(f'Detected Level-Up Events ({len(events_df)} found)',
                  fontsize=10, fontweight='bold', loc='left')
    ax3.legend(fontsize=8, loc='upper right')
    
    # Panel 4: Inter-event intervals
    ax4 = axes[3]
    if len(events_df) > 1:
        intervals = np.diff(events_df['timestamp_start'].values)
        positions = range(1, len(intervals) + 1)
        mean_int = schedule_stats.get('mean_interval', 0)
        colors = ['#e74c3c' if iv > mean_int * 1.5 else '#3498db' 
                  for iv in intervals]
        ax4.bar(positions, intervals, color=colors, alpha=0.7, edgecolor='white')
        ax4.axhline(y=mean_int, color='red', linestyle='--', linewidth=1.5,
                    label=f"Mean: {mean_int}s")
        cv_val = schedule_stats.get('cv_interval', 0)
        sched = schedule_stats.get('schedule_classification', '?')
        ax4.set_title(f'Inter-Level-Up Intervals (CV={cv_val:.2f} → {sched})',
                      fontsize=10, fontweight='bold', loc='left')
        ax4.legend(fontsize=8)
    ax4.set_xlabel('Time (seconds)', fontsize=10)
    ax4.set_ylabel('Interval (seconds)', fontsize=9)
    
    menu_patch = mpatches.Patch(color=menu_color, alpha=0.15, 
                                label='Level-up menu active')
    axes[0].legend(handles=[menu_patch] + axes[0].get_legend_handles_labels()[0],
                   fontsize=7, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {output_path}")
    
    plt.close(fig)


# ============================================================
# SECTION 5: MAIN PIPELINE
# ============================================================

# def run_pipeline(video_path, output_dir='results', sample_rate=2):
#     """
#     Run the OCR-based level-up detection pipeline.
#     """
#     # Convert video_path and output_dir to Path objects
#     video_path = Path(video_path)
#     output_dir = Path(output_dir)
#     # Create output directory if it doesn't exist
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Start the pipeline timer
#     pipeline_start = time.time()
    
#     print("=" * 60)
#     print("LEVEL-UP DETECTION (OCR) — Vampire Survivors")
#     print("Media Neuroscience Lab, UCSB")
#     print("=" * 60)
    
#     # Step 1: Extract features with OCR
#     print("\n[1/5] Extracting features + OCR text detection...")
#     step_start = time.time()
#     # Call the feature extraction function
#     features = extract_features_ocr(video_path, sample_rate=sample_rate)
#     # Save extracted features to a CSV file
#     features.to_csv(output_dir / 'levelup_features_ocr.csv', index=False)
#     step_time = time.time() - step_start
#     print(f"  ⏱ Feature extraction: {timedelta(seconds=int(step_time))}")
    
#     # Step 2: Detect level-ups
#     print("\n[2/5] Grouping detections into events...")
#     events = detect_levelups(features)
#     # Save detected level-ups to a CSV file
#     events.to_csv(output_dir / 'detected_levelups_ocr.csv', index=False)
    
#     # Step 3: Schedule analysis
#     print("\n[3/5] Analyzing reward schedule...")
#     schedule = analyze_schedule(events)
    
#     # Step 4: Visualizations
#     print("\n[4/5] Generating visualizations...")
#     duration = features['timestamp'].max()  # Get the maximum timestamp
#     # Call the plotting function to generate visualizations
#     plot_results(features, events, schedule, 
#                 output_dir / 'levelup_detection_ocr.png')
    
#     # Step 5: Save summary
#     print("\n[5/5] Saving summary...")
#     pipeline_total = time.time() - pipeline_start  # Total pipeline time
    
#     # Create a summary dictionary with results
#     summary = {
#         'video': str(video_path),
#         'duration_seconds': round(duration, 1),
#         'method': 'OCR (EasyOCR)',
#         'sample_rate_hz': sample_rate,
#         'levelups_detected': len(events),
#         'schedule_stats': schedule,
#         'events': events.to_dict('records') if len(events) > 0 else [],
#         'performance': {
#             'total_time_seconds': round(pipeline_total, 2),
#             'total_time_display': str(timedelta(seconds=int(pipeline_total))),
#         },
#     }
    
#     # Save the summary to a JSON file
#     with open(output_dir / 'levelup_summary_ocr.json', 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     # Final report
#     print("\n" + "=" * 60)
#     print("PIPELINE COMPLETE")
#     print("=" * 60)
#     print(f"  Video: {video_path.name}")
#     print(f"  Duration: {timedelta(seconds=int(duration))}")
#     print(f"  Level-ups detected: {len(events)}")
#     print(f"  Method: OCR text detection ('Level Up!' on screen)")
#     if schedule:
#         print(f"  Schedule: {schedule.get('schedule_classification', '?')}")
#         print(f"  Interval CV: {schedule.get('cv_interval', '?')}")
#     print(f"  ⏱ TOTAL: {timedelta(seconds=int(pipeline_total))}")
    
#     return features, events, summary

def run_pipeline(video_path, output_dir='results', sample_rate=4):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_start = time.time()
    
    print("=" * 60)
    print("EVENT DETECTION (OCR) — Vampire Survivors")
    print("Media Neuroscience Lab, UCSB")
    print("=" * 60)
    
    # Step 1: Extract features with OCR
    print("\n[1/6] Extracting features + OCR text detection...")
    step_start = time.time()
    features = extract_features_ocr(video_path, sample_rate=sample_rate)
    features.to_csv(output_dir / 'all_features_ocr.csv', index=False)
    step_time = time.time() - step_start
    print(f"  ⏱ Feature extraction: {timedelta(seconds=int(step_time))}")
    
    # Step 2: Detect Level-Up events
    print("\n[2/6] Detecting Level-Up events...")
    levelup_events = detect_levelups(features)
    levelup_events.to_csv(output_dir / 'detected_levelups.csv', index=False)
    
    # Step 3: Detect Death events
    print("\n[3/6] Detecting Death events...")
    death_events = detect_events_by_type(
        features, 'is_death', 'Death',
        config={'min_duration': 0.5, 'max_duration': 60.0, 'merge_gap': 3.0, 'min_ocr_confidence': 0.2}
    )
    death_events.to_csv(output_dir / 'detected_deaths.csv', index=False)
    
    # Step 4: Detect Results events
    print("\n[4/6] Detecting Results events...")
    results_events = detect_events_by_type(
        features, 'is_results', 'Results',
        config={'min_duration': 0.5, 'max_duration': 60.0, 'merge_gap': 3.0, 'min_ocr_confidence': 0.2}
    )
    results_events.to_csv(output_dir / 'detected_results.csv', index=False)
    
    # Step 5: Schedule analysis (on level-ups)
    print("\n[5/6] Analyzing reward schedule...")
    schedule = analyze_schedule(levelup_events)
    
    # Step 6: Save combined summary
    print("\n[6/6] Saving summary...")
    pipeline_total = time.time() - pipeline_start
    duration = features['timestamp'].max()
    
    # Visualizations
    plot_results(features, levelup_events, schedule, output_dir / 'levelup_detection_ocr.png')
    
    summary = {
        'video': str(video_path),
        'duration_seconds': round(duration, 1),
        'duration_display': str(timedelta(seconds=int(duration))),
        'method': 'OCR (EasyOCR)',
        'sample_rate_hz': sample_rate,
        'event_counts': {
            'levelups': len(levelup_events),
            'deaths': len(death_events),
            'results_screens': len(results_events),
        },
        'schedule_stats': schedule,
        'levelup_events': levelup_events.to_dict('records') if len(levelup_events) > 0 else [],
        'death_events': death_events.to_dict('records') if len(death_events) > 0 else [],
        'results_events': results_events.to_dict('records') if len(results_events) > 0 else [],
        'performance': {
            'total_time_seconds': round(pipeline_total, 2),
            'total_time_display': str(timedelta(seconds=int(pipeline_total))),
        },
    }
    
    with open(output_dir / 'event_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final report
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Video: {video_path.name}")
    print(f"  Duration: {timedelta(seconds=int(duration))}")
    print(f"\n  Event Counts:")
    print(f"    Level-Ups:       {len(levelup_events)}")
    print(f"    Deaths:          {len(death_events)}")
    print(f"    Results Screens: {len(results_events)}")
    if schedule:
        print(f"\n  Level-Up Schedule: {schedule.get('schedule_classification', '?')}")
        print(f"  Interval CV: {schedule.get('cv_interval', '?')}")
    print(f"\n  ⏱ TOTAL: {timedelta(seconds=int(pipeline_total))}")
    print(f"\n  Output: {output_dir}/")
    print(f"    ├── all_features_ocr.csv")
    print(f"    ├── detected_levelups.csv")
    print(f"    ├── detected_deaths.csv")
    print(f"    ├── detected_results.csv")
    print(f"    ├── levelup_detection_ocr.png")
    print(f"    └── event_summary.json")
    
    return features, levelup_events, death_events, results_events, summary

if __name__ == '__main__':
    # Run the pipeline with specified video path and output directory
    run_pipeline(
        video_path='/Users/artiramanathan/Projects/mnl-learning/module4_capstone/video4_Imelda_100.mp4',
        output_dir='/Users/artiramanathan/Projects/mnl-learning/module4_capstone/results_ocr_v5_video4_fps4/',
        sample_rate=4
    )