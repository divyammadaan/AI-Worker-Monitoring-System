import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque

# Load YOLOv8 model with tracking
model = YOLO("yolov8n.pt")
# Video input: webcam (0) or video file
cap = cv2.VideoCapture(0)

# Define desk zone polygon (adjust as needed for your setup)
zone_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
zone_polygon = np.array(zone_points, np.int32)

# Smoothing parameters
SMOOTHING_BUFFER_SIZE = 8   # Number of frames to consider for smoothing
MIN_STABLE_FRAMES = 3       # Minimum frames a state must persist to be considered stable
DEBOUNCE_TIME = 1.5         # Minimum seconds between entry/exit events for same person

# Enhanced tracking parameters
FEATURE_UPDATE_INTERVAL = 10    # Update features every N frames
MAX_TRACK_LOST_FRAMES = 60     # Maximum frames before considering track truly lost
RE_ID_CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for better re-identification

# Time tracking structures
entry_times = {}
durations = defaultdict(float)
present_inside = set()

# Person re-identification structures
person_features = {}
consistent_id_counter = 0
track_to_consistent_map = {}
consistent_to_last_seen = {}
consistent_id_last_position = {}
feature_update_counter = defaultdict(int)
re_id_threshold = RE_ID_CONFIDENCE_THRESHOLD
max_missing_frames = MAX_TRACK_LOST_FRAMES

# Smoothing structures
person_zone_history = defaultdict(lambda: deque(maxlen=SMOOTHING_BUFFER_SIZE))
person_stable_state = {}
last_event_time = defaultdict(float)

# CSV file paths
LOG_FILENAME = "worker_monitoring_log.csv"
SUMMARY_FILENAME = "worker_monitoring_summary.csv"

# Enhanced logging structures
class WorkerSession:
    def __init__(self, worker_id, entry_time):
        self.worker_id = worker_id
        self.entry_time = entry_time
        self.exit_time = None
        self.duration = None
    
    def complete_session(self, exit_time):
        self.exit_time = exit_time
        self.duration = exit_time - self.entry_time
    
    def mark_incomplete(self, end_time):
        self.exit_time = end_time
        self.duration = end_time - self.entry_time

# Store all sessions for comprehensive logging
all_sessions = []
active_sessions = {}  # worker_id -> WorkerSession

def initialize_csv_files():
    """Initialize CSV files with headers if they don't exist"""
    # Initialize main log file
    if not os.path.exists(LOG_FILENAME):
        with open(LOG_FILENAME, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Worker_ID",
                "Entry_Date_Time", 
                "Exit_Date_Time", 
                "Duration"
            ])
        print(f"Created new log file: {LOG_FILENAME}")
    else:
        print(f"Using existing log file: {LOG_FILENAME}")
    
    # Initialize summary file
    if not os.path.exists(SUMMARY_FILENAME):
        with open(SUMMARY_FILENAME, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Worker_ID",
                "Total_Duration",
                "Session_Count",
                "Last_Activity"
            ])
        print(f"Created new summary file: {SUMMARY_FILENAME}")
    else:
        print(f"Using existing summary file: {SUMMARY_FILENAME}")

def write_entry_to_csv(worker_id, entry_time):
    """Write worker entry to CSV immediately"""
    entry_datetime = datetime.fromtimestamp(entry_time).strftime("%Y-%m-%d %H:%M:%S")
    
    # Append entry with empty exit time and duration
    with open(LOG_FILENAME, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            worker_id,
            entry_datetime,
            "",  # Empty exit time
            ""   # Empty duration
        ])
    
    print(f"📝 Entry logged to CSV for Worker {worker_id}")

def update_exit_in_csv(worker_id, entry_time, exit_time, duration):
    """Update the CSV file with exit time and duration for a specific worker entry"""
    entry_datetime = datetime.fromtimestamp(entry_time).strftime("%Y-%m-%d %H:%M:%S")
    exit_datetime = datetime.fromtimestamp(exit_time).strftime("%Y-%m-%d %H:%M:%S")
    duration_formatted = format_duration(duration)
    
    # Read all rows from CSV
    rows = []
    with open(LOG_FILENAME, mode="r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Find and update the matching row (last entry for this worker with empty exit time)
    updated = False
    for i in range(len(rows) - 1, 0, -1):  # Start from end, skip header
        if (rows[i][0] == str(worker_id) and 
            rows[i][1] == entry_datetime and 
            rows[i][2] == ""):  # Empty exit time
            rows[i][2] = exit_datetime
            rows[i][3] = duration_formatted
            updated = True
            break
    
    if updated:
        # Write back all rows
        with open(LOG_FILENAME, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        print(f"📝 Exit logged to CSV for Worker {worker_id}")
    else:
        print(f"⚠️ Could not find matching entry to update for Worker {worker_id}")

def update_summary_csv():
    """Update the summary CSV with current totals"""
    # Read existing summary data
    existing_data = {}
    if os.path.exists(SUMMARY_FILENAME):
        with open(SUMMARY_FILENAME, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_data[int(row['Worker_ID'])] = row
    
    # Update with current session data
    current_time = time.time()
    
    with open(SUMMARY_FILENAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Worker_ID",
            "Total_Duration", 
            "Session_Count",
            "Last_Activity"
        ])
        
        # Write data for all workers who have had sessions
        all_worker_ids = set(durations.keys()) | set(active_sessions.keys())
        
        for worker_id in sorted(all_worker_ids):
            # Calculate total duration including current active session
            total_duration = durations[worker_id]
            if worker_id in active_sessions:
                total_duration += current_time - active_sessions[worker_id].entry_time
            
            # Count completed sessions plus active session
            completed_sessions = len([s for s in all_sessions if s.worker_id == worker_id])
            active_session_count = 1 if worker_id in active_sessions else 0
            total_sessions = completed_sessions + active_session_count
            
            # Find last activity time
            last_activity = 0
            if worker_id in active_sessions:
                last_activity = active_sessions[worker_id].entry_time
            else:
                worker_sessions = [s for s in all_sessions if s.worker_id == worker_id]
                if worker_sessions:
                    last_activity = max(s.entry_time for s in worker_sessions)
            
            last_activity_str = datetime.fromtimestamp(last_activity).strftime("%Y-%m-%d %H:%M:%S") if last_activity else "N/A"
            
            writer.writerow([
                worker_id,
                format_duration(total_duration),
                total_sessions,
                last_activity_str
            ])

def extract_features(frame, bbox):
    """Extract simple features from a person's bounding box region"""
    x1, y1, x2, y2 = bbox
    # Crop the person region
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return None
    
    # Resize to standard size for consistent feature extraction
    person_crop = cv2.resize(person_crop, (64, 128))
    
    # Extract color histogram features
    hist_b = cv2.calcHist([person_crop], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([person_crop], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([person_crop], [2], None, [32], [0, 256])
    
    # Normalize histograms
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    
    # Combine features
    features = np.concatenate([hist_b, hist_g, hist_r])
    return features

def calculate_similarity(feat1, feat2):
    """Calculate cosine similarity between two feature vectors"""
    if feat1 is None or feat2 is None:
        return 0
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def find_best_match(features, existing_features, current_position, threshold=0.25):
    """Find the best matching consistent ID for given features and position"""
    if features is None:
        return None
        
    best_match = None
    best_score = threshold
    
    for consistent_id, stored_features in existing_features.items():
        # Calculate feature similarity
        feature_similarity = calculate_similarity(features, stored_features)
        
        # Calculate position similarity if we have last known position
        position_similarity = 0
        if consistent_id in consistent_id_last_position:
            last_pos = consistent_id_last_position[consistent_id]
            current_pos = current_position
            distance = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
            # Normalize distance (closer = higher similarity)
            position_similarity = max(0, 1 - distance / 500)  # 500 pixels max reasonable movement
        
        # Combined score (weighted towards features but considering position)
        combined_score = 0.8 * feature_similarity + 0.2 * position_similarity
        
        if combined_score > best_score:
            best_score = combined_score
            best_match = consistent_id
    
    return best_match

def get_smoothed_zone_state(person_id, current_inside_state):
    """Apply smoothing to zone entry/exit detection"""
    # Add current state to history
    person_zone_history[person_id].append(current_inside_state)
    
    # If we don't have enough history, return None (uncertain)
    if len(person_zone_history[person_id]) < MIN_STABLE_FRAMES:
        return None
    
    # Check if the recent states are consistent
    recent_states = list(person_zone_history[person_id])[-MIN_STABLE_FRAMES:]
    
    if all(state for state in recent_states):
        # All recent states show "inside"
        return True
    elif not any(state for state in recent_states):
        # All recent states show "outside"  
        return False
    else:
        # Mixed states - keep current stable state
        return person_stable_state.get(person_id, None)

def should_trigger_event(person_id, new_state, current_time):
    """Check if we should trigger an entry/exit event based on debouncing"""
    # Check time-based debouncing
    if current_time - last_event_time[person_id] < DEBOUNCE_TIME:
        return False
    
    # Check if state actually changed
    current_stable_state = person_stable_state.get(person_id, None)
    return new_state != current_stable_state and new_state is not None

def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def save_final_csv():
    """Save final CSV data for any remaining active sessions"""
    program_end_time = time.time()
    
    # Mark any remaining active sessions as incomplete
    for worker_id, session in active_sessions.items():
        session.mark_incomplete(program_end_time)
        all_sessions.append(session)
        
        # Update the CSV with final exit time
        update_exit_in_csv(worker_id, session.entry_time, session.exit_time, session.duration)
    
    # Update final summary
    update_summary_csv()
    
    return LOG_FILENAME, SUMMARY_FILENAME

frame_count = 0

try:
    start_time = datetime.now()
    print("Worker Monitoring System Started")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Initialize CSV files
    initialize_csv_files()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera/video source")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Run YOLOv8 with tracking enabled
        results = model.track(frame, persist=True, verbose=False)[0]
        
        # Draw desk area on frame
        cv2.polylines(frame, [zone_polygon], True, (0, 255, 0), 2)
        
        current_track_ids = set()
        detected_persons = []  # Store all detected persons for batch processing
        
        if results.boxes is not None:
            # First pass: collect all person detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id.item()) if box.id is not None else None
                cls = int(box.cls.item())
                
                if cls != 0:  # Only process 'person' class
                    continue
                
                # Calculate bounding box center and zone status
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                raw_inside = cv2.pointPolygonTest(zone_polygon, (cx, cy), False) >= 0
                
                detected_persons.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': track_id,
                    'center': (cx, cy),
                    'raw_inside': raw_inside,
                    'consistent_id': None
                })
                
                if track_id is not None:
                    current_track_ids.add(track_id)
        
        # Second pass: process each detection for consistent ID assignment
        for detection in detected_persons:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']
            cx, cy = detection['center']
            raw_inside = detection['raw_inside']
            
            # Extract features for this detection
            features = extract_features(frame, detection['bbox'])
            consistent_id = None
            
            # Case 1: We have a valid track_id and it's already mapped
            if track_id is not None and track_id in track_to_consistent_map:
                consistent_id = track_to_consistent_map[track_id]
                
                # Periodically update features for this consistent ID
                feature_update_counter[consistent_id] += 1
                if feature_update_counter[consistent_id] % FEATURE_UPDATE_INTERVAL == 0 and features is not None:
                    person_features[consistent_id] = features
            
            # Case 2: We have a track_id but no mapping, or no track_id at all
            else:
                if features is not None:
                    # Try to match with existing persons using features and position
                    matched_id = find_best_match(features, person_features, (cx, cy), re_id_threshold)
                    
                    if matched_id is not None:
                        # Check if this matched ID hasn't been seen recently enough to be re-identification
                        frames_since_last_seen = frame_count - consistent_to_last_seen.get(matched_id, 0)
                        
                        # Allow re-identification if not seen for a while but not too long
                        if 5 <= frames_since_last_seen <= max_missing_frames:
                            consistent_id = matched_id
                            if track_id is not None:
                                track_to_consistent_map[track_id] = consistent_id
                            person_features[consistent_id] = features  # Update features
                            print(f"🔄 Re-identified Worker {consistent_id} after {frames_since_last_seen} frames")
                
                # If no match found, create new consistent ID
                if consistent_id is None:
                    consistent_id = consistent_id_counter
                    consistent_id_counter += 1
                    if track_id is not None:
                        track_to_consistent_map[track_id] = consistent_id
                    if features is not None:
                        person_features[consistent_id] = features
                    print(f"🆕 New Worker {consistent_id} detected")
            
            # Update tracking information
            consistent_to_last_seen[consistent_id] = frame_count
            consistent_id_last_position[consistent_id] = (cx, cy)
            detection['consistent_id'] = consistent_id
                
        # Third pass: handle zone entry/exit logic and visualization
        for detection in detected_persons:
            if detection['consistent_id'] is None:
                continue
                
            consistent_id = detection['consistent_id']
            x1, y1, x2, y2 = detection['bbox']
            raw_inside = detection['raw_inside']
            
            # Apply smoothing to zone detection
            smoothed_inside = get_smoothed_zone_state(consistent_id, raw_inside)
            
            # Check if we should trigger an entry/exit event
            if smoothed_inside is not None and should_trigger_event(consistent_id, smoothed_inside, current_time):
                if smoothed_inside and consistent_id not in present_inside:
                    # Person entered the zone (smoothed)
                    entry_times[consistent_id] = current_time
                    present_inside.add(consistent_id)
                    person_stable_state[consistent_id] = True
                    last_event_time[consistent_id] = current_time
                    
                    # Create new session
                    new_session = WorkerSession(consistent_id, current_time)
                    active_sessions[consistent_id] = new_session
                    
                    # Write entry to CSV immediately
                    write_entry_to_csv(consistent_id, current_time)
                    
                    # Update summary CSV
                    update_summary_csv()
                    
                    entry_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"✓ Worker {consistent_id} ENTERED at {entry_time_str}")
                    
                elif not smoothed_inside and consistent_id in present_inside:
                    # Person exited the zone (smoothed)
                    present_inside.remove(consistent_id)
                    person_stable_state[consistent_id] = False
                    last_event_time[consistent_id] = current_time
                    
                    # Complete the active session
                    if consistent_id in active_sessions:
                        session = active_sessions[consistent_id]
                        session.complete_session(current_time)
                        all_sessions.append(session)
                        
                        # Update CSV with exit time and duration immediately
                        update_exit_in_csv(consistent_id, session.entry_time, session.exit_time, session.duration)
                        
                        del active_sessions[consistent_id]
                        
                        # Update summary CSV
                        update_summary_csv()
                        
                        exit_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
                        duration_str = format_duration(session.duration)
                        print(f"✗ Worker {consistent_id} EXITED at {exit_time_str} (Duration: {duration_str})")
                    
                    # Update total durations
                    if consistent_id in entry_times:
                        durations[consistent_id] += current_time - entry_times[consistent_id]
            
            # Calculate live display time
            display_time = durations[consistent_id]
            is_currently_inside = consistent_id in present_inside
            
            if is_currently_inside and consistent_id in entry_times:
                display_time += current_time - entry_times[consistent_id]
            
            # Use stable state for visual display
            display_inside = person_stable_state.get(consistent_id, False)
            
            label = f"ID {consistent_id} {format_duration(display_time)}"
            color = (0, 255, 0) if display_inside else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show detection status
            if len(person_zone_history[consistent_id]) < MIN_STABLE_FRAMES:
                status_text = "DETECTING..."
                cv2.putText(frame, status_text, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Show confidence indicator
            last_seen_frames = frame_count - consistent_to_last_seen[consistent_id]
            if last_seen_frames > 30:
                cv2.putText(frame, "TRACKING...", (x1, y2 + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        # Clean up old mappings more aggressively but keep consistent IDs longer
        inactive_tracks = set(track_to_consistent_map.keys()) - current_track_ids
        for track_id in inactive_tracks:
            consistent_id = track_to_consistent_map[track_id]
            frames_since_last_seen = frame_count - consistent_to_last_seen.get(consistent_id, 0)
            
            # Only remove track mapping (not consistent ID) after track is lost
            if frames_since_last_seen > max_missing_frames // 2:
                del track_to_consistent_map[track_id]
        
        # Clean up very old consistent IDs that haven't been seen for a very long time
        old_consistent_ids = []
        for consistent_id, last_seen_frame in consistent_to_last_seen.items():
            if frame_count - last_seen_frame > max_missing_frames * 3:
                old_consistent_ids.append(consistent_id)
        
        for old_id in old_consistent_ids:
            if old_id in person_features:
                del person_features[old_id]
            if old_id in consistent_to_last_seen:
                del consistent_to_last_seen[old_id]
            if old_id in consistent_id_last_position:
                del consistent_id_last_position[old_id]
        
        # Display current active workers
        active_count = len(active_sessions)
        cv2.putText(frame, f"Active Workers: {active_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Worker Monitoring - Real-time CSV Updates", frame)
        
        # Improved key detection with multiple methods
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q', or ESC key
            print("\nExit key detected. Stopping...")
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Cleanup and save data
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("FINALIZING DATA...")
    log_file, summary_file = save_final_csv()
    
    print("\nFiles updated:")
    print(f"1. {log_file} - Real-time entry/exit log")
    print(f"2. {summary_file} - Live summary statistics")
    
    # Final console report
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    
    if all_sessions:
        print(f"\nTotal Sessions Recorded: {len(all_sessions)}")
        print("\nSession Details:")
        for i, session in enumerate(all_sessions, 1):
            entry_time = datetime.fromtimestamp(session.entry_time).strftime('%Y-%m-%d %H:%M:%S')
            exit_time = datetime.fromtimestamp(session.exit_time).strftime('%Y-%m-%d %H:%M:%S') if session.exit_time else "N/A"
            duration = format_duration(session.duration) if session.duration else "N/A"
            print(f"  Worker {session.worker_id}: {entry_time} → {exit_time} ({duration})")
    
    if durations:
        print("\nTotal Time Summary:")
        for worker_id, total_time in durations.items():
            print(f"  Worker {worker_id}: {format_duration(total_time)} total")
    else:
        print("\nNo worker activity detected.")
    
    print(f"\nMonitoring completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")