"""
Tobii Gaze Data Preprocessing Module

This module provides functions for loading, cleaning, and integrating Tobii eye-tracking data
with response and stimuli information.

Main features:
- Load raw gaze data
- Parse and normalize coordinates
- Classify events (fixation/saccade)
- Infer trial information
- Merge response metadata
- Extract AOI boundaries
- Summarize data quality
"""

import os
import pandas as pd
import numpy as np
import glob
import re


def parse_point(point_str):
    """
    Parse a coordinate string such as "(0.5, 0.3)" into `(0.5, 0.3)`.
    """
    try:
        if pd.isna(point_str) or point_str == '':
            return (np.nan, np.nan)
        # Remove brackets and whitespace, then split.
        point_str = str(point_str).strip('()').replace(' ', '')
        if ',' in point_str:
            parts = point_str.split(',')
            if len(parts) == 2:
                x = float(parts[0])
                y = float(parts[1])
                return (x, y)
        return (np.nan, np.nan)
    except:
        return (np.nan, np.nan)


def load_raw_gaze(participant_id, gaze_dir):
    """
    Load the raw gaze-data file for one participant.

    Parameters:
        participant_id: Participant ID (str or int)
        gaze_dir: Directory containing gaze files

    Returns:
        DataFrame or None if no matching file is found.
    """
    pid = str(participant_id)
    
    # Candidate filename patterns.
    candidate_patterns = [
        f"gaze_data_list_P-{pid}.csv",
        f"gaze_data_P-{pid}.csv", 
        f"gaze_P-{pid}.csv",
        f"raw_gaze_P-{pid}.csv",
        f"*P-{pid}*.csv",  # Wildcard fallback.
    ]
    
    # Search for a matching file.
    for pattern in candidate_patterns:
        if '*' in pattern:
            # Search with glob.
            search_path = os.path.join(gaze_dir, pattern)
            matches = glob.glob(search_path)
            if matches:
                target_file = matches[0]  # Use the first match.
                break
        else:
            # Direct file path.
            target_file = os.path.join(gaze_dir, pattern)
            if os.path.exists(target_file):
                break
    else:
        # If nothing matched, search for files containing the participant ID.
        try:
            all_files = [f for f in os.listdir(gaze_dir) if f.endswith('.csv')]
            for filename in all_files:
                if pid in filename and ('gaze' in filename.lower() or 'data' in filename.lower()):
                    target_file = os.path.join(gaze_dir, filename)
                    break
            else:
                print(f"❌ No gaze file found for participant P-{pid}")
                print(f"   Search directory: {gaze_dir}")
                print(f"   Available CSV files: {all_files[:5]}...")  # Show the first 5.
                return None
        except:
            print(f"❌ Cannot access directory: {gaze_dir}")
            return None
    
    # Load the file.
    try:
        print(f"📄 Loading raw gaze file: {os.path.basename(target_file)}")
        df = pd.read_csv(target_file)
        
        # Basic validation.
        if len(df) == 0:
            print(f"⚠️ File is empty: {target_file}")
            return None
        
        # Sort by time if a timestamp column exists.
        time_cols = ['device_time_stamp', 'timestamp', 'time', 'system_time_stamp']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
            print(f"✅ Sorted by {time_col}; rows: {len(df)}")
        else:
            print(f"✅ Loaded successfully; rows: {len(df)} (no timestamp sort)")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to read file {target_file}: {e}")
        return None


# Classify each gaze point as fixation or saccade.
def classify_events_ivt(
    gaze: pd.DataFrame,
    screen_width_px: int = 1920,
    screen_height_px: int = 1080,
    screen_width_cm: float = 59.7,
    screen_height_cm: float = 33.6,
    view_dist_cm: float = 65.0,
    vel_thresh_dps: float = 30.0,     # Angular-velocity threshold (°/s); Tobii often uses ~30.
    smooth_window: int = 5,           # Rolling median smoothing window in samples.
    min_fix_dur_ms: int = 80,         # Minimum fixation duration in ms.
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Use I-VT (velocity-threshold classification) to label gaze points as
    fixation or saccade.

    Requires `device_time_stamp` (microseconds) and normalized 0..1 coordinates
    (`avg_x_raw`/`avg_y_raw`). If those are unavailable, the function back-
    converts from `stimuli_x`/`stimuli_y` in the -1..1 coordinate space.

    Output columns:
        - `velocity_deg_s`: estimated angular velocity (°/s), smoothed
        - `event_type`: `fixation` or `saccade`
        - `fixation_id`: segment ID for fixation runs, `NaN` for saccades
    """

    df = gaze.copy()

    # 0) Check required columns.
    if 'device_time_stamp' not in df.columns:
        if verbose:
            print("⚠️ Missing device_time_stamp; velocity-based classification is skipped and points are labeled 'other'.")
        df['event_type'] = 'other'
        df['velocity_deg_s'] = np.nan
        df['fixation_id'] = np.nan
        return df

    # 1) Get coordinates: prefer 0..1 avg_x_raw/avg_y_raw, otherwise infer from -1..1 stimuli_x/y.
    if {'avg_x_raw', 'avg_y_raw'}.issubset(df.columns):
        nx = pd.to_numeric(df['avg_x_raw'], errors='coerce')  # 0..1
        ny = pd.to_numeric(df['avg_y_raw'], errors='coerce')  # 0..1
    elif {'stimuli_x', 'stimuli_y'}.issubset(df.columns):
        # Earlier convention: stimuli_x = 2*avg_x_raw - 1, stimuli_y = 1 - 2*avg_y_raw.
        # Invert that mapping here.
        nx = (pd.to_numeric(df['stimuli_x'], errors='coerce') + 1.0) / 2.0
        ny = (1.0 - pd.to_numeric(df['stimuli_y'], errors='coerce')) / 2.0
    else:
        if verbose:
            print("⚠️ Cannot find avg_x_raw/avg_y_raw or stimuli_x/stimuli_y; velocity cannot be computed and points are labeled 'other'.")
        df['event_type'] = 'other'
        df['velocity_deg_s'] = np.nan
        df['fixation_id'] = np.nan
        return df

    # 2) Time (μs → s).
    t_us = pd.to_numeric(df['device_time_stamp'], errors='coerce')
    # Keep strict ascending time order.
    order = np.argsort(t_us.values)
    df = df.iloc[order].reset_index(drop=True)
    nx = nx.iloc[order].reset_index(drop=True)
    ny = ny.iloc[order].reset_index(drop=True)
    t_us = t_us.iloc[order].reset_index(drop=True)

    # 3) Convert pixel displacement to centimeters.
    x_px = nx * float(screen_width_px)
    y_px = ny * float(screen_height_px)
    # Assume square pixels and convert per axis using px/cm.
    px_per_cm_x = float(screen_width_px) / float(screen_width_cm)
    px_per_cm_y = float(screen_height_px) / float(screen_height_cm)

    dx_px = x_px.diff().fillna(0.0)
    dy_px = y_px.diff().fillna(0.0)
    dx_cm = dx_px / px_per_cm_x
    dy_cm = dy_px / px_per_cm_y
    d_cm = np.sqrt(dx_cm**2 + dy_cm**2)

    # 4) Angular displacement (radians/degrees): 2 * arctan(d / (2 * D)).
    # Here d_cm is the planar displacement between adjacent samples and D is the viewing distance.
    angle_rad = 2.0 * np.arctan2(d_cm.astype(float), 2.0 * float(view_dist_cm))
    angle_deg = np.degrees(angle_rad)

    # 5) Angular velocity (°/s).
    dt_s = (t_us.diff().fillna(0.0) / 1_000_000.0).astype(float)
    # Avoid zero or negative intervals.
    dt_s = dt_s.mask(dt_s <= 0, other=np.nan)
    vel_dps = angle_deg / dt_s
    vel_dps = vel_dps.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 6) Smooth with a rolling median.
    if smooth_window and smooth_window > 1:
        vel_s = pd.Series(vel_dps).rolling(window=smooth_window, center=True, min_periods=1).median().values
    else:
        vel_s = vel_dps.values

    df['velocity_deg_s'] = vel_s

    # 7) Initial I-VT classification (below threshold -> fixation).
    low = vel_s < float(vel_thresh_dps)

    # 8) Merge consecutive low-velocity runs and apply the minimum-duration filter.
    event_type = np.array(['saccade'] * len(df), dtype=object)
    fixation_id = np.full(len(df), np.nan)

    # Find start/end indices of low-velocity runs.
    low_int = low.astype(int)
    # Transition boundaries.
    trans = np.diff(np.r_[0, low_int, 0])
    starts = np.where(trans == 1)[0]
    ends   = np.where(trans == -1)[0] - 1

    fix_counter = 0
    for s, e in zip(starts, ends):
        # Duration of the segment.
        seg_dt_s = (t_us.iloc[e] - t_us.iloc[s]) / 1_000_000.0
        if seg_dt_s * 1000.0 >= float(min_fix_dur_ms):
            # Keep as a fixation.
            event_type[s:e+1] = 'fixation'
            fix_counter += 1
            fixation_id[s:e+1] = fix_counter
        # Otherwise keep it as a saccade.

    df['event_type'] = event_type
    df['fixation_id'] = fixation_id

    if verbose:
        vc = pd.Series(event_type).value_counts().to_dict()
        tot = len(event_type)
        print(
            f"📊 I-VT classification: fixation={vc.get('fixation',0)} ({vc.get('fixation',0)/tot*100:.1f}%), "
            f"saccade={vc.get('saccade',0)} ({vc.get('saccade',0)/tot*100:.1f}%), "
            f"threshold={vel_thresh_dps}°/s, smoothing={smooth_window}, min_fix={min_fix_dur_ms}ms"
        )

    return df


def integrate_participant(participant_id, gaze_dir, response_dir, cleaned_dir, stimuli_dir,
                         source='raw', save=True, save_cleaned=False, verbose=True, 
                         join_meta=True, force_reclean=False):
    """
        Integrate a participant's gaze data with response metadata and assign AOIs.

        Steps:
            1. Load cleaned or raw data and clean coordinates
            2. Classify events (fixation / saccade) using relaxed reading parameters
            3. Infer `trial_num` in a vectorized way
            4. Merge response metadata at the trial level
            5. Assign AOIs based on y-coordinate boundary checks
            6. Save `integrated_gaze_P-*.csv`

        Parameters:
                participant_id: Participant ID
                gaze_dir: Gaze-data directory
                response_dir: Response-data directory
                cleaned_dir: Output directory
                stimuli_dir: Stimuli-data directory used for AOI-boundary files
                source: `raw` or `cleaned`
                save: Whether to save the integrated file
                save_cleaned: Whether to save the cleaned intermediate file
                verbose: Whether to print detailed logs
                join_meta: Whether to merge trial metadata
                force_reclean: Whether to force re-cleaning

        Returns:
                DataFrame: Integrated gaze data
    """
    pid = str(participant_id)
    cleaned_path = os.path.join(cleaned_dir, f"icassp_cleaned_gaze_P-{pid}.csv")

    gaze = None
    use_raw = (source == 'raw')
    if not use_raw:
        if os.path.exists(cleaned_path) and not force_reclean:
            gaze = pd.read_csv(cleaned_path)
            if verbose: 
                print(f"📄 已加载 cleaned 文件: {os.path.basename(cleaned_path)}")
        else:
            if verbose:
                reason = '不存在' if not os.path.exists(cleaned_path) else 'force_reclean=True'
                print(f"⚠️ cleaned 文件{reason} -> 回退 raw")
            use_raw = True

    if use_raw:
        raw = load_raw_gaze(pid, gaze_dir)
        if raw is None:
            raise FileNotFoundError(f"未找到原始 gaze 文件 P-{pid}")
        gaze = raw.copy()

        # Parse coordinates.
        if 'left_gaze_point_on_display_area' in gaze.columns:
            left_xy = gaze['left_gaze_point_on_display_area'].apply(parse_point)
            gaze['left_x'] = [p[0] for p in left_xy]
            gaze['left_y'] = [p[1] for p in left_xy]
        if 'right_gaze_point_on_display_area' in gaze.columns:
            right_xy = gaze['right_gaze_point_on_display_area'].apply(parse_point)
            gaze['right_x'] = [p[0] for p in right_xy]
            gaze['right_y'] = [p[1] for p in right_xy]
        if {'left_x','right_x'}.issubset(gaze.columns):
            gaze['avg_x_raw'] = gaze[['left_x','right_x']].mean(axis=1)
            gaze['avg_y_raw'] = gaze[['left_y','right_y']].mean(axis=1)
        elif 'left_x' in gaze.columns:
            gaze['avg_x_raw'] = gaze['left_x']; gaze['avg_y_raw'] = gaze['left_y']
        elif 'right_x' in gaze.columns:
            gaze['avg_x_raw'] = gaze['right_x']; gaze['avg_y_raw'] = gaze['right_y']

        # Normalize to the (-1, 1) space assuming the raw coordinates are 0..1.
        if 'avg_x_raw' in gaze.columns:
            gaze['stimuli_x'] = 2*gaze['avg_x_raw'] - 1
            gaze['stimuli_y'] = 1 - 2*gaze['avg_y_raw']


        gaze = classify_events_ivt(
            gaze,
            screen_width_px=1920, screen_height_px=1080,
            screen_width_cm=59.7,  screen_height_cm=33.6,
            view_dist_cm=60.0,
            vel_thresh_dps=35.0, #30.0,     # 角速度阈值，越大 fixation越多 saccade越少
            smooth_window=3, #5,         # 中位数平滑窗口（样本数）
            min_fix_dur_ms=90, #80,       # 最小注视持续，越小 fixation越多 saccade越少
            verbose=True
        )

        # Pupil diameter.
        if 'left_pupil_diameter' in gaze.columns:
            gaze['pupil_diameter'] = pd.to_numeric(gaze['left_pupil_diameter'], errors='coerce')
        elif 'pupil_diameter' not in gaze.columns:
            gaze['pupil_diameter'] = np.nan

        if save_cleaned:
            os.makedirs(cleaned_dir, exist_ok=True)
            gaze.to_csv(cleaned_path, index=False)
            if verbose: print(f"💾 保存 cleaned: {cleaned_path}")
        else:
            if verbose: print("🧪 内存清洗 (未写入 cleaned 文件)")

    gaze['participant_id'] = pid

    # Infer `trial_num`.
    if ('trial_num' not in gaze.columns) or (gaze['trial_num'].notna().sum()==0):
        if 'phase' in gaze.columns:
            phase = gaze['phase'].astype(str)
            starts = (phase=='START').astype(int)
            ends = (phase=='END').astype(int)
            starts_cum = starts.cumsum(); ends_cum = ends.cumsum()
            in_trial = (starts_cum - ends_cum) > 0
            trial_num = starts_cum.where(in_trial | (phase=='START'))
            trial_num = trial_num.where(trial_num>0)
            gaze['trial_num'] = trial_num
            try: gaze['trial_num'] = gaze['trial_num'].astype('Int64')
            except: pass
        else:
            if verbose: print('⚠️ No phase column found; trial_num remains NaN')
            gaze['trial_num'] = pd.Series([np.nan]*len(gaze))
    else:
        try: gaze['trial_num'] = gaze['trial_num'].astype('Int64')
        except: pass

    # Load response metadata at the trial level.
    trial_meta = pd.DataFrame(); response_df = None
    # Prefer response files that include `trial_type` (`all_responses` first).
    candidate_files = []
    # Collect existing candidate files.
    raw_candidates = [
        os.path.join(response_dir, f"all_responses_P-{pid}.csv"),  # Highest priority.
        os.path.join(response_dir, f"responses_P-{pid}.csv"),
        os.path.join(response_dir, f"response_P-{pid}.csv"),
    ]
    for fp in raw_candidates:
        if os.path.exists(fp):
            candidate_files.append(fp)
    # If no direct matches exist, try a wildcard fallback.
    if not candidate_files:
        glob_pattern = os.path.join(response_dir, f"*P-{pid}*.csv")
        for fp in glob.glob(glob_pattern):
            candidate_files.append(fp)
    # Read candidates and select the best one: prefer `trial_type`, then more columns.
    loaded = []
    for fp in candidate_files:
        try:
            tmp = pd.read_csv(fp)
            loaded.append((fp, tmp))
        except Exception as e:
            if verbose:
                print(f"⚠️ Skipping unreadable response file {os.path.basename(fp)}: {e}")
            continue
    if loaded:
        # Sort by whether `trial_type` exists, then by column count.
        def score(item):
            fp, dfc = item
            has_trial_type = int('trial_type' in dfc.columns)
            return (has_trial_type, len(dfc.columns))
        loaded.sort(key=score, reverse=True)
        response_df = loaded[0][1]
        chosen_fp = loaded[0][0]
        if verbose:
            print(f"📄 Selected response file: {os.path.basename(chosen_fp)} (columns={len(response_df.columns)}, trial_type={'✓' if 'trial_type' in response_df.columns else '×'})")
    else:
        if verbose:
            print("⚠️ No readable response file found")

    if response_df is not None and 'trial_num' in response_df.columns:
        # Use all rows directly because the file is already participant-specific.
        keep_cols = [c for c in ['trial_num','stimulus_id','condition','reasoning_type','trial_type'] if c in response_df.columns]
        if verbose and 'trial_type' in response_df.columns:
            print(f"✅ Found a trial_type column in the response file")
        elif verbose:
            print(f"⚠️ Response file has no trial_type column; available columns: {list(response_df.columns)[:10]}")
        trial_meta = (response_df[keep_cols]
                      .dropna(subset=['trial_num'])
                      .drop_duplicates('trial_num')
                      .copy())
        try: trial_meta['trial_num'] = trial_meta['trial_num'].astype('Int64')
        except: pass
    else:
        if verbose: print('⚠️ No valid trial-level response metadata found')

    # Fallback: use `presented_stimuli.csv` only to supplement `stimulus_id`.
    presented_path = os.path.join(cleaned_dir, '../tracking/presented_stimuli.csv')
    if (trial_meta.empty or 'stimulus_id' not in trial_meta.columns or trial_meta['stimulus_id'].isna().all()) and os.path.exists(presented_path):
        try:
            stim_df = pd.read_csv(presented_path)
            # If the file has no header, assign column names manually.
            if 'participant_id' not in stim_df.columns:
                stim_df = pd.read_csv(presented_path, header=None)
                stim_df.columns = ['participant_id','stimulus_id','timestamp','trial_num','complexity_steps','condition','reasoning_type','times_presented','trial_type_old']
            stim_df['participant_id'] = stim_df['participant_id'].astype(str).str.lower()
            pid_lc = pid.lower()
            stim_df = stim_df[stim_df['participant_id']==f'p-{pid_lc}']
            stim_df['trial_num'] = stim_df['trial_num'].astype('Int64')
            stim_meta = stim_df[['trial_num','stimulus_id','condition','reasoning_type']].drop_duplicates('trial_num')
            if not trial_meta.empty:
                trial_meta = pd.merge(trial_meta, stim_meta, on='trial_num', how='outer', suffixes=('','_stim'))
                for col in ['stimulus_id','condition','reasoning_type']:
                    if f"{col}_stim" in trial_meta.columns:
                        trial_meta[col] = trial_meta[col].combine_first(trial_meta[f"{col}_stim"])
                # 确保选择存在的列
                cols_to_keep = ['trial_num','stimulus_id','condition','reasoning_type']
                if 'trial_type' in trial_meta.columns:
                    cols_to_keep.append('trial_type')
                trial_meta = trial_meta[cols_to_keep]
            else:
                trial_meta = stim_meta
            if verbose: print('✅ stimulus_id was supplemented from presented_stimuli.csv')
        except Exception as e:
            if verbose: print(f'⚠️ Failed to read presented_stimuli.csv: {e}')

    # Merge trial metadata.
    if not trial_meta.empty and join_meta:
        before_cols = set(gaze.columns)
        if verbose and 'trial_type' in trial_meta.columns:
            print(f"✅ trial_meta contains trial_type; ready to merge")
        gaze = gaze.merge(trial_meta, on='trial_num', how='left', suffixes=('','_resp'))
        if verbose:
            added = [c for c in gaze.columns if c not in before_cols]
            if added:
                print(f"✅ Added trial-metadata columns: {added}")
            if 'trial_type' in gaze.columns:
                print(f"✅ trial_type was added to the gaze data")
            else:
                print(f"⚠️ trial_type was not added to the gaze data")
    else:
        if 'stimulus_id' not in gaze.columns:
            gaze['stimulus_id'] = np.nan

    if 'stimulus_id' in gaze.columns and 'stimuli_id' not in gaze.columns:
        gaze['stimuli_id'] = gaze['stimulus_id']

    # Ensure `trial_type` exists even if metadata is missing.
    if 'trial_type' not in gaze.columns:
        gaze['trial_type'] = np.nan

    # Simple AOI assignment based on `stimuli_y` and AOI boundaries.
    gaze['aoi'] = 'OOD'  # Default to Out Of Domain.
    
    if 'stimuli_y' in gaze.columns and 'trial_num' in gaze.columns:
        target_file = os.path.join(stimuli_dir, f"token_coordinates_P-{pid}.csv")
        
        if os.path.exists(target_file):
            # Read AOI data.
            aoi_df = pd.read_csv(target_file)
            aoi_data = aoi_df.dropna(subset=['aoi_left','aoi_right','aoi_top','aoi_bottom']).copy()
            
            if len(aoi_data) > 0:
                if verbose:
                    print(f"🎯 Loaded AOI-boundary data: {len(aoi_data)} AOI regions")
                
                # Assign AOIs trial by trial.
                for trial_num in gaze['trial_num'].dropna().unique():
                    # Get gaze points for the current trial.
                    trial_mask = (gaze['trial_num'] == trial_num)
                    
                    # Get AOI boundaries for the current trial.
                    trial_aois = aoi_data[aoi_data['trial_num'] == trial_num]
                    
                    if len(trial_aois) > 0:
                        if verbose:
                            print(f"  Trial {trial_num}: {trial_mask.sum()} gaze points, {len(trial_aois)} AOIs")
                        
                        # Check the boundary of each AOI.
                        for _, aoi_row in trial_aois.iterrows():
                            aoi_name = aoi_row['section']
                            aoi_top = aoi_row['aoi_top']
                            aoi_bottom = aoi_row['aoi_bottom']

                            # Add a shared boundary margin.
                            margin = 0.005  # Margin around the AOI boundary.
                            aoi_top_with_margin = aoi_top + margin
                            aoi_bottom_with_margin = aoi_bottom - margin

                            # Simple rule: bottom <= stimuli_y <= top, with a margin.
                            mask = (trial_mask &
                                   (gaze['aoi'] == 'OOD') &  # Only assign points that are still unassigned.
                                   (gaze['stimuli_y'] >= aoi_bottom_with_margin) &
                                   (gaze['stimuli_y'] <= aoi_top_with_margin))
                            
                            matched = mask.sum()
                            if matched > 0:
                                gaze.loc[mask, 'aoi'] = aoi_name
                                if verbose:
                                    print(f"    {aoi_name}: {matched} points (y: {aoi_bottom:.3f} to {aoi_top:.3f})")
                
                if verbose:
                    total_assigned = (gaze['aoi'] != 'OOD').sum()
                    print(f"✅ AOI assignment complete: {total_assigned}/{len(gaze)} points assigned")

                # Add AOI-boundary columns.
                gaze['aoi_left'] = np.nan
                gaze['aoi_right'] = np.nan
                gaze['aoi_top'] = np.nan
                gaze['aoi_bottom'] = np.nan
                
                # For each non-OOD gaze point, fill in the matching AOI boundaries.
                for trial_num in gaze['trial_num'].dropna().unique():
                    trial_mask = (gaze['trial_num'] == trial_num)
                    trial_aois = aoi_data[aoi_data['trial_num'] == trial_num]
                    
                    for _, aoi_row in trial_aois.iterrows():
                        aoi_name = aoi_row['section']
                        # Find gaze points assigned to this AOI.
                        aoi_mask = trial_mask & (gaze['aoi'] == aoi_name)
                        
                        if aoi_mask.sum() > 0:
                            gaze.loc[aoi_mask, 'aoi_left'] = aoi_row['aoi_left']
                            gaze.loc[aoi_mask, 'aoi_right'] = aoi_row['aoi_right']
                            gaze.loc[aoi_mask, 'aoi_top'] = aoi_row['aoi_top']
                            gaze.loc[aoi_mask, 'aoi_bottom'] = aoi_row['aoi_bottom']
                
                if verbose:
                    boundary_filled = gaze['aoi_left'].notna().sum()
                    print(f"✅ AOI boundary info added: {boundary_filled}/{len(gaze)} points have boundary data")

    # Quality summary.
    if verbose:
        total = len(gaze)
        trials_detected = gaze['trial_num'].nunique(dropna=True)
        missing_trial = gaze['trial_num'].isna().sum()
        stim_present = gaze['stimulus_id'].notna().sum() if 'stimulus_id' in gaze.columns else 0
        print(f"🧪 Participant {pid}: {total} gaze rows | trials={trials_detected} | missing trial_num {missing_trial} ({missing_trial/total*100:.1f}%)")
        if stim_present:
            print(f"   stimulus_id matched: {stim_present}/{total} ({stim_present/total*100:.1f}%)")
        else:
            print("   stimulus_id missing for all rows")

    # Save output.
    out_path = None
    if save:
        os.makedirs(cleaned_dir, exist_ok=True)
        out_path = os.path.join(cleaned_dir, f"integrated_gaze_P-{pid}.csv")
        gaze.to_csv(out_path, index=False)
        if verbose: print(f"💾 Saved integrated file: {out_path}")

    return gaze


def analyze_fixation_quality(integrated_data, verbose=True):
    """
    分析 Fixation 质量和持续时间分布
    
    参数:
        integrated_data: 集成后的gaze数据
        verbose: 详细输出
    
    返回:
        dict: 分析结果统计
    """
    if verbose:
        print("🔍 分析 Fixation 质量和持续时间分布...")

    results = {}
    
    if 'event_type' not in integrated_data.columns or 'device_time_stamp' not in integrated_data.columns:
        if verbose:
            print("❌ 缺少必需列: event_type 或 device_time_stamp")
        return results

    # 1. 分析 fixation 的连续片段
    fixation_data = integrated_data[integrated_data['event_type'] == 'fixation'].copy()
    
    if len(fixation_data) == 0:
        if verbose:
            print("❌ 没有检测到 fixation 数据")
        return results

    # 按时间排序
    fixation_data = fixation_data.sort_values('device_time_stamp')
    
    # 计算 fixation 的连续片段
    fixation_data['time_diff'] = fixation_data['device_time_stamp'].diff() / 1000  # 转换为毫秒
    
    # 如果时间间隔 > 100ms，认为是新的 fixation 片段
    new_fixation = (fixation_data['time_diff'] > 100) | (fixation_data['time_diff'].isna())
    fixation_data['fixation_id'] = new_fixation.cumsum()
    
    # 计算每个 fixation 片段的统计信息
    fixation_stats = fixation_data.groupby('fixation_id').agg({
        'device_time_stamp': ['count', 'min', 'max'],
        'trial_num': 'first'
    })
    
    # 计算持续时间
    fixation_stats['duration_ms'] = (fixation_stats[('device_time_stamp', 'max')] - 
                                    fixation_stats[('device_time_stamp', 'min')]) / 1000
    
    results['fixation_segments'] = len(fixation_stats)
    results['avg_duration'] = fixation_stats['duration_ms'].mean()
    results['median_duration'] = fixation_stats['duration_ms'].median()
    results['duration_range'] = (fixation_stats['duration_ms'].min(), fixation_stats['duration_ms'].max())
    
    if verbose:
        print(f"📈 检测到 {len(fixation_stats)} 个独立的 fixation 片段")
        print(f"   平均持续时间: {results['avg_duration']:.1f} ms")
        print(f"   中位数持续时间: {results['median_duration']:.1f} ms")
        print(f"   持续时间范围: {results['duration_range'][0]:.1f} - {results['duration_range'][1]:.1f} ms")
    
    # 合理 fixation 持续时间的标准 (150-600ms)
    reasonable_fixations = fixation_stats[(fixation_stats['duration_ms'] >= 150) & 
                                        (fixation_stats['duration_ms'] <= 600)]
    
    results['reasonable_fixation_ratio'] = len(reasonable_fixations) / len(fixation_stats)
    
    if verbose:
        print(f"   合理持续时间的 fixation (150-600ms): {len(reasonable_fixations)}/{len(fixation_stats)} ({results['reasonable_fixation_ratio']*100:.1f}%)")
    
    # 显示持续时间分布
    duration_bins = [0, 100, 150, 200, 300, 400, 500, 600, 1000, float('inf')]
    duration_labels = ['<100ms', '100-150ms', '150-200ms', '200-300ms', '300-400ms', 
                      '400-500ms', '500-600ms', '600-1000ms', '>1000ms']
    
    fixation_stats['duration_bin'] = pd.cut(fixation_stats['duration_ms'], 
                                           bins=duration_bins, labels=duration_labels, right=False)
    
    duration_dist = fixation_stats['duration_bin'].value_counts().sort_index()
    results['duration_distribution'] = duration_dist.to_dict()
    
    if verbose:
        print(f"\n📊 Fixation 持续时间分布:")
        for bin_name, count in duration_dist.items():
            pct = count / len(fixation_stats) * 100
            print(f"   {bin_name}: {count} 个 ({pct:.1f}%)")
    
    # 2. 分析速度分布
    if 'velocity' in integrated_data.columns:
        fixation_velocities = integrated_data[integrated_data['event_type'] == 'fixation']['velocity']
        saccade_velocities = integrated_data[integrated_data['event_type'] == 'saccade']['velocity']
        
        results['fixation_avg_velocity'] = fixation_velocities.mean()
        results['fixation_95p_velocity'] = fixation_velocities.quantile(0.95)
        results['saccade_avg_velocity'] = saccade_velocities.mean()
        results['saccade_5p_velocity'] = saccade_velocities.quantile(0.05)
        
        if verbose:
            print(f"\n🚀 速度分布分析:")
            print(f"   Fixation 平均速度: {results['fixation_avg_velocity']:.3f} pixel/ms")
            print(f"   Fixation 95%分位数速度: {results['fixation_95p_velocity']:.3f} pixel/ms")
            print(f"   Saccade 平均速度: {results['saccade_avg_velocity']:.3f} pixel/ms")
            print(f"   Saccade 5%分位数速度: {results['saccade_5p_velocity']:.3f} pixel/ms")
        
        # 检查分类质量
        wrong_fixations = (fixation_velocities > 1.0).sum()  # 速度 > 1.0 的 fixation
        wrong_saccades = (saccade_velocities < 0.1).sum()   # 速度 < 0.1 的 saccade
        
        results['wrong_fixation_ratio'] = wrong_fixations / len(fixation_velocities)
        results['wrong_saccade_ratio'] = wrong_saccades / len(saccade_velocities)
        
        if verbose:
            print(f"\n⚠️ 可能的分类错误:")
            print(f"   高速度 fixation (>1.0 pixel/ms): {wrong_fixations} 个 ({results['wrong_fixation_ratio']*100:.1f}%)")
            print(f"   低速度 saccade (<0.1 pixel/ms): {wrong_saccades} 个 ({results['wrong_saccade_ratio']*100:.1f}%)")

    return results


def integrate_participant_with_trial_type(participant_id,
                                          gaze_dir,
                                          response_dir,
                                          cleaned_dir,
                                          stimuli_dir,
                                          source='raw',
                                          save=True,
                                          verbose=True,
                                          force_reclean=False,
                                          prefer='all',
                                          extra_meta_cols=None):
    """集成参与者数据 + 追加 trial_type 元数据（不修改原 integrate_participant 实现）。

    用法：调用本函数即可生成含 trial_type 的 integrated_gaze_P-{pid}.csv。

    流程：
      1. 调用原始 integrate_participant(join_meta=False) 仅做 gaze 清洗 + trial_num 推断 + AOI 分配
      2. 独立加载 response 文件（优先 all_responses_P-*.csv，其次 responses_P-*.csv, response_P-*.csv）
      3. 仅提取 ['trial_num','trial_type'] (以及用户指定的 extra_meta_cols) 进行左连接
      4. 若 trial_type 缺失则创建空列
      5. 覆盖保存原 integrated_gaze_P-{pid}.csv

    参数:
        participant_id: 参与者 ID
        gaze_dir / response_dir / cleaned_dir / stimuli_dir: 路径
        source: 'raw' | 'cleaned'
        save: 是否保存
        verbose: 输出开关
        force_reclean: 传递给 integrate_participant
        prefer: 'all' 表示优先 all_responses 文件；否则按默认顺序
        extra_meta_cols: 额外希望合并的列列表 (例如 ['condition','reasoning_type'])

    返回:
        DataFrame (含 trial_type)
    """
    pid = str(participant_id)
    if extra_meta_cols is None:
        extra_meta_cols = []

    # 1) 先调用原始函数（关闭元数据合并）
    base_df = integrate_participant(
        participant_id=pid,
        gaze_dir=gaze_dir,
        response_dir=response_dir,
        cleaned_dir=cleaned_dir,
        stimuli_dir=stimuli_dir,
        source=source,
        save=False,            # Do not save yet; save after merging `trial_type`.
        save_cleaned=False,
        verbose=verbose,
        join_meta=False,       # Disable the internal response merge in the base function.
        force_reclean=force_reclean
    )

    if verbose:
        print(f"🔄 integrate_participant 基础数据生成: {len(base_df)} rows")

    # 2) Find the response file.
    candidates = []
    ordered = [
        f"all_responses_P-{pid}.csv",
        f"responses_P-{pid}.csv",
        f"response_P-{pid}.csv",
    ]
    # If `prefer != 'all'`, do not force `all_responses` to the front.
    if prefer != 'all':
        ordered = ordered[1:] + ordered[:1]

    for name in ordered:
        fp = os.path.join(response_dir, name)
        if os.path.exists(fp):
            candidates.append(fp)
    if not candidates:
        # Fallback wildcard search for slight naming differences.
        pattern = os.path.join(response_dir, f"*P-{pid}*.csv")
        for fp in glob.glob(pattern):
            candidates.append(fp)

    response_df = None; chosen = None
    for fp in candidates:
        try:
            tmp = pd.read_csv(fp)
            response_df = tmp; chosen = fp
            break
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to read {os.path.basename(fp)}: {e}")
            continue

    if response_df is None:
        if verbose:
            print("⚠️ No usable response file found; an empty trial_type column will be created.")
        base_df['trial_type'] = np.nan
    else:
        if verbose:
            print(f"📄 使用 response 文件: {os.path.basename(chosen)} (columns={len(response_df.columns)})")
        # 3) Select the columns to merge.
        cols = ['trial_num']
        if 'trial_type' in response_df.columns:
            cols.append('trial_type')
        else:
            if verbose:
                print("⚠️ The response file has no trial_type column; it will be filled with NaN.")
        # Add any extra columns requested by the caller.
        for c in extra_meta_cols:
            if c in response_df.columns and c not in cols:
                cols.append(c)
        meta = response_df[cols].dropna(subset=['trial_num']).copy()
        try:
            meta['trial_num'] = meta['trial_num'].astype('Int64')
        except: pass

        before_cols = set(base_df.columns)
        base_df = base_df.merge(meta, on='trial_num', how='left', suffixes=('', '_resp2'))

        # 4) Ensure `trial_type` exists.
        if 'trial_type' not in base_df.columns:
            base_df['trial_type'] = np.nan
        if verbose:
            added = [c for c in base_df.columns if c not in before_cols]
            print(f"✅ Merge complete; added columns: {added if added else 'none'}; trial_type present: {'trial_type' in base_df.columns}")

    # 5) Save the merged file.
    if save:
        os.makedirs(cleaned_dir, exist_ok=True)
        out_path = os.path.join(cleaned_dir, f"integrated_gaze_P-{pid}.csv")
        base_df.to_csv(out_path, index=False)
        if verbose:
            print(f"💾 Saved integrated file with trial_type: {out_path}")

    return base_df


def print_participant_aoi_bounds(participant_id, stimuli_dir):
    """
    Read the participant-specific `token_coordinates` CSV and print AOI boxes for
    all trials.

    Logic:
    1. Read `token_coordinates_P-{pid}.csv`
    2. Remove rows where all AOI-boundary columns are NaN
    3. Treat the remaining rows as AOI-boundary records, with `section` as the AOI name
    4. Print AOI box coordinates grouped by trial
    """
    pid = str(participant_id).strip()
    
    # Build the file path.
    target_file = os.path.join(stimuli_dir, f"token_coordinates_P-{pid}.csv")
    
    if not os.path.exists(target_file):
        print(f"❌ File not found: {target_file}")
        return
    
    print(f"📄 Reading file: {os.path.basename(target_file)}")
    
    try:
        df = pd.read_csv(target_file)
        print(f"✅ Rows: {len(df)}, columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return
    
    # Check required columns.
    required_cols = ['trial_num', 'stimulus_id', 'section', 'aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"📂 Available columns: {list(df.columns)}")
        return
    
    # Keep only rows with valid AOI boundaries.
    print(f"\n🔍 Total rows before filtering: {len(df)}")
    aoi_data = df.dropna(subset=['aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']).copy()
    print(f"🔍 Rows after dropping NaN AOI boundaries: {len(aoi_data)}")
    
    if len(aoi_data) == 0:
        print("❌ No valid AOI-boundary data found")
        return
    
    # Convert coordinates to numeric values.
    for col in ['aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']:
        aoi_data[col] = pd.to_numeric(aoi_data[col], errors='coerce')
    
    # Compute width and height.
    aoi_data['width'] = aoi_data['aoi_right'] - aoi_data['aoi_left']
    aoi_data['height'] = aoi_data['aoi_bottom'] - aoi_data['aoi_top']
    
    print(f"\n📊 AOI types found: {sorted(aoi_data['section'].unique())}")
    
    # Print results grouped by trial.
    trials = sorted(aoi_data['trial_num'].unique())
    print(f"\n🎯 AOI box coordinates for all trials ({len(trials)} trials total):\n")
    
    for trial_num in trials:
        trial_data = aoi_data[aoi_data['trial_num'] == trial_num]
        stimulus_id = trial_data['stimulus_id'].iloc[0] if len(trial_data) > 0 else 'unknown'
        condition = trial_data['condition'].iloc[0] if 'condition' in trial_data.columns else 'unknown'
        
        print(f"Trial {trial_num} | Stimulus: {stimulus_id} | Condition: {condition}")
        
        for _, row in trial_data.iterrows():
            aoi_name = row['section']
            left, right, top, bottom = row['aoi_left'], row['aoi_right'], row['aoi_top'], row['aoi_bottom']
            width, height = row['width'], row['height']
            
            print(f"  ✅ {aoi_name}: left={left:.3f}, right={right:.3f}, top={top:.3f}, bottom={bottom:.3f} "
                  f"(width={width:.3f}, height={height:.3f})")
        print()  # Blank line between trials.
    
    # Summary statistics.
    print("📈 AOI summary:")
    aoi_summary = aoi_data.groupby('section').agg({
        'trial_num': 'count',
        'width': ['mean', 'std'],
        'height': ['mean', 'std']
    }).round(3)
    
    for aoi_type in aoi_summary.index:
        count = aoi_summary.loc[aoi_type, ('trial_num', 'count')]
        avg_width = aoi_summary.loc[aoi_type, ('width', 'mean')]
        avg_height = aoi_summary.loc[aoi_type, ('height', 'mean')]
        print(f"  {aoi_type}: present in {count} trials, mean size {avg_width:.3f} × {avg_height:.3f}")


# def print_participant_aoi_bounds(participant_id, stimuli_dir,
#                                  clamp_bounds=None):
#     """
#     读取指定参与者的 token_coordinates CSV，打印所有 trial 的 AOI 区域框坐标（合并/调整后）。

#     变更点：
#     1) 将 claim 与 evidence 合并为 claim_evidence（使用两者最小外接矩形）
#     2) 调整边界：
#        - reasoning: top += 0.025, bottom -= 0.025
#        - answer:    top += 0.015, bottom -= 0.015
#        - response:  top += 0.015, bottom 不变
#     3) 可选 clamp 到范围（如 [0,1]），通过 clamp_bounds=(0,1) 开启

#     参数：
#       participant_id: int | str
#       stimuli_dir: token_coordinates_* 文件所在目录
#       clamp_bounds: None 或 (lo, hi)，若给出则会把 left/right/top/bottom 裁剪到此范围
#     """
#     pid = str(participant_id).strip()

#     # 构造文件路径
#     target_file = os.path.join(stimuli_dir, f"token_coordinates_P-{pid}.csv")

#     if not os.path.exists(target_file):
#         print(f"❌ 未找到文件: {target_file}")
#         return

#     print(f"📄 读取文件: {os.path.basename(target_file)}")

#     try:
#         df = pd.read_csv(target_file)
#         print(f"✅ 数据行数: {len(df)}, 列数: {len(df.columns)}")
#     except Exception as e:
#         print(f"❌ 读取失败: {e}")
#         return

#     # 必需列检查
#     required_cols = ['trial_num', 'stimulus_id', 'section', 'aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#         print(f"❌ 缺少必需列: {missing_cols}")
#         print(f"📂 可用列: {list(df.columns)}")
#         return

#     # 过滤掉没有 AOI 边界的行
#     print(f"\n🔍 过滤前总行数: {len(df)}")
#     aoi_data = df.dropna(subset=['aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']).copy()
#     print(f"🔍 过滤掉 AOI 边界为 NaN 的行后: {len(aoi_data)} 行")

#     if len(aoi_data) == 0:
#         print("❌ 没有找到有效的 AOI 边界数据")
#         return

#     # 统一 section 命名为不带后缀的简名
#     def _norm_section(s):
#         s = str(s).strip().lower()
#         s = s.replace('_aoi', '')
#         return s  # 期望：claim / evidence / reasoning / answer / response

#     aoi_data['section_norm'] = aoi_data['section'].map(_norm_section)

#     # 转数值
#     for col in ['aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']:
#         aoi_data[col] = pd.to_numeric(aoi_data[col], errors='coerce')

#     # —— 合并 claim + evidence → claim_evidence（按 trial 合并，取外接矩形）——
#     merged_rows = []
#     for trial_num, tdf in aoi_data.groupby('trial_num'):
#         # 先保留非 claim/evidence 的 AOI
#         keep_mask = ~tdf['section_norm'].isin(['claim', 'evidence'])
#         keep_df = tdf[keep_mask].copy()

#         # 合并 claim / evidence
#         ce_df = tdf[tdf['section_norm'].isin(['claim', 'evidence'])]
#         if len(ce_df) > 0:
#             new_row = {
#                 'trial_num': trial_num,
#                 'stimulus_id': ce_df['stimulus_id'].iloc[0] if 'stimulus_id' in ce_df.columns else 'unknown',
#                 'section': 'claim_evidence',         # 输出列
#                 'section_norm': 'claim_evidence',    # 规范名
#                 'aoi_left':  float(ce_df['aoi_left'].min()),
#                 'aoi_right': float(ce_df['aoi_right'].max()),
#                 'aoi_top':   float(ce_df['aoi_top'].min()),
#                 'aoi_bottom':float(ce_df['aoi_bottom'].max()),
#             }
#             # 如果原表里有 condition，也带上一个（取首个）
#             if 'condition' in ce_df.columns:
#                 new_row['condition'] = ce_df['condition'].iloc[0]
#             keep_df = pd.concat([keep_df, pd.DataFrame([new_row])], ignore_index=True)

#         merged_rows.append(keep_df)

#     aoi_merged = pd.concat(merged_rows, ignore_index=True)

#     # —— 按规则扩张 AOI 边界（数值直接按你的加减）——
#     def _adjust_bounds(row):
#         sec = row['section_norm']
#         top = row['aoi_top']
#         bottom = row['aoi_bottom']

#         if sec == 'reasoning':
#             top += 0.025
#             bottom -= 0.025
#         elif sec == 'answer':
#             top += 0.015
#             bottom -= 0.015
#         elif sec == 'response':
#             top += 0.015
#             # bottom 不变
#         # claim_evidence 不变

#         row['aoi_top'] = top
#         row['aoi_bottom'] = bottom
#         return row

#     aoi_adj = aoi_merged.apply(_adjust_bounds, axis=1)

#     # 可选：裁剪边界到指定范围（如 [0,1]）
#     if clamp_bounds is not None and isinstance(clamp_bounds, (tuple, list)) and len(clamp_bounds) == 2:
#         lo, hi = clamp_bounds
#         for col in ['aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']:
#             aoi_adj[col] = aoi_adj[col].clip(lower=lo, upper=hi)

#     # 重新计算宽高
#     aoi_adj['width']  = aoi_adj['aoi_right'] - aoi_adj['aoi_left']
#     aoi_adj['height'] = aoi_adj['aoi_bottom'] - aoi_adj['aoi_top']

#     # 打印发现的 AOI 类型
#     print(f"\n📊 发现的 AOI 类型（合并/调整后）: {sorted(aoi_adj['section'].unique())}")

#     # 按 trial 分组输出
#     trials = sorted(aoi_adj['trial_num'].unique())
#     print(f"\n🎯 所有 Trial 的 AOI 边界框坐标 (共 {len(trials)} 个 trials):\n")

#     for trial_num in trials:
#         trial_data = aoi_adj[aoi_adj['trial_num'] == trial_num]
#         stimulus_id = trial_data['stimulus_id'].iloc[0] if len(trial_data) > 0 else 'unknown'
#         condition = trial_data['condition'].iloc[0] if 'condition' in trial_data.columns else 'unknown'

#         print(f"Trial {trial_num} | Stimulus: {stimulus_id} | Condition: {condition}")
#         for _, row in trial_data.iterrows():
#             aoi_name = row['section']
#             left, right, top, bottom = row['aoi_left'], row['aoi_right'], row['aoi_top'], row['aoi_bottom']
#             width, height = row['width'], row['height']
#             print(f"  ✅ {aoi_name}: left={left:.3f}, right={right:.3f}, top={top:.3f}, bottom={bottom:.3f} "
#                   f"(width={width:.3f}, height={height:.3f})")
#         print()

#     # 统计摘要
#     print("📈 AOI 统计摘要（合并/调整后）:")
#     aoi_summary = aoi_adj.groupby('section').agg({
#         'trial_num': 'count',
#         'width': ['mean', 'std'],
#         'height': ['mean', 'std']
#     }).round(3)

#     for aoi_type in aoi_summary.index:
#         count = aoi_summary.loc[aoi_type, ('trial_num', 'count')]
#         avg_width = aoi_summary.loc[aoi_type, ('width', 'mean')]
#         avg_height = aoi_summary.loc[aoi_type, ('height', 'mean')]
#         print(f"  {aoi_type}: 出现在 {count} 个 trials, 平均尺寸 {avg_width:.3f} × {avg_height:.3f}")

# ===== Batch-processing helpers =====

def detect_available_participants(gaze_path):
    """Detect participants with available gaze files."""
    if not os.path.exists(gaze_path):
        print(f"❌ Gaze directory does not exist: {gaze_path}")
        return []
    
    gaze_files = glob.glob(os.path.join(gaze_path, "gaze_data_list_P-*.csv"))
    participants = []
    
    for file in gaze_files:
        filename = os.path.basename(file)
        if filename.startswith("gaze_data_list_P-") and filename.endswith(".csv"):
            pid = filename.replace("gaze_data_list_P-", "").replace(".csv", "")
            participants.append(pid)
    
    participants = sorted(set(participants))
    return participants


def process_individual_participants(base_path, gaze_dir="gaze", response_dir="responses", 
                                  output_dir="icassp_cleaned_gaze", stimuli_dir="stimuli_data",
                                  participant_list=None, force_reprocess=False, verbose=True):
    """
    Step 1: process each participant and create individual `integrated_gaze_P-*.csv` files.
    
    Parameters:
    - base_path: Root data directory
    - gaze_dir: Gaze-data folder name
    - response_dir: Response-data folder name
    - output_dir: Output folder name
    - stimuli_dir: Stimuli-data folder name (used for AOI boundaries)
    - participant_list: Participants to process; `None` auto-detects all
    - force_reprocess: Whether to reprocess existing files
    - verbose: Whether to print detailed progress
    
    Returns:
    - summary: Processing summary
    """
    
    gaze_path = os.path.join(base_path, gaze_dir)
    response_path = os.path.join(base_path, response_dir)
    output_path = os.path.join(base_path, output_dir)
    stimuli_path = os.path.join(base_path, stimuli_dir)
    
    # Create the output directory.
    os.makedirs(output_path, exist_ok=True)
    
    # Auto-detect participant IDs.
    if participant_list is None:
        participant_list = detect_available_participants(gaze_path)
    
    if verbose:
        print(f"📊 第一步：处理 {len(participant_list)} 个参与者的individual数据")
        print(f"📁 输出目录: {output_path}")
        print(f"🔧 参与者列表: {participant_list}")
    
    # Collect processing results.
    summary = {
        'total_participants': len(participant_list),
        'successful_participants': 0,
        'failed_participants': [],
        'skipped_participants': [],
        'participants_summary': {},
        'total_gaze_points': 0,
        'total_trials': 0
    }
    
    # Process participants one by one.
    for i, pid in enumerate(participant_list):
        try:
            # 检查输出文件是否已存在
            integrated_file = os.path.join(output_path, f"integrated_gaze_P-{pid}.csv")
            
            if os.path.exists(integrated_file) and not force_reprocess:
                if verbose:
                    print(f"⏭️ 跳过参与者 {pid} ({i+1}/{len(participant_list)}) - 文件已存在")
                summary['skipped_participants'].append(pid)
                continue
            
            if verbose:
                print(f"🔄 处理参与者 {pid} ({i+1}/{len(participant_list)})...")
            
            # 使用已有的integrate_participant函数
            participant_data = integrate_participant(
                participant_id=pid,
                gaze_dir=gaze_path,
                response_dir=response_path,
                cleaned_dir=output_path,
                stimuli_dir=stimuli_path,
                source='raw',
                save=True,  # 强制保存individual文件
                save_cleaned=False,
                verbose=False,  # 减少单个参与者的输出
                join_meta=True,
                force_reclean=False
            )
            
            if participant_data is not None and len(participant_data) > 0:
                summary['successful_participants'] += 1
                
                # 收集参与者统计信息
                trials_count = participant_data['trial_num'].nunique(dropna=True)
                event_counts = participant_data['event_type'].value_counts().to_dict()
                fixation_pct = event_counts.get('fixation', 0) / len(participant_data) * 100
                
                summary['participants_summary'][pid] = {
                    'gaze_points': len(participant_data),
                    'trials': trials_count,
                    'fixation_percentage': fixation_pct,
                    'event_counts': event_counts,
                    'file_path': integrated_file
                }
                
                summary['total_gaze_points'] += len(participant_data)
                summary['total_trials'] += trials_count
                
                if verbose:
                    print(f"   ✅ 成功: {len(participant_data)} gaze points, {trials_count} trials, "
                          f"{fixation_pct:.1f}% fixation")
                    print(f"   💾 已保存: {os.path.basename(integrated_file)}")
            else:
                summary['failed_participants'].append(pid)
                if verbose:
                    print(f"   ❌ 失败: 无数据")
                
        except Exception as e:
            summary['failed_participants'].append(pid)
            if verbose:
                print(f"   ❌ 错误: {str(e)}")
    
    # 打印第一步摘要
    if verbose:
        print(f"\n📈 第一步处理摘要:")
        print(f"   总参与者数: {summary['total_participants']}")
        print(f"   成功处理: {summary['successful_participants']}")
        print(f"   跳过处理: {len(summary['skipped_participants'])}")
        print(f"   处理失败: {len(summary['failed_participants'])}")
        if summary['failed_participants']:
            print(f"   失败列表: {summary['failed_participants']}")
        print(f"   总gaze点数: {summary['total_gaze_points']:,}")
        print(f"   总trial数: {summary['total_trials']}")
    
    return summary


def merge_all_participants(base_path, output_dir="icassp_cleaned_gaze", 
                          output_filename="all_participants_integrated_gaze.csv",
                          participant_list=None, verbose=True):
    """
    第二步：读取所有individual integrated_gaze_P-*.csv文件并合并
    
    Parameters:
    - base_path: 数据根目录
    - output_dir: integrated文件所在目录
    - output_filename: 合并后的文件名
    - participant_list: 指定要合并的参与者列表，None则自动检测所有文件
    - verbose: 是否显示详细进度
    
    Returns:
    - merged_df: 合并后的DataFrame
    - merge_summary: 合并结果摘要
    """
    
    output_path = os.path.join(base_path, output_dir)
    
    if verbose:
        print(f"\n📊 第二步：合并所有participant的integrated数据")
        print(f"📁 输入目录: {output_path}")
    
    # 查找所有integrated文件
    if participant_list is None:
        integrated_files = glob.glob(os.path.join(output_path, "integrated_gaze_P-*.csv"))
        participant_list = []
        for file in integrated_files:
            filename = os.path.basename(file)
            if filename.startswith("integrated_gaze_P-") and filename.endswith(".csv"):
                pid = filename.replace("integrated_gaze_P-", "").replace(".csv", "")
                participant_list.append(pid)
        participant_list = sorted(set(participant_list))
    
    if verbose:
        print(f"🔍 找到 {len(participant_list)} 个participant文件")
        print(f"👥 参与者列表: {participant_list}")
    
    # 合并统计
    merge_summary = {
        'total_files': len(participant_list),
        'successful_files': 0,
        'failed_files': [],
        'total_gaze_points': 0,
        'total_trials': 0,
        'participants_summary': {}
    }
    
    # 读取并合并所有文件
    dataframes = []
    
    for pid in participant_list:
        try:
            file_path = os.path.join(output_path, f"integrated_gaze_P-{pid}.csv")
            
            if not os.path.exists(file_path):
                merge_summary['failed_files'].append(pid)
                if verbose:
                    print(f"   ❌ 文件不存在: integrated_gaze_P-{pid}.csv")
                continue
            
            if verbose:
                print(f"📖 读取参与者 {pid} 的数据...")
            
            df = pd.read_csv(file_path)
            
            if len(df) > 0:
                dataframes.append(df)
                merge_summary['successful_files'] += 1
                merge_summary['total_gaze_points'] += len(df)
                
                trials_count = df['trial_num'].nunique(dropna=True)
                merge_summary['total_trials'] += trials_count
                
                event_counts = df['event_type'].value_counts().to_dict()
                fixation_pct = event_counts.get('fixation', 0) / len(df) * 100
                
                merge_summary['participants_summary'][pid] = {
                    'gaze_points': len(df),
                    'trials': trials_count,
                    'fixation_percentage': fixation_pct
                }
                
                if verbose:
                    print(f"   ✅ 成功读取: {len(df)} gaze points, {trials_count} trials")
            else:
                merge_summary['failed_files'].append(pid)
                if verbose:
                    print(f"   ❌ 空文件: integrated_gaze_P-{pid}.csv")
                
        except Exception as e:
            merge_summary['failed_files'].append(pid)
            if verbose:
                print(f"   ❌ 读取错误 {pid}: {str(e)}")
    
    # 执行合并
    if dataframes:
        if verbose:
            print(f"\n🔗 合并 {len(dataframes)} 个文件...")
        
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # 保存合并后的文件
        output_file = os.path.join(output_path, output_filename)
        merged_df.to_csv(output_file, index=False)
        
        if verbose:
            print(f"💾 已保存合并文件: {output_file}")
            print(f"📊 合并后数据规模: {len(merged_df)} 行 × {len(merged_df.columns)} 列")
            
            # 整体统计
            overall_events = merged_df['event_type'].value_counts()
            total_points = len(merged_df)
            print(f"\n📊 整体事件分布:")
            for event, count in overall_events.items():
                pct = count / total_points * 100
                print(f"   {event}: {count:,} ({pct:.1f}%)")
    else:
        merged_df = pd.DataFrame()
        if verbose:
            print("❌ 没有可合并的数据")
    
    # 打印合并摘要
    if verbose:
        print(f"\n📈 第二步合并摘要:")
        print(f"   总文件数: {merge_summary['total_files']}")
        print(f"   成功合并: {merge_summary['successful_files']}")
        print(f"   合并失败: {len(merge_summary['failed_files'])}")
        if merge_summary['failed_files']:
            print(f"   失败列表: {merge_summary['failed_files']}")
        print(f"   总gaze点数: {merge_summary['total_gaze_points']:,}")
        print(f"   总trial数: {merge_summary['total_trials']}")
    
    return merged_df, merge_summary


def process_all_participants_two_step(base_path, participant_list=None, force_reprocess=False, 
                                     output_filename="all_participants_integrated_gaze.csv", 
                                     stimuli_dir="stimuli_data", verbose=True):
    """
    完整的两步骤处理流程
    
    Parameters:
    - base_path: 数据根目录
    - participant_list: 要处理的参与者列表，None则自动检测
    - force_reprocess: 是否强制重新处理已存在的individual文件
    - output_filename: 最终合并文件名
    - stimuli_dir: stimuli数据文件夹名 (用于AOI边界)
    - verbose: 是否显示详细进度
    
    Returns:
    - merged_df: 合并后的DataFrame
    - process_summary: 处理摘要
    - merge_summary: 合并摘要
    """
    
    if verbose:
        print("🚀 开始两步骤批量处理流程...")
    
    # 第一步：处理individual participants
    process_summary = process_individual_participants(
        base_path=base_path,
        participant_list=participant_list,
        force_reprocess=force_reprocess,
        stimuli_dir=stimuli_dir,
        verbose=verbose
    )
    
    # 第二步：合并所有文件
    merged_df, merge_summary = merge_all_participants(
        base_path=base_path,
        participant_list=participant_list,
        output_filename=output_filename,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n🎉 两步骤处理完成！")
        print(f"📁 Individual文件位置: {os.path.join(base_path, 'icassp_cleaned_gaze')}")
        print(f"📄 合并文件: {output_filename}")
    
    return merged_df, process_summary, merge_summary

