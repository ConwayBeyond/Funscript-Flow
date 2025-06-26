"""Mathematical utilities and signal processing functions"""

import numpy as np


def sine_fit(data, error_threshold=5000.0, gain=2.0, min_points=3, max_points=30):
    """
    Fits half-wave sine segments (center=50) to `data` by testing candidate endpoints
    from min_points to max_points ahead. After segmentation, if two consecutive segments
    have the same sign, we split them with an inserted corrective half-wave (with inverted amplitude)
    to help catch missed alternations.

    Returns the fitted array.
    """
    n = len(data)
    segments = []  # each segment is a dict with {'start', 'end', 'A'}
    start = 0

    # --- First pass: Segment the data ---
    while start < n - 1:
        best_err = np.inf
        best_end = None
        best_A = 0.0

        for seg_len in range(min_points, max_points + 1):
            end = start + seg_len
            if end >= n:
                break

            T = seg_len  # segment length (points between endpoints)
            x = np.arange(T + 1)
            model = np.sin(np.pi * x / T)
            segment = data[start:end + 1]
            denom = np.sum(model**2)
            if denom == 0:
                continue
            # Linear LS solution for amplitude A.
            A = np.sum(model * (segment - 50)) / denom
            fit = 50 + A * model
            err = np.sqrt(np.mean((segment - fit) ** 2))

            if err < best_err:
                best_err = err
                best_end = end
                best_A = A

        if best_end is None:
            break

        # Error correction: if error too high, flatten the segment.
        if best_err > error_threshold:
            best_A = 0.0
        # Boost low amplitude segments (because sometimes they're just shy).
        #best_A = np.sign(best_A) * (abs(best_A) ** (1.0 / gain))

        segments.append({'start': start, 'end': best_end, 'A': best_A})
        start = best_end

    # --- Second pass: Correction for consecutive segments with the same sign ---
    corrected_segments = []
    i = 0
    while i < len(segments):
        # If the next segment exists and both segments have nonzero, same-signed amplitude...
        if (i < len(segments) - 1 and segments[i]['A'] != 0 and segments[i+1]['A'] != 0 and
            np.sign(segments[i]['A']) == np.sign(segments[i+1]['A'])):
            combined_start = segments[i]['start']
            combined_end = segments[i+1]['end']
            if (combined_end - combined_start) >= min_points*2:
                L = combined_end - combined_start
                # Split the combined region into three parts.
                mid1 = combined_start + L // 3
                mid2 = combined_start + 2 * L // 3

                # Re-fit first sub-segment.
                T1 = mid1 - combined_start
                if T1 < 2:
                    T1 = 2
                    mid1 = combined_start + T1
                x1 = np.arange(T1 + 1)
                model1 = np.sin(np.pi * x1 / T1)
                seg1 = data[combined_start:mid1 + 1]
                denom1 = np.sum(model1 ** 2)
                A1 = np.sum(model1 * (seg1 - 50)) / denom1 if denom1 != 0 else 0

                # Re-fit third sub-segment.
                T3 = combined_end - mid2
                if T3 < 2:
                    T3 = 2
                    mid2 = combined_end - T3
                x3 = np.arange(T3 + 1)
                model3 = np.sin(np.pi * x3 / T3)
                seg3 = data[mid2:combined_end + 1]
                denom3 = np.sum(model3 ** 2)
                A3 = np.sum(model3 * (seg3 - 50)) / denom3 if denom3 != 0 else 0

                # Corrective (middle) segment: force amplitude opposite in sign.
                A2 = -np.sign(segments[i]['A']) * (0.5 * (abs(A1) + abs(A3)))

                corrected_segments.append({'start': combined_start, 'end': mid1, 'A': A1})
                corrected_segments.append({'start': mid1, 'end': mid2, 'A': A2})
                corrected_segments.append({'start': mid2, 'end': combined_end, 'A': A3})
                i += 2  # skip the next segment; we've merged it
                continue
            else:
                #Comvine them into one segment
                combined_A = segments[i]['A'] + segments[i+1]['A']
                combined_start = segments[i]['start']
                combined_end = segments[i+1]['end']
                corrected_segments.append({'start': combined_start, 'end': combined_end, 'A': combined_A})
                i += 2
                continue

        corrected_segments.append(segments[i])
        i += 1

    # --- Third pass: Detect and fix missed periods ---
    final_segments = []
    for j in range(len(corrected_segments)):
        if j > 0 and j < len(corrected_segments) - 1:
            
            prev_L = corrected_segments[j-1]['end'] - corrected_segments[j-1]['start']
            curr_L = corrected_segments[j]['end'] - corrected_segments[j]['start']
            next_L = corrected_segments[j+1]['end'] - corrected_segments[j+1]['start']
            
            if curr_L > prev_L + next_L:
                # Split into a number of segments depending on the calculated number of missed periods
                missed_periods = round(curr_L / (prev_L + next_L))

                segment_splits = np.linspace(corrected_segments[j]['start'], corrected_segments[j]['end'], 2*missed_periods + 1, dtype=int)
                invert = False
                for split_idx in range(len(segment_splits) - 1):
                    split_segment = {'start': segment_splits[split_idx], 'end': segment_splits[split_idx + 1], 'A': corrected_segments[j]['A'] * (-1 if invert else 1)}
                    invert = not invert
                    final_segments.append(split_segment)
                continue
        final_segments.append(corrected_segments[j])
    #plot the rolling variance of segment lengths, with outliers flagged
    segment_lengths = [seg['end'] - seg['start'] for seg in final_segments]
    # Calculate the rolling variance of segment lengths in a window of 5 segments
    rolling_var = np.full(len(segment_lengths), np.nan)
    for i in range(2, len(segment_lengths) - 2):
        rolling_var[i] = np.var(segment_lengths[i-2:i+3])
    # Flag outliers (variance > 1.5 * median variance)
    var_threshold = 1.5 * np.nanmedian(rolling_var)
    for i in range(len(rolling_var)):
        if rolling_var[i] > var_threshold:
            final_segments[i]['outlier'] = True
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(rolling_var, label='Variance', marker='o')
    # plt.axhline(y=np.mean(rolling_var), color='r', linestyle='--', label='Mean Segment Length')
    # plt.axhline(y=var_threshold, color='g', linestyle=':', label='Variance Threshold', xmin=0, xmax=len(segment_lengths)-1)
    # plt.title('Rolling Variance of Segment Lengths with Outliers Flagged')
    # plt.xlabel('Segment Index')
    # plt.ylabel('Length')
    # plt.legend()
    # plt.show()

    # --- Build the fitted curve from the corrected segments ---
    fitted = np.full(n, 50.0)
    for seg in final_segments:
        s, e = seg['start'], seg['end']
        T = e - s
        if T < 1:
            continue
        x_seg = np.arange(T + 1)
        fitted[s:e + 1] = 50 + seg['A'] * np.sin(np.pi * x_seg / T)

    return fitted