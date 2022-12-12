import numpy as np


def detect_significant_trends(points, x_scale=1.0):
    points = np.array(points)
    points_dif = points[1:] - points[:-1]
    all_best = {}

    min_trend_size = int(np.floor((len(points) // 40) // x_scale))

    for i in range(len(points_dif)):
        # cp = points_dif[i]
        best = (np.Inf, 0, -1, -1, np.Inf)
        for j in range(i+min_trend_size, len(points_dif)):
            # next_p = points_dif[j]
            point_range = points_dif[i:(j+1)]
            slope = np.mean(point_range) / x_scale
            # slope = (points[j+1] - points[i]) / x_scale
            std = np.std(point_range)
            # corr = np.corrcoef(point_range)
            interval_length = (j-i+1) * x_scale
            score = std/(interval_length ** 1)

            if score < best[-1]:
                best = (std, slope, i, j, score)
        # print('best in {}: ({}, {}) avg: {} / std: {}'.format(i, i, best[3], best[1], best[0]))
        if best[0] != np.Inf:
            if not all_best.get(best[3]):
                all_best.setdefault(best[3], [best])
            else:
                all_best[best[3]].append(best)

    candidates = []

    for end in all_best:
        points = all_best[end]
        interval = sorted(points, key=lambda x: x[-1])[0]
        print("Interval at ({}, {}) with a slope of {} and a score of {}".format(
            interval[2], interval[3]+1, interval[1], interval[-1]))
        candidates.append({'slope': interval[1],
                           'interval': (interval[2], interval[3]+1),
                           'metric': interval[-1],
                           'std': interval[0]})

    candidates = np.array(sorted(candidates, key=lambda x: x['interval'][0]))

    candidates_removed = np.full(candidates.size, False)

    # Detecting candidates inside each other
    sz = len(candidates)
    for i in range(sz):
        if candidates_removed[i] == True:
            continue

        ci = candidates[i]
        i0, i1 = ci['interval']
        for j in range(i+1, sz):
            cj = candidates[j]
            j0, j1 = cj['interval']
            # print("({},{}), ({},{})".format(i0, i1, j0, j1))
            if i0 <= j0 and i1 >= j1 and ci['metric'] < cj['metric']:
                candidates_removed[j] = True

    candidates = candidates[candidates_removed == False]
    print('semi cand', candidates)

    final_intervals = []
    candidates_merged = np.full(candidates.size, False)

    # Detecting overlapping candidates
    sz = len(candidates)
    for i in range(sz):
        if candidates_merged[i]:
            continue

        candidates_merged[i] = True
        overlap_interval = candidates[i]

        for j in range(i+1, sz):
            ci = candidates[j-1]
            i0, i1 = ci['interval']
            cj = candidates[j]
            j0, j1 = cj['interval']
            overlap = i1 - j0
            if overlap > 0:

                overlap_percent = np.mean([overlap/(i1-i0), overlap/(j1-j0)])

                slope_ratio = abs(
                    max(ci['slope'], cj['slope']) /
                    min(ci['slope'], cj['slope'])
                )
                std_ratio = max(ci['metric'], cj['metric']) / \
                    min(ci['metric'], cj['metric'])

                if overlap_percent > 0.5 and slope_ratio < 3 and std_ratio < 3:
                    candidates_merged[j] = True
                    cf = overlap_interval
                    f0, f1 = cf['interval']
                    new_slope = (cf['slope'] * (f1-f0) +
                                 cj['slope'] * (j1-j0)) / (f1-f0 + j1-j0)
                    new_score = (cf['metric'] * (f1-f0) +
                                 cj['metric'] * (j1-j0)) / (f1-f0 + j1-j0)
                    new_std = np.sqrt(cf['std']**2 + cj['std']**2)

                    itv = {'slope': new_slope, 'interval': (
                        f0, j1), 'metric': new_score, 'std': new_std}

                    overlap_interval = itv
                elif (ci['slope'] > 0.0) == (cj['slope'] > 0.0):
                    candidates_merged[j] = True
                    if j == i+1 and cj['metric'] < ci['metric']:
                        overlap_interval = cj
            else:
                break

        overlap_interval['interval'] = (
            float(overlap_interval['interval'][0]) * x_scale,
            float(overlap_interval['interval'][1]) * x_scale
        )

        final_intervals.append(overlap_interval)

    final_intervals = list(
        filter(
            lambda x: abs(x['slope']) > 0.001,
            final_intervals
        )
    )

    final_intervals = sorted(
        final_intervals, key=lambda x: abs(x['interval'][1] - x['interval'][0]), reverse=True)

    print('final', final_intervals)

    # return final_intervals[:10]
    return final_intervals
