"""Select resident history; this does not recover frames already evicted."""


def select_history_frames(body, keep_n, recent, scores, threshold, previous=(), swap=-1):
    """Return chronological resident IDs and the non-pinned subset.

    Keep recent temporal context, select other frames by overlap, and fill
    rejected slots by recency. Limit optional replacements while always filling
    the window; slots vacated by aging/pinning require mandatory replacements.
    """
    body = list(body)
    if keep_n < 0 or keep_n > len(body) or len(set(body)) != len(body):
        raise ValueError("Invalid resident history budget")
    recent = min(max(0, recent), keep_n)
    pinned = body[-recent:] if recent else []
    candidates = [f for f in body if f not in pinned]
    room = keep_n - len(pinned)
    ranked = sorted(candidates, key=lambda f: (-scores.get(f, 0.0), -f))
    selected = [f for f in ranked if scores.get(f, 0.0) >= threshold][:room]
    fallback = sorted((f for f in candidates if f not in selected), reverse=True)
    selected += fallback[:room - len(selected)]
    prev = [f for f in previous if f in candidates]
    if prev and swap >= 0:
        held = [f for f in prev if f in selected][:room]
        incoming = [f for f in selected if f not in held]
        keep_prev = [f for f in prev if f not in held]
        # New slots must be filled even if there is no previous occupant.
        budget = max(swap, room - len(prev))
        take = incoming[:min(budget, room - len(held))]
        selected = held + take
        selected += keep_prev[:room - len(selected)]
        selected += [f for f in incoming if f not in selected][:room - len(selected)]
    keep = sorted(pinned + selected)
    if len(keep) != keep_n or len(set(keep)) != keep_n:
        raise RuntimeError("History selector did not fill the window uniquely")
    return keep, selected
