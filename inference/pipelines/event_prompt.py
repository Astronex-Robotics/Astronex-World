"""Shared encoding for the independent event-prompt channel.

The event prompt is a second text input that reaches the DiT through its own
cross-attention (see ``enable_event_conditioning``) instead of being appended to
the caption. Keeping it separate is what makes it controllable: it can be set,
swapped or dropped without disturbing the caption embedding, and "no event" is a
genuinely different code path rather than a shorter string.

All five inference pipelines take an ``event_prompts`` argument and route it
through here, so the flag behaves identically whichever one ``wan_inference``
picks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union


def merge_event_into_captions(
    text_prompts: Sequence[str],
    event_prompts: Optional[Union[str, Sequence[str]]],
) -> List[str]:
    """Append event semantics to the main caption for pretrained Wan control.

    The small independent event branch in this repository was trained only on
    DROID manipulation text.  Routing the event through the pretrained caption
    cross-attention lets open-domain events (fire, explosions, weather, etc.)
    use Wan's existing semantic prior.  Callers may still also feed the event
    branch when a suitable overlay is loaded.
    """
    captions = list(text_prompts)
    if not event_prompts:
        return captions
    events = [event_prompts] if isinstance(event_prompts, str) else list(event_prompts)
    if len(events) == 1 and len(captions) > 1:
        events *= len(captions)
    if len(events) != len(captions):
        raise ValueError(
            f"{len(events)} event prompts for {len(captions)} captions; "
            "pass one per row, or a single string to apply to all")
    return [f"{caption.rstrip()} {event.strip()}" if event and event.strip() else caption
            for caption, event in zip(captions, events)]


def add_event_embeds(
    conditional_dict: dict,
    text_encoder,
    event_prompts: Optional[Union[str, Sequence[str]]],
    batch_size: int,
) -> dict:
    """Put ``event_embeds`` on ``conditional_dict`` when there is event text.

    An absent or blank event prompt leaves the key off entirely, which makes the
    block skip its event branch -- not the same as encoding an empty string,
    which would run the branch on padding and cost a cross-attention per block
    for nothing.
    """
    if not event_prompts:
        return conditional_dict

    prompts: List[str] = (
        [event_prompts] if isinstance(event_prompts, str) else list(event_prompts))
    if len(prompts) == 1 and batch_size > 1:
        prompts = prompts * batch_size
    if len(prompts) != batch_size:
        raise ValueError(
            f"{len(prompts)} event prompts for a batch of {batch_size}; "
            "pass one per row, or a single string to apply to all")
    if not any(p and p.strip() for p in prompts):
        return conditional_dict

    conditional_dict["event_embeds"] = text_encoder(
        text_prompts=prompts)["prompt_embeds"]
    return conditional_dict
