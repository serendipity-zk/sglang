import pytest

from sglang.srt.managers.router_message_tracker import RouterMessageAckTracker


def test_tracker_records_contiguous_ids():
    tracker = RouterMessageAckTracker()
    assert tracker.get_state() == (None, None)

    tracker.record(100, 0)
    assert tracker.get_state() == (100, 0)

    tracker.record(100, 2)
    # Missing message 1, so contiguous progress stops at 0
    assert tracker.get_state() == (100, 0)

    tracker.record(100, 1)
    assert tracker.get_state() == (100, 2)


def test_tracker_generation_rollover_resets_state():
    tracker = RouterMessageAckTracker()

    tracker.record(100, 0)
    tracker.record(100, 1)
    assert tracker.get_state() == (100, 1)

    tracker.record(101, 0)
    assert tracker.get_state() == (101, 0)

    # Older generation updates are ignored
    tracker.record(100, 2)
    assert tracker.get_state() == (101, 0)

    tracker.record(101, 2)
    assert tracker.get_state() == (101, 0)

    tracker.record(101, 1)
    assert tracker.get_state() == (101, 2)
