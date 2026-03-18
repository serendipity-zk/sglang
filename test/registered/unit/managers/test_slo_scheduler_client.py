import unittest
from types import SimpleNamespace
from unittest.mock import patch

import zmq

from sglang.srt.managers.slo_scheduler_client import SLOSchedulerClient


class FakeSocket:
    def __init__(self, poll_results=None, recv_frames=None):
        self.poll_results = list(poll_results or [])
        self.recv_frames = list(recv_frames or [])
        self.sockopts = []
        self.connected_addr = None
        self.sent_frames = []
        self.closed = False

    def setsockopt(self, opt, value):
        self.sockopts.append((opt, value))

    def connect(self, addr):
        self.connected_addr = addr

    def send_multipart(self, frames, flags=0):
        self.sent_frames.append((frames, flags))

    def poll(self, timeout):
        if self.poll_results:
            return self.poll_results.pop(0)
        return 0

    def recv_multipart(self, flags=0):
        if self.recv_frames:
            return self.recv_frames.pop(0)
        raise zmq.Again()

    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self, socket):
        self._socket = socket
        self.terminated = False

    def socket(self, _socket_type):
        return self._socket

    def term(self):
        self.terminated = True


class TestSLOSchedulerClient(unittest.TestCase):
    def make_client(self, socket):
        context = FakeContext(socket)
        with patch(
            "sglang.srt.managers.slo_scheduler_client.zmq.Context",
            return_value=context,
        ):
            client = SLOSchedulerClient("ipc:///tmp/sidecar.sock", timeout_ms=50)
        return client, context

    def test_send_and_recv_returns_matching_decision_after_draining_stale(self):
        socket = FakeSocket(
            poll_results=[1],
            recv_frames=[[b"", b"stale"], [b"", b"match"]],
        )
        client, _context = self.make_client(socket)

        with (
            patch.object(client, "_serialize_engine_state", return_value=b"payload"),
            patch.object(
                client,
                "_deserialize_decision",
                side_effect=[
                    SimpleNamespace(iteration_count=4),
                    SimpleNamespace(iteration_count=5),
                ],
            ),
        ):
            decision = client.send_and_recv(object(), current_iteration=5)

        self.assertEqual(decision.iteration_count, 5)
        self.assertEqual(socket.sent_frames, [([b"", b"payload"], zmq.DONTWAIT)])

    def test_send_and_recv_returns_none_on_timeout(self):
        socket = FakeSocket(poll_results=[0])
        client, _context = self.make_client(socket)

        with patch.object(client, "_serialize_engine_state", return_value=b"payload"):
            decision = client.send_and_recv(object(), current_iteration=5)

        self.assertIsNone(decision)
        self.assertEqual(socket.sent_frames, [([b"", b"payload"], zmq.DONTWAIT)])

    def test_send_and_recv_rejects_only_stale_decisions(self):
        socket = FakeSocket(
            poll_results=[1, 0],
            recv_frames=[[b"", b"stale"]],
        )
        client, _context = self.make_client(socket)

        with (
            patch.object(client, "_serialize_engine_state", return_value=b"payload"),
            patch.object(
                client,
                "_deserialize_decision",
                return_value=SimpleNamespace(iteration_count=4),
            ),
        ):
            decision = client.send_and_recv(object(), current_iteration=5)

        self.assertIsNone(decision)

    def test_close_closes_socket_and_context(self):
        socket = FakeSocket()
        client, context = self.make_client(socket)

        client.close()

        self.assertTrue(socket.closed)
        self.assertTrue(context.terminated)


if __name__ == "__main__":
    unittest.main()
