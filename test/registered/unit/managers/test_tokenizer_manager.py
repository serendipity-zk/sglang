import types
import unittest
from unittest.mock import AsyncMock, MagicMock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _DummyAsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyPauseCond:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def wait_for(self, predicate):
        assert predicate()


class TestTokenizerManagerSLOMargin(unittest.IsolatedAsyncioTestCase):
    async def test_generate_request_applies_slo_margin_before_tokenization(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = types.SimpleNamespace(
            slo_target_margin=0.1,
            dp_size=1,
            language_only=False,
            tokenizer_worker_num=1,
        )
        manager.auto_create_handle_loop = MagicMock()
        manager._set_default_priority = MagicMock()
        manager._validate_rid = MagicMock()
        manager._req_stats_init = MagicMock()
        manager._validate_and_resolve_lora = AsyncMock()
        manager._send_one_request = MagicMock()
        manager._handle_epd_disaggregation_encode_request = MagicMock()
        manager._attach_multi_http_worker_info = MagicMock()
        manager.request_logger = types.SimpleNamespace(log_received_request=MagicMock())
        manager.tokenizer = object()
        manager.is_pause_cond = _DummyPauseCond()
        manager.is_pause = False
        manager.model_update_lock = types.SimpleNamespace(
            reader_lock=_DummyAsyncContext()
        )
        manager.rid_to_state = {"rid-1": object()}

        captured = {}

        async def _tokenize_one_request(obj):
            captured["target_ttft_ms"] = obj.target_ttft_ms
            captured["target_tpot_ms"] = obj.target_tpot_ms
            captured["arrival_time_ms"] = obj.arrival_time_ms
            return "tokenized"

        async def _wait_one_response(_obj, _state, _request):
            yield "done"

        manager._tokenize_one_request = _tokenize_one_request
        manager._wait_one_response = _wait_one_response

        req = GenerateReqInput(
            text="Hello",
            sampling_params={},
            rid="rid-1",
            target_ttft_ms=100.0,
            target_tpot_ms=50.0,
            arrival_time_ms=1234.5,
        )

        results = []
        async for item in manager.generate_request(req):
            results.append(item)

        self.assertEqual(results, ["done"])
        self.assertEqual(captured["target_ttft_ms"], 90.0)
        self.assertEqual(captured["target_tpot_ms"], 45.0)
        self.assertEqual(captured["arrival_time_ms"], 1234.5)


if __name__ == "__main__":
    unittest.main()
