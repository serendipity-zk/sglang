import unittest

from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.session_controller import Session
from sglang.srt.sampling.sampling_params import SamplingParams


class TestSidecarReqState(unittest.TestCase):
    def test_req_initializes_sidecar_runtime_state(self):
        req = Req(
            rid="req-1",
            origin_input_text="hello",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_new_tokens=8),
            target_ttft_ms=120.0,
            target_tpot_ms=35.0,
            arrival_time_ms=123456.0,
        )

        self.assertEqual(req.target_ttft_ms, 120.0)
        self.assertEqual(req.target_tpot_ms, 35.0)
        self.assertEqual(req.arrival_time_ms, 123456.0)
        self.assertFalse(req.slo_violated)

    def test_session_create_req_preserves_sidecar_fields(self):
        session = Session(capacity_of_str_len=1024, session_id="session-1")
        recv_req = TokenizedGenerateReqInput(
            input_text="hello",
            input_ids=[1, 2, 3],
            mm_inputs=None,
            sampling_params=SamplingParams(max_new_tokens=4),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            rid="req-2",
            session_params=SessionParams(id="session-1"),
            target_ttft_ms=150.0,
            target_tpot_ms=40.0,
            arrival_time_ms=98765.0,
        )

        req = session.create_req(
            recv_req,
            tokenizer=None,
            vocab_size=32000,
            eos_token_ids={2},
        )

        self.assertEqual(req.target_ttft_ms, 150.0)
        self.assertEqual(req.target_tpot_ms, 40.0)
        self.assertEqual(req.arrival_time_ms, 98765.0)
        self.assertFalse(req.slo_violated)


if __name__ == "__main__":
    unittest.main()
