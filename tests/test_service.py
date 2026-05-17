"""Tests for the OmnilingualASRService configuration helpers."""

from unittest.mock import patch

import pytest

from app.service import OmnilingualASRService


class TestOmnilingualASRService:
    """Tests for the OmnilingualASRService class."""

    @pytest.mark.parametrize(
        "model_name,expected",
        [
            ("omniASR_W2V_300M", False),
            ("omniASR_W2V_1B", False),
            ("omniASR_W2V_3B", False),
            ("omniASR_W2V_7B", False),
            ("omniASR_CTC_300M", False),
            ("omniASR_CTC_1B", False),
            ("omniASR_CTC_3B", False),
            ("omniASR_CTC_7B", False),
            ("omniASR_CTC_300M_v2", False),
            ("omniASR_CTC_1B_v2", False),
            ("omniASR_CTC_3B_v2", False),
            ("omniASR_CTC_7B_v2", False),
            ("omniASR_LLM_300M", True),
            ("omniASR_LLM_1B", True),
            ("omniASR_LLM_3B", True),
            ("omniASR_LLM_7B", True),
            ("omniASR_LLM_300M_v2", True),
            ("omniASR_LLM_1B_v2", True),
            ("omniASR_LLM_3B_v2", True),
            ("omniASR_LLM_7B_v2", True),
            ("omniASR_LLM_Unlimited_300M_v2", True),
            ("omniASR_LLM_Unlimited_1B_v2", True),
            ("omniASR_LLM_Unlimited_3B_v2", True),
            ("omniASR_LLM_Unlimited_7B_v2", True),
            ("omniASR_LLM_7B_ZS", True),
        ],
    )
    def test_is_llm_model(self, model_name: str, expected: bool):
        """The standard-model LLM flag should follow MODEL_NAME."""

        with patch("app.service.MODEL_NAME", model_name):
            service = OmnilingualASRService()
            assert service.is_llm_model == expected

    def test_configured_model_names_include_zero_shot_model(self):
        """Configured models should expose both standard and zero-shot cards."""

        with (
            patch("app.service.MODEL_NAME", "omniASR_CTC_300M_v2"),
            patch("app.service.ZERO_SHOT_MODEL_NAME", "omniASR_LLM_7B_ZS"),
        ):
            service = OmnilingualASRService()

        assert service.configured_model_names == [
            "omniASR_CTC_300M_v2",
            "omniASR_LLM_7B_ZS",
        ]
        assert service.has_zero_shot_model is True

    def test_validate_model_configuration_rejects_zero_shot_primary_model(self):
        """The standard endpoint must not be bound to a zero-shot model card."""

        with patch("app.service.MODEL_NAME", "omniASR_LLM_7B_ZS"):
            service = OmnilingualASRService()

        with pytest.raises(RuntimeError, match="standard non-zero-shot model"):
            service._validate_model_configuration()

    def test_validate_model_configuration_rejects_non_zero_shot_secondary_model(self):
        """ZERO_SHOT_MODEL_NAME must point to a *_ZS model."""

        with (
            patch("app.service.MODEL_NAME", "omniASR_CTC_300M_v2"),
            patch("app.service.ZERO_SHOT_MODEL_NAME", "omniASR_LLM_1B_v2"),
        ):
            service = OmnilingualASRService()

        with pytest.raises(RuntimeError, match=r"ZERO_SHOT_MODEL_NAME must point to a \*_ZS model"):
            service._validate_model_configuration()

    @patch("app.service.OMNILINGUAL_DEVICE", "auto")
    @patch("app.service.torch.backends.mps.is_available", return_value=False)
    @patch("app.service.torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 5090")
    @patch("app.service.torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("app.service.torch.cuda.get_arch_list", return_value=["sm_90"])
    @patch("app.service.torch.cuda.is_available", return_value=True)
    def test_select_device_falls_back_to_cpu_when_cuda_arch_is_unsupported(
        self,
        _mock_cuda_available,
        _mock_arch_list,
        _mock_capability,
        _mock_device_name,
        _mock_mps_available,
    ):
        """Auto mode should fall back to CPU if the GPU arch is unsupported."""

        service = OmnilingualASRService()

        assert service._select_device() == "cpu"

    @patch("app.service.OMNILINGUAL_DEVICE", "cuda")
    @patch("app.service.torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 5090")
    @patch("app.service.torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("app.service.torch.cuda.get_arch_list", return_value=["sm_90"])
    @patch("app.service.torch.cuda.is_available", return_value=True)
    def test_select_device_raises_for_forced_unsupported_cuda(
        self,
        _mock_cuda_available,
        _mock_arch_list,
        _mock_capability,
        _mock_device_name,
    ):
        """Forced CUDA mode should fail fast with a useful error."""

        service = OmnilingualASRService()

        with pytest.raises(RuntimeError, match="cannot run on"):
            service._select_device()

    def test_select_vad_device_uses_cuda_only_for_cuda_inference(self):
        """Silero VAD should follow CUDA inference, otherwise stay on CPU."""

        service = OmnilingualASRService()

        assert service._select_vad_device("cuda").type == "cuda"
        assert service._select_vad_device("cpu").type == "cpu"
        assert service._select_vad_device("mps").type == "cpu"
