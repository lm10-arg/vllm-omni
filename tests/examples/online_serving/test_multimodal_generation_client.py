# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the shared multimodal OpenAI example client."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass(frozen=True)
class _Args:
    model: str = "Qwen/Qwen2.5-Omni-7B"
    query_type: str = "text"
    video_path: None = None
    image_path: None = None
    audio_path: None = None
    prompt: str = "Summarize the input."
    modalities: str = "audio,text"
    stream: bool = False
    num_concurrent_requests: int = 1
    prompts: None = None
    speaker: None = None
    speakers: None = None


def _load_client_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    client_path = (
        repo_root / "examples" / "online_serving" / "openai_chat_completion_client_for_multimodal_generation.py"
    )
    spec = importlib.util.spec_from_file_location("multimodal_generation_client", client_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multimodal_generation_requests_are_deterministic(mocker) -> None:
    module = _load_client_module()
    completion = mocker.Mock(choices=[])
    create = mocker.Mock(return_value=completion)
    client = mocker.Mock()
    client.chat.completions.create = create

    module.run_multimodal_generation(_Args(), client)

    create.assert_called_once()
    request = create.call_args.kwargs
    assert request["seed"] == module.SEED == 42
    assert request["temperature"] == 0
