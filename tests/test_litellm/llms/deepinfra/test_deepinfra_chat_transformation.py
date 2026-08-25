import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add litellm to path
import litellm


def test_deepseek_supported_openai_params(monkeypatch):
    """
    Test "reasoning_effort" is an openai param supported for the DeepSeek model on deepinfra
    """
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    # Ensure we're using the local model cost map
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    supported_openai_params = DeepInfraConfig().get_supported_openai_params(
        model="deepinfra/deepseek-ai/DeepSeek-V3.1"
    )
    print(supported_openai_params)
    assert "reasoning_effort" in supported_openai_params


def test_deepinfra_tool_message_content_transformation():
    """
    Test that DeepInfra transforms tool message content from array to string.

    This fixes the issue where LibreChat sends tool messages with content as an array:
    {"role": "tool", "content": [{"type": "text", "text": "20"}]}

    DeepInfra requires content to be a string, so we transform it to:
    {"role": "tool", "content": "20"}

    Related to issue #13982
    """
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    config = DeepInfraConfig()

    # Test case 1: Simple single text item in array (common case from LibreChat)
    messages_with_array_content = [
        {"role": "user", "content": "Calculate 10 + 10"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"input": "10 + 10"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "calculator",
            "content": [{"type": "text", "text": "20"}],  # Array format from LibreChat
        },
    ]

    transformed_messages = config._transform_messages(
        messages=messages_with_array_content, model="deepinfra/Qwen/Qwen3-235B-A22B"
    )

    # Verify the tool message content was converted to string
    tool_message = transformed_messages[2]
    assert tool_message["role"] == "tool"
    assert isinstance(tool_message["content"], str)
    assert tool_message["content"] == "20"
    print(f"✓ Test case 1 passed: {tool_message['content']}")

    # Test case 2: Complex content array (multiple items)
    messages_with_complex_content = [
        {"role": "user", "content": "Test"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": [
                {"type": "text", "text": "Result 1"},
                {"type": "text", "text": "Result 2"},
            ],
        },
    ]

    transformed_messages_complex = config._transform_messages(
        messages=messages_with_complex_content, model="deepinfra/Qwen/Qwen3-235B-A22B"
    )

    tool_message_complex = transformed_messages_complex[2]
    assert tool_message_complex["role"] == "tool"
    assert isinstance(tool_message_complex["content"], str)
    # For complex content, it should be JSON stringified
    parsed_content = json.loads(tool_message_complex["content"])
    assert len(parsed_content) == 2
    assert parsed_content[0]["text"] == "Result 1"
    print(f"✓ Test case 2 passed: {tool_message_complex['content']}")

    # Test case 3: Tool message with string content (should remain unchanged)
    messages_with_string_content = [
        {"role": "user", "content": "Test"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_789",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_789",
            "content": "Simple string result",  # Already a string
        },
    ]

    transformed_messages_string = config._transform_messages(
        messages=messages_with_string_content, model="deepinfra/Qwen/Qwen3-235B-A22B"
    )

    tool_message_string = transformed_messages_string[2]
    assert tool_message_string["role"] == "tool"
    assert isinstance(tool_message_string["content"], str)
    assert tool_message_string["content"] == "Simple string result"
    print(f"✓ Test case 3 passed: {tool_message_string['content']}")

    print("\n✅ All DeepInfra tool message transformation tests passed!")


@pytest.mark.asyncio
async def test_deepinfra_tool_message_content_transformation_async():
    """
    Test that DeepInfra transforms tool message content from array to string in async mode.

    This ensures the async path works correctly when is_async=True.

    Related to issue #13982
    """
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    config = DeepInfraConfig()

    # Test async transformation with tool message containing array content
    messages_with_array_content = [
        {"role": "user", "content": "Calculate 10 + 10"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"input": "10 + 10"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "calculator",
            "content": [{"type": "text", "text": "20"}],  # Array format from LibreChat
        },
    ]

    # Call with is_async=True
    transformed_messages = await config._transform_messages(
        messages=messages_with_array_content,
        model="deepinfra/Qwen/Qwen3-235B-A22B",
        is_async=True,
    )

    # Verify the tool message content was converted to string
    tool_message = transformed_messages[2]
    assert tool_message["role"] == "tool"
    assert isinstance(tool_message["content"], str)
    assert tool_message["content"] == "20"
    print(f"✓ Async test passed: {tool_message['content']}")

    print("\n✅ DeepInfra async tool message transformation test passed!")


def test_deepinfra_prompt_cache_retention_supported_params():
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    supported = DeepInfraConfig().get_supported_openai_params(
        model="deepinfra/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"
    )
    assert "prompt_cache_key" in supported
    assert "prompt_cache_options" in supported


def test_deepinfra_prompt_cache_retention_models():
    import json
    from pathlib import Path

    repo_root = Path(__file__).parents[4]
    with open(repo_root / "model_prices_and_context_window.json") as f:
        main_cost = json.load(f)
    with open(repo_root / "litellm/model_prices_and_context_window_backup.json") as f:
        backup_cost = json.load(f)

    expected = {
        "deepinfra/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": (5e-07, 2.2e-06, 1e-07),
        "deepinfra/moonshotai/Kimi-K2.7-Code": (6.8e-07, 3.4e-06, 1.36e-07),
    }
    for model, (input_cost, output_cost, cache_read_cost) in expected.items():
        assert backup_cost.get(model) == main_cost.get(model)
        info = main_cost[model]
        assert info["supports_prompt_caching"] is True
        assert info["input_cost_per_token"] == input_cost
        assert info["output_cost_per_token"] == output_cost
        assert info["cache_read_input_token_cost"] == cache_read_cost


def test_deepinfra_prompt_cache_retention_request():
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig
    from litellm.utils import get_optional_params

    optional_params = get_optional_params(
        model="deepinfra/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
        custom_llm_provider="deepinfra",
        prompt_cache_key="session-1",
        prompt_cache_options={"mode": "explicit", "ttl": "1h"},
        drop_params=False,
    )
    assert optional_params["prompt_cache_key"] == "session-1"
    assert optional_params["extra_body"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "1h",
    }

    request = DeepInfraConfig().transform_request(
        model="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "large stable prefix",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {"role": "user", "content": "variable question"},
        ],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )
    assert request["prompt_cache_key"] == "session-1"
    assert request["extra_body"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "1h",
    }
    assert request["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
