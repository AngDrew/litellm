"""LiteLLM provider for TypeSafe's decisions API (Jev), so Jev calls are metered like any other model.

Jev is not a chat model: OpenRouter answers `/chat/completions` for it with 400 and serves the same
model on `/api/alpha/decisions`, whose body is `{model, state, questions}` and whose reply is typed
answers rather than text. This provider carries that endpoint into LiteLLM, which is what lets the
complexity router's Jev classifier calls land in the spend log, feed the per-tier budgets, and be
attributed to the caller that triggered them. A raw HTTP call from the classifier plugin is invisible
to all three, so the classifier would cost money nobody recorded.

The `messages` argument is the transport body, not a conversation: exactly one message whose content
is the JSON decisions request (`{"state": ..., "questions": ...}`), the shape
`callbacks.jev_classifier.build_payload` returns. The answered body comes back verbatim as the
message content, so the caller reads typed answers out of it. Wired up in the proxy config as:

    litellm_settings:
      custom_provider_map:
        - provider: typesafe
          custom_handler: callbacks.jev_provider.handler

and used by a deployment whose `model` is the author-qualified id (`jev-1.13` with
`custom_llm_provider: typesafe`), which this provider posts to OpenRouter as `typesafe/jev-1.13`.

Cost: OpenRouter reports the exact charge for each call in `usage.cost`, so that is the recorded
`response_cost`. The deployment's own `input_cost_per_token`/`output_cost_per_token` price the call
when the field is absent. Environment: `OR_API_KEY` (or the deployment's `api_key`), `OR_BASE_URL`
or an explicit `JEV_DECISIONS_URL`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any, Final

import httpx

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import ModelResponse, Usage

PROVIDER: Final = "typesafe"
DEFAULT_BASE_URL: Final = "https://openrouter.ai/api/v1"
DECISIONS_PATH: Final = "/alpha/decisions"
DEFAULT_TIMEOUT_SECONDS: Final = 10.0

_TRANSPORT_CONTRACT: Final = (
    "typesafe is a decisions provider: send exactly one message whose content is the JSON decisions "
    'request {"state": ..., "questions": ...}, the body POSTed to /api/alpha/decisions'
)


def decisions_url(api_base: str | None) -> str:
    """The decisions endpoint for a deployment's api_base, which is the v1 base on OpenRouter."""
    explicit = os.environ.get("JEV_DECISIONS_URL")
    if explicit:
        return explicit
    base = (api_base or os.environ.get("OR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    return f"{base.removesuffix('/v1')}{DECISIONS_PATH}"


def openrouter_model(model: str) -> str:
    """The OpenRouter id for the deployment's model name.

    The provider name is Jev's author prefix, so a deployment written as `jev-1.13` with
    `custom_llm_provider: typesafe` posts as `typesafe/jev-1.13`. The prefix is added here rather
    than written into the deployment because LiteLLM registers a deployment's custom pricing under
    `{custom_llm_provider}/{model}` and looks it up the same way: a model string that already
    carries the prefix registers as `typesafe/typesafe/jev-1.13`, which the router's own pre-call
    check then fails to find, logging an error and skipping the context-window guard on every call.
    """
    return model if model.startswith(f"{PROVIDER}/") else f"{PROVIDER}/{model}"


def decisions_request(messages: list, model: str) -> dict[str, Any]:
    """The decisions body to POST: the caller's JSON plus the model this deployment names."""
    if len(messages) != 1 or not isinstance(messages[0], Mapping):
        raise CustomLLMError(status_code=400, message=_TRANSPORT_CONTRACT)
    content = messages[0].get("content")
    if not isinstance(content, str):
        raise CustomLLMError(status_code=400, message=_TRANSPORT_CONTRACT)
    try:
        body = json.loads(content)
    except json.JSONDecodeError:
        raise CustomLLMError(status_code=400, message=_TRANSPORT_CONTRACT) from None
    if not isinstance(body, dict) or "state" not in body or "questions" not in body:
        raise CustomLLMError(status_code=400, message=_TRANSPORT_CONTRACT)
    return {**body, "model": openrouter_model(model)}


def _request_headers(api_key: Any, headers: Mapping[str, Any]) -> dict[str, str]:
    key = api_key or os.environ.get("OR_API_KEY")
    if not key:
        raise CustomLLMError(status_code=401, message="No OpenRouter key: set OR_API_KEY on the deployment")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **dict(headers)}


def _usage(answer: Mapping[str, Any]) -> Usage:
    usage = answer.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return Usage(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens
    )


def _reported_cost(answer: Mapping[str, Any]) -> float | None:
    """OpenRouter's exact charge, or None to let LiteLLM price the call from the deployment."""
    usage = answer.get("usage")
    cost = usage.get("cost") if isinstance(usage, Mapping) else None
    return float(cost) if isinstance(cost, (int, float)) else None


def build_model_response(
    answer: Mapping[str, Any], model: str, cost: float | None, model_response: Any = None
) -> ModelResponse:
    """The answered decisions body as a completion, with its cost recorded on the response."""
    response: Final = ModelResponse(
        id=str(answer.get("id") or "jev-decisions"),
        created=int(time.time()),
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(answer)},
                "finish_reason": "stop",
            }
        ],
        usage=_usage(answer),
    )
    if cost is not None:
        response._hidden_params["response_cost"] = cost
    if model_response is not None:
        model_response._hidden_params["response_cost"] = cost
    return response


def record_response_cost(logging_obj: Any, cost: float | None) -> None:
    """Hand this call's cost to LiteLLM's logging, which is where the spend log reads it."""
    if cost is None:
        return
    model_call_details = getattr(logging_obj, "model_call_details", None)
    if isinstance(model_call_details, dict):
        model_call_details["response_cost"] = cost


def _answer_from_response(response: httpx.Response, model: str, logging_obj: Any, model_response: Any) -> ModelResponse:
    if response.status_code >= 400:
        raise CustomLLMError(status_code=response.status_code, message=response.text[:500])
    try:
        answer = response.json()
    except ValueError:
        raise CustomLLMError(status_code=502, message="Decisions endpoint returned a non-JSON body") from None
    if not isinstance(answer, Mapping):
        raise CustomLLMError(status_code=502, message="Decisions endpoint returned an unexpected body")
    cost: Final = _reported_cost(answer)
    record_response_cost(logging_obj, cost)
    return build_model_response(answer, model, cost, model_response)


class TypeSafeDecisions(CustomLLM):
    """Jev through the decisions endpoint; see the module docstring for the `messages` contract."""

    provider = PROVIDER

    def __init__(self, transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def _timeout(self, timeout: Any) -> Any:
        return timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: Any,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: Any = {},
        timeout: Any = None,
        client: Any = None,
    ) -> ModelResponse:
        body: Final = decisions_request(messages, model)
        async with httpx.AsyncClient(
            timeout=self._timeout(timeout),
            transport=self._transport,  # type: ignore[arg-type]  # mock transports implement both halves
        ) as http_client:
            response: Final = await http_client.post(
                decisions_url(api_base), json=body, headers=_request_headers(api_key, headers)
            )
        return _answer_from_response(response, model, logging_obj, model_response)

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: Any,
        print_verbose: Any,
        encoding: Any,
        api_key: Any,
        logging_obj: Any,
        optional_params: dict,
        acompletion: Any = None,
        litellm_params: Any = None,
        logger_fn: Any = None,
        headers: Any = {},
        timeout: Any = None,
        client: Any = None,
    ) -> ModelResponse:
        if acompletion is True:
            return self.acompletion(  # type: ignore[return-value]  # the async half is what litellm awaits
                model,
                messages,
                api_base,
                custom_prompt_dict,
                model_response,
                print_verbose,
                encoding,
                api_key,
                logging_obj,
                optional_params,
                acompletion,
                litellm_params,
                logger_fn,
                headers,
                timeout,
                client,
            )
        body: Final = decisions_request(messages, model)
        with httpx.Client(
            timeout=self._timeout(timeout),
            transport=self._transport,  # type: ignore[arg-type]  # same halves as above
        ) as http_client:
            response: Final = http_client.post(
                decisions_url(api_base), json=body, headers=_request_headers(api_key, headers)
            )
        return _answer_from_response(response, model, logging_obj, model_response)

    def streaming(self, *args: Any, **kwargs: Any) -> Any:
        raise CustomLLMError(status_code=400, message="decisions answers are typed, not streamed")

    async def astreaming(self, *args: Any, **kwargs: Any) -> Any:
        raise CustomLLMError(status_code=400, message="decisions answers are typed, not streamed")


handler: Final = TypeSafeDecisions()
