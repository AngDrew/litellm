import json
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.utils import CustomStreamWrapper


class NeuralWattChatCompletion(CustomLLM):
    provider = "neuralwatt"

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
    ):
        if acompletion is True:
            return self.acompletion(
                model=model,
                messages=messages,
                api_base=api_base,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                api_key=api_key,
                logging_obj=logging_obj,
                optional_params=optional_params,
                acompletion=acompletion,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=headers,
                timeout=timeout,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        markup_pct = _get_markup_pct(litellm_params, optional_params)
        upstream_model = _get_upstream_model(model)
        request_data = {
            "model": upstream_model,
            "messages": messages,
            **_get_upstream_optional_params(optional_params),
        }
        request_headers = _get_headers(api_key=api_key, headers=headers)
        url = _get_chat_completions_url(api_base)

        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={"complete_input_dict": request_data, "api_base": url},
        )

        if optional_params.get("stream") is True:
            stream = _sync_stream(
                url=url,
                headers=request_headers,
                request_data=request_data,
                timeout=timeout,
                client=client,
                markup_pct=markup_pct,
                logging_obj=logging_obj,
            )
            return CustomStreamWrapper(
                completion_stream=stream,
                model=model,
                custom_llm_provider=self.provider,
                logging_obj=logging_obj,
            )

        if client is None:
            client = HTTPHandler(timeout=timeout)  # type: ignore[arg-type]

        response = client.post(url=url, headers=request_headers, data=json.dumps(request_data))
        if response.status_code >= 400:
            raise CustomLLMError(status_code=response.status_code, message=response.text)

        response_json = response.json()
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )
        returned_response = _build_model_response(
            response_json=response_json,
            markup_pct=markup_pct,
            litellm_model=model,
        )
        _record_neuralwatt_response_cost(
            logging_obj=logging_obj,
            response_cost=returned_response._hidden_params["response_cost"],
        )
        _record_neuralwatt_spend_metadata(
            logging_obj=logging_obj,
            energy=response_json.get("energy"),
            cost=response_json.get("cost"),
        )
        return returned_response

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ):
        markup_pct = _get_markup_pct(litellm_params, optional_params)
        upstream_model = _get_upstream_model(model)
        request_data = {
            "model": upstream_model,
            "messages": messages,
            **_get_upstream_optional_params(optional_params),
        }
        request_headers = _get_headers(api_key=api_key, headers=headers)
        url = _get_chat_completions_url(api_base)

        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={"complete_input_dict": request_data, "api_base": url},
        )

        if optional_params.get("stream") is True:
            stream = _async_stream(
                url=url,
                headers=request_headers,
                request_data=request_data,
                timeout=timeout,
                client=client,
                markup_pct=markup_pct,
                logging_obj=logging_obj,
            )
            return CustomStreamWrapper(
                completion_stream=stream,
                model=model,
                custom_llm_provider=self.provider,
                logging_obj=logging_obj,
            )

        if client is None:
            client = litellm.module_level_aclient

        response = await client.post(
            url=url,
            headers=request_headers,
            data=json.dumps(request_data),
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise CustomLLMError(status_code=response.status_code, message=response.text)

        response_json = response.json()
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )
        returned_response = _build_model_response(
            response_json=response_json,
            markup_pct=markup_pct,
            litellm_model=model,
        )
        _record_neuralwatt_response_cost(
            logging_obj=logging_obj,
            response_cost=returned_response._hidden_params["response_cost"],
        )
        _record_neuralwatt_spend_metadata(
            logging_obj=logging_obj,
            energy=response_json.get("energy"),
            cost=response_json.get("cost"),
        )
        return returned_response


def _get_upstream_model(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("neuralwatt/") else model


def _get_upstream_optional_params(optional_params: dict) -> dict:
    return {k: v for k, v in optional_params.items() if k != "markup_pct"}


def _get_chat_completions_url(api_base: Optional[str]) -> str:
    if not api_base:
        raise CustomLLMError(
            status_code=400, message="api_base is required for this model"
        )
    return api_base.rstrip("/") + "/chat/completions"


def _get_headers(api_key: Optional[str], headers: Optional[dict]) -> dict:
    dynamic_api_key = api_key or get_secret_str("NEURALWATT_API_KEY")
    if not dynamic_api_key:
        raise CustomLLMError(
            status_code=401, message="api_key is required for this model"
        )
    request_headers = dict(headers or {})
    request_headers["Authorization"] = f"Bearer {dynamic_api_key}"
    request_headers["Content-Type"] = "application/json"
    return request_headers


def _get_markup_pct(
    litellm_params: Optional[dict], optional_params: Optional[dict] = None
) -> Decimal:
    raw_markup = (litellm_params or {}).get(
        "markup_pct", (optional_params or {}).get("markup_pct", 0)
    )
    try:
        markup_pct = Decimal(str(raw_markup))
    except (InvalidOperation, ValueError):
        raise CustomLLMError(status_code=400, message="markup_pct must be a number")
    if markup_pct < 0:
        raise CustomLLMError(status_code=400, message="markup_pct must be non-negative")
    return markup_pct


def _calculate_user_cost(upstream_cost: Any, markup_pct: Decimal) -> float:
    try:
        base_cost = Decimal(str(upstream_cost))
    except (InvalidOperation, ValueError):
        raise CustomLLMError(
            status_code=502,
            message="Provider response missing numeric cost.request_cost_usd",
        )
    return float(base_cost * (Decimal("1") + (markup_pct / Decimal("100"))))


def _sanitize_cost(cost: Any, markup_pct: Decimal) -> Dict[str, Any]:
    if not isinstance(cost, dict) or cost.get("request_cost_usd") is None:
        raise CustomLLMError(
            status_code=502, message="Provider response missing cost.request_cost_usd"
        )
    user_cost = _calculate_user_cost(cost.get("request_cost_usd"), markup_pct)
    return {
        "upstream_request_cost_usd": cost.get("request_cost_usd"),
        "request_cost_usd": user_cost,
        "markup_pct": float(markup_pct),
    }


def _energy_headers(energy: Any) -> Dict[str, str]:
    if not isinstance(energy, dict):
        return {}
    header_map = {
        "energy_joules": "X-Energy-Joules",
        "energy_kwh": "X-Energy-Kwh",
        "avg_power_watts": "X-Energy-Avg-Power-Watts",
        "duration_seconds": "X-Energy-Duration-Seconds",
    }
    return {
        header: str(energy[key])
        for key, header in header_map.items()
        if energy.get(key) is not None
    }


def _apply_neuralwatt_metadata(
    response_json: Dict[str, Any], markup_pct: Decimal
) -> float:
    sanitized_cost = _sanitize_cost(response_json.get("cost"), markup_pct)
    usage = response_json.setdefault("usage", {})
    usage["cost_usd"] = sanitized_cost["request_cost_usd"]
    response_json["cost"] = sanitized_cost
    return sanitized_cost["request_cost_usd"]


def _record_neuralwatt_spend_metadata(
    *, logging_obj: Any, energy: Any, cost: Any
) -> None:
    if not isinstance(energy, dict) and not isinstance(cost, dict):
        return
    model_call_details = getattr(logging_obj, "model_call_details", None)
    if not isinstance(model_call_details, dict):
        return
    litellm_params = model_call_details.setdefault("litellm_params", {})
    if not isinstance(litellm_params, dict):
        return
    metadata = litellm_params.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        return
    spend_logs_metadata = metadata.setdefault("spend_logs_metadata", {})
    if not isinstance(spend_logs_metadata, dict):
        return
    if isinstance(energy, dict):
        spend_logs_metadata["neuralwatt_energy"] = energy
    if isinstance(cost, dict):
        spend_logs_metadata["neuralwatt_cost"] = cost


def _record_neuralwatt_response_cost(
    *, logging_obj: Any, response_cost: Optional[float]
) -> None:
    model_call_details = getattr(logging_obj, "model_call_details", None)
    if isinstance(model_call_details, dict):
        model_call_details["response_cost"] = response_cost


def _build_model_response(
    *,
    response_json: Dict[str, Any],
    markup_pct: Decimal,
    litellm_model: str,
) -> ModelResponse:
    response_cost = _apply_neuralwatt_metadata(response_json, markup_pct)
    returned_response = ModelResponse(**response_json)
    returned_response.model = (
        litellm_model
        if litellm_model.startswith("neuralwatt/")
        else f"neuralwatt/{litellm_model}"
    )
    returned_response._hidden_params["response_cost"] = response_cost
    returned_response._hidden_params["additional_headers"] = _energy_headers(
        response_json.get("energy")
    )
    return returned_response


def _parse_sse_payload(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def _parse_comment_payload(line: str, prefix: str) -> Optional[Dict[str, Any]]:
    marker = f": {prefix} "
    if not line.startswith(marker):
        return None
    return json.loads(line[len(marker) :].strip())


def _chunk_from_payload(payload: Dict[str, Any]) -> GenericStreamingChunk:
    usage = payload.get("usage")
    choice = payload.get("choices", [{}])[0] if payload.get("choices") else {}
    delta = choice.get("delta") or {}
    tool_calls = delta.get("tool_calls")
    return {
        "text": delta.get("content") or "",
        "is_finished": choice.get("finish_reason") is not None,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "tool_use": (
            tool_calls[0] if isinstance(tool_calls, list) and tool_calls else None
        ),
    }


def _final_metadata_chunks(
    energy: Dict[str, Any], cost: Dict[str, Any]
) -> List[GenericStreamingChunk]:
    return [
        {
            "text": "",
            "is_finished": False,
            "finish_reason": None,
            "usage": None,
            "tool_use": None,
            "provider_specific_fields": {
                "neuralwatt_sse_comment": f": energy {json.dumps(energy)}"
            },
        },
        {
            "text": "",
            "is_finished": False,
            "finish_reason": None,
            "usage": None,
            "tool_use": None,
            "provider_specific_fields": {
                "neuralwatt_sse_comment": f": cost {json.dumps(cost)}"
            },
        },
    ]


def _sync_stream(
    *,
    url: str,
    headers: dict,
    request_data: dict,
    timeout: Optional[Union[float, httpx.Timeout]],
    client: Optional[HTTPHandler],
    markup_pct: Decimal,
    logging_obj,
):
    if client is None:
        client = HTTPHandler(timeout=timeout)  # type: ignore[arg-type]
    response = client.post(
        url=url,
        headers=headers,
        data=json.dumps(request_data),
        stream=True,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise CustomLLMError(status_code=response.status_code, message=response.read())
    energy: Dict[str, Any] = {}
    cost: Dict[str, Any] = {}
    pending_usage_payload: Optional[Dict[str, Any]] = None
    for line in response.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        energy = _parse_comment_payload(line, "energy") or energy
        upstream_cost = _parse_comment_payload(line, "cost")
        if upstream_cost is not None:
            cost = _sanitize_cost(upstream_cost, markup_pct)
            _record_neuralwatt_response_cost(
                logging_obj=logging_obj,
                response_cost=cost["request_cost_usd"],
            )
            if pending_usage_payload is not None:
                pending_usage_payload["usage"]["cost_usd"] = cost["request_cost_usd"]
                yield _chunk_from_payload(pending_usage_payload)
                pending_usage_payload = None
            continue
        payload = _parse_sse_payload(line)
        if payload is None:
            continue
        if isinstance(payload.get("usage"), dict):
            if cost:
                payload["usage"]["cost_usd"] = cost["request_cost_usd"]
            else:
                pending_usage_payload = payload
                continue
        yield _chunk_from_payload(payload)
    if not cost:
        raise CustomLLMError(
            status_code=502, message="Provider stream missing cost.request_cost_usd"
        )
    if pending_usage_payload is not None:
        pending_usage_payload["usage"]["cost_usd"] = cost["request_cost_usd"]
        yield _chunk_from_payload(pending_usage_payload)
    _record_neuralwatt_spend_metadata(
        logging_obj=logging_obj,
        energy=energy,
        cost=cost,
    )
    for chunk in _final_metadata_chunks(energy=energy, cost=cost):
        yield chunk


async def _async_stream(
    *,
    url: str,
    headers: dict,
    request_data: dict,
    timeout: Optional[Union[float, httpx.Timeout]],
    client: Optional[AsyncHTTPHandler],
    markup_pct: Decimal,
    logging_obj,
) -> AsyncIterator[GenericStreamingChunk]:
    if client is None:
        client = litellm.module_level_aclient
    response = await client.post(
        url=url,
        headers=headers,
        data=json.dumps(request_data),
        stream=True,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise CustomLLMError(status_code=response.status_code, message=response.text)
    energy: Dict[str, Any] = {}
    cost: Dict[str, Any] = {}
    pending_usage_payload: Optional[Dict[str, Any]] = None
    async for line in response.aiter_lines():
        if not line:
            continue
        energy = _parse_comment_payload(line, "energy") or energy
        upstream_cost = _parse_comment_payload(line, "cost")
        if upstream_cost is not None:
            cost = _sanitize_cost(upstream_cost, markup_pct)
            _record_neuralwatt_response_cost(
                logging_obj=logging_obj,
                response_cost=cost["request_cost_usd"],
            )
            if pending_usage_payload is not None:
                pending_usage_payload["usage"]["cost_usd"] = cost["request_cost_usd"]
                yield _chunk_from_payload(pending_usage_payload)
                pending_usage_payload = None
            continue
        payload = _parse_sse_payload(line)
        if payload is None:
            continue
        if isinstance(payload.get("usage"), dict):
            if cost:
                payload["usage"]["cost_usd"] = cost["request_cost_usd"]
            else:
                pending_usage_payload = payload
                continue
        yield _chunk_from_payload(payload)
    if not cost:
        raise CustomLLMError(
            status_code=502, message="Provider stream missing cost.request_cost_usd"
        )
    if pending_usage_payload is not None:
        pending_usage_payload["usage"]["cost_usd"] = cost["request_cost_usd"]
        yield _chunk_from_payload(pending_usage_payload)
    _record_neuralwatt_spend_metadata(
        logging_obj=logging_obj,
        energy=energy,
        cost=cost,
    )
    for chunk in _final_metadata_chunks(energy=energy, cost=cost):
        yield chunk
