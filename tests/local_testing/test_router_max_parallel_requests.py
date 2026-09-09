# What is this?
## Unit tests for the max_parallel_requests feature on Router
import asyncio
import inspect
import os
import time
import traceback
from datetime import datetime

import pytest

from typing import Optional

import litellm
from litellm.utils import calculate_max_parallel_requests

"""
- only rpm
- only tpm
- only max_parallel_requests 
- max_parallel_requests + rpm 
- max_parallel_requests + tpm
- max_parallel_requests + tpm + rpm 
"""


max_parallel_requests_values = [None, 10]
tpm_values = [None, 20, 300000]
rpm_values = [None, 30]
default_max_parallel_requests = [None, 40]


@pytest.mark.parametrize(
    "max_parallel_requests, tpm, rpm, default_max_parallel_requests",
    [
        (mp, tp, rp, dmp)
        for mp in max_parallel_requests_values
        for tp in tpm_values
        for rp in rpm_values
        for dmp in default_max_parallel_requests
    ],
)
def test_scenario(max_parallel_requests, tpm, rpm, default_max_parallel_requests):
    calculated_max_parallel_requests = calculate_max_parallel_requests(
        max_parallel_requests=max_parallel_requests,
        rpm=rpm,
        tpm=tpm,
        default_max_parallel_requests=default_max_parallel_requests,
    )
    if max_parallel_requests is not None:
        assert max_parallel_requests == calculated_max_parallel_requests
    elif rpm is not None:
        assert rpm == calculated_max_parallel_requests
    elif tpm is not None:
        calculated_rpm = int(tpm / 1000 * 6)
        if calculated_rpm == 0:
            calculated_rpm = 1
        print(
            f"test calculated_rpm: {calculated_rpm}, calculated_max_parallel_requests={calculated_max_parallel_requests}"
        )
        assert calculated_rpm == calculated_max_parallel_requests
    elif default_max_parallel_requests is not None:
        assert calculated_max_parallel_requests == default_max_parallel_requests
    else:
        assert calculated_max_parallel_requests is None


@pytest.mark.parametrize(
    "max_parallel_requests, tpm, rpm, default_max_parallel_requests",
    [
        (mp, tp, rp, dmp)
        for mp in max_parallel_requests_values
        for tp in tpm_values
        for rp in rpm_values
        for dmp in default_max_parallel_requests
    ],
)
def test_setting_mpr_limits_per_model(
    max_parallel_requests, tpm, rpm, default_max_parallel_requests
):
    deployment = {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "gpt-3.5-turbo",
            "max_parallel_requests": max_parallel_requests,
            "tpm": tpm,
            "rpm": rpm,
        },
        "model_info": {"id": "my-unique-id"},
    }

    router = litellm.Router(
        model_list=[deployment],
        default_max_parallel_requests=default_max_parallel_requests,
    )

    mpr_client: Optional[asyncio.Semaphore] = router._get_client(
        deployment=deployment,
        kwargs={},
        client_type="max_parallel_requests",
    )

    if max_parallel_requests is not None:
        assert max_parallel_requests == mpr_client._value
    elif rpm is not None:
        assert rpm == mpr_client._value
    elif tpm is not None:
        calculated_rpm = int(tpm / 1000 * 6)
        if calculated_rpm == 0:
            calculated_rpm = 1
        print(
            f"test calculated_rpm: {calculated_rpm}, calculated_max_parallel_requests={mpr_client._value}"
        )
        assert calculated_rpm == mpr_client._value
    elif default_max_parallel_requests is not None:
        assert mpr_client._value == default_max_parallel_requests
    else:
        assert mpr_client is None

    # raise Exception("it worked!")


@pytest.mark.asyncio
async def test_provider_max_parallel_requests_shared_semaphore():
    """
    Deployments with provider_max_parallel_requests pointing at the same
    provider (same api_base + api_key) must share ONE semaphore, so N models
    on a single account can't multiply the concurrency cap.
    """
    from litellm.router_utils.client_initalization_utils import (
        InitalizeCachedClient,
    )

    deployments = [
        {
            "model_name": f"model-{i}",
            "litellm_params": {
                "model": f"openai/model-{i}",
                "api_base": "https://ollama.example.com/v1",
                "api_key": "sk-ollama",
                "provider_max_parallel_requests": 3,
            },
            "model_info": {"id": f"deployment-{i}"},
        }
        for i in range(3)
    ]

    router = litellm.Router(model_list=deployments)

    semaphores = [
        router._get_client(
            deployment=d,
            kwargs={},
            client_type="max_parallel_requests",
        )
        for d in deployments
    ]

    # all three deployments resolve to the SAME semaphore object
    assert all(s is semaphores[0] for s in semaphores)
    assert semaphores[0]._value == 3

    # per-deployment max_parallel_requests still gets its own semaphore
    solo = {
        "model_name": "solo",
        "litellm_params": {
            "model": "openai/solo",
            "api_base": "https://ollama.example.com/v1",
            "api_key": "sk-ollama",
            "max_parallel_requests": 5,
        },
        "model_info": {"id": "solo-deployment"},
    }
    solo_semaphore = router._get_client(
        deployment=solo,
        kwargs={},
        client_type="max_parallel_requests",
    )
    assert solo_semaphore is not semaphores[0]
    assert solo_semaphore._value == 5

    # different api_key ⇒ different provider ⇒ different semaphore
    other = {
        "model_name": "other",
        "litellm_params": {
            "model": "openai/other",
            "api_base": "https://ollama.example.com/v1",
            "api_key": "sk-other",
            "provider_max_parallel_requests": 3,
        },
        "model_info": {"id": "other-deployment"},
    }
    other_semaphore = router._get_client(
        deployment=other,
        kwargs={},
        client_type="max_parallel_requests",
    )
    assert other_semaphore is not semaphores[0]

    # cache key helper agrees with the lookup
    assert "sk-ollama" not in InitalizeCachedClient.get_max_parallel_requests_cache_key(deployments[0])


@pytest.mark.asyncio
async def test_provider_max_parallel_requests_mixed_values_use_min():
    """
    Deployments sharing a provider (same api_base + api_key) with DIFFERENT
    provider_max_parallel_requests values share one semaphore capped at the
    MINIMUM value, regardless of which deployment is looked up first.
    """
    from litellm.router_utils.client_initalization_utils import (
        InitalizeCachedClient,
    )

    deployments = [
        {
            "model_name": f"model-{i}",
            "litellm_params": {
                "model": f"openai/model-{i}",
                "api_base": "https://ollama.example.com/v1",
                "api_key": "sk-ollama",
                "provider_max_parallel_requests": cap,
            },
            "model_info": {"id": f"deployment-{i}"},
        }
        for i, cap in enumerate([5, 3, 8])
    ]

    router = litellm.Router(model_list=deployments)

    # look up the deployment with the HIGHEST declared cap first: the shared
    # semaphore must still end up at the min (3), not the first-seen value
    semaphores = [
        router._get_client(
            deployment=d,
            kwargs={},
            client_type="max_parallel_requests",
        )
        for d in deployments
    ]
    assert all(s is semaphores[0] for s in semaphores)
    assert semaphores[0]._value == 3


@pytest.mark.asyncio
async def test_least_busy_skips_saturated_deployment():
    """
    least-busy must not select a deployment whose max_parallel_requests
    semaphore is saturated: a saturated deployment cannot accept the request,
    so it shouldn't be selectable.
    """
    router = litellm.Router(
        routing_strategy="least-busy",
        model_list=[
            {
                "model_name": "test-group",
                "litellm_params": {
                    "model": "openai/cap-model",
                    "provider_max_parallel_requests": 1,
                    "api_base": "https://cap.example.com/v1",
                    "api_key": "sk-cap",
                },
                "model_info": {"id": "cap-deployment"},
            },
            {
                "model_name": "test-group",
                "litellm_params": {
                    "model": "openai/free-model",
                    "provider_max_parallel_requests": 1,
                    "api_base": "https://free.example.com/v1",
                    "api_key": "sk-free",
                },
                "model_info": {"id": "free-deployment"},
            },
        ],
    )

    cap_semaphore = router._get_client(
        router.model_list[0], {}, "max_parallel_requests"
    )
    assert cap_semaphore is not None
    await cap_semaphore.acquire()  # saturate cap-deployment
    assert cap_semaphore.locked()

    deployment = await router.async_get_available_deployment(
        model="test-group",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )
    # the saturated deployment must be skipped; least-busy picks the free one
    assert deployment["model_info"]["id"] == "free-deployment"

    # when everything is saturated, selection still returns a deployment (the
    # call-time acquire is what raises MaxParallelRequestsError -> fallback)
    free_semaphore = router._get_client(
        router.model_list[1], {}, "max_parallel_requests"
    )
    await free_semaphore.acquire()
    deployment = await router.async_get_available_deployment(
        model="test-group",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )
    assert deployment["model_info"]["id"] in ("cap-deployment", "free-deployment")


@pytest.mark.asyncio
async def test_mpr_slot_acquire_raises_on_saturation():
    """
    _acquire_mpr_slot_or_raise must acquire without blocking while a slot is
    free, and raise MaxParallelRequestsError (a RateLimitError subclass, so the
    fallback machinery picks it up) once the semaphore is saturated.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "test-group",
                "litellm_params": {
                    "model": "openai/cap-model",
                    "max_parallel_requests": 1,
                },
                "model_info": {"id": "cap-deployment"},
            }
        ],
    )
    deployment = router.model_list[0]
    semaphore = router._get_client(deployment, {}, "max_parallel_requests")
    assert semaphore is not None

    # free slot -> acquires cleanly
    await router._acquire_mpr_slot_or_raise(semaphore, deployment, {})
    assert semaphore.locked()

    # saturated -> raises instead of blocking
    with pytest.raises(litellm.MaxParallelRequestsError) as exc_info:
        await router._acquire_mpr_slot_or_raise(semaphore, deployment, {})
    assert isinstance(exc_info.value, litellm.RateLimitError)
    assert exc_info.value.status_code == 429

    # after a release a slot is available again
    semaphore.release()
    await router._acquire_mpr_slot_or_raise(semaphore, deployment, {})
    semaphore.release()

    # the guard releases its slot on normal exit and on body exceptions
    async with router._mpr_capacity_guard(semaphore, deployment, {}):
        assert semaphore.locked()
    assert not semaphore.locked()

    # uncapped deployments (None semaphore) just pass through
    await router._acquire_mpr_slot_or_raise(None, deployment, {})
    async with router._mpr_capacity_guard(None, deployment, {}):
        pass


@pytest.mark.asyncio
async def test_mpr_saturation_triggers_default_fallbacks(monkeypatch):
    """
    When the only deployment in a model group is at max_parallel_requests
    capacity, the call-time acquire raises MaxParallelRequestsError and the
    router's default_fallbacks divert the request to another model group.
    """
    called_models = []

    async def _fake_acompletion(**kwargs):
        called_models.append(kwargs.get("model"))
        response = litellm.ModelResponse(model=kwargs.get("model", ""))
        response.choices = [
            litellm.Choices(
                message=litellm.Message(role="assistant", content="ok"),
                finish_reason="stop",
                index=0,
            )
        ]
        return response

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    router = litellm.Router(
        routing_strategy="simple-shuffle",
        num_retries=0,
        default_fallbacks=["fb-group"],
        model_list=[
            {
                "model_name": "cap-group",
                "litellm_params": {
                    "model": "openai/cap-model",
                    "provider_max_parallel_requests": 1,
                    "api_base": "https://cap.example.com/v1",
                    "api_key": "sk-cap",
                },
                "model_info": {"id": "cap-deployment"},
            },
            {
                "model_name": "fb-group",
                "litellm_params": {
                    "model": "openai/fb-model",
                    "api_base": "https://fb.example.com/v1",
                    "api_key": "sk-fb",
                },
                "model_info": {"id": "fb-deployment"},
            },
        ],
    )

    cap_semaphore = router._get_client(
        router.model_list[0], {}, "max_parallel_requests"
    )
    assert cap_semaphore is not None
    await cap_semaphore.acquire()  # saturate the cap-group deployment

    response = await router.acompletion(
        model="cap-group",
        messages=[{"role": "user", "content": "hi"}],
    )

    # the request was diverted to the fallback group's deployment
    assert called_models == ["openai/fb-model"]
    assert response.choices[0].message.content == "ok"
    assert "x-litellm-attempted-fallbacks" in response._hidden_params.get(
        "additional_headers", {}
    )


async def _handle_router_calls(router):
    pre_fill = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc ut finibus massa. Quisque a magna magna. Quisque neque diam, varius sit amet tellus eu, elementum fermentum sapien. Integer ut erat eget arcu rutrum blandit. Morbi a metus purus. Nulla porta, urna at finibus malesuada, velit ante suscipit orci, vitae laoreet dui ligula ut augue. Cras elementum pretium dui, nec luctus nulla aliquet ut. Nam faucibus, diam nec semper interdum, nisl nisi viverra nulla, vitae sodales elit ex a purus. Donec tristique malesuada lobortis. Donec posuere iaculis nisl, vitae accumsan libero dignissim dignissim. Suspendisse finibus leo et ex mattis tempor. Praesent at nisl vitae quam egestas lacinia. Donec in justo non erat aliquam accumsan sed vitae ex. Vivamus gravida diam vel ipsum tincidunt dignissim.

    Cras vitae efficitur tortor. Curabitur vel erat mollis, euismod diam quis, consequat nibh. Ut vel est eu nulla euismod finibus. Aliquam euismod at risus quis dignissim. Integer non auctor massa. Nullam vitae aliquet mauris. Etiam risus enim, dignissim ut volutpat eget, pulvinar ac augue. Mauris elit est, ultricies vel convallis at, rhoncus nec elit. Aenean ornare maximus orci, ut maximus felis cursus venenatis. Nulla facilisi.

    Maecenas aliquet ante massa, at ullamcorper nibh dictum quis. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Quisque id egestas justo. Suspendisse fringilla in massa in consectetur. Quisque scelerisque egestas lacus at posuere. Vestibulum dui sem, bibendum vehicula ultricies vel, blandit id nisi. Curabitur ullamcorper semper metus, vitae commodo magna. Nulla mi metus, suscipit in neque vitae, porttitor pharetra erat. Vestibulum libero velit, congue in diam non, efficitur suscipit diam. Integer arcu velit, fermentum vel tortor sit amet, venenatis rutrum felis. Donec ultricies enim sit amet iaculis mattis.

    Integer at purus posuere, malesuada tortor vitae, mattis nibh. Mauris ex quam, tincidunt et fermentum vitae, iaculis non elit. Nullam dapibus non nisl ac sagittis. Duis lacinia eros iaculis lectus consectetur vehicula. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Interdum et malesuada fames ac ante ipsum primis in faucibus. Ut cursus semper est, vel interdum turpis ultrices dictum. Suspendisse posuere lorem et accumsan ultrices. Duis sagittis bibendum consequat. Ut convallis vestibulum enim, non dapibus est porttitor et. Quisque suscipit pulvinar turpis, varius tempor turpis. Vestibulum semper dui nunc, vel vulputate elit convallis quis. Fusce aliquam enim nulla, eu congue nunc tempus eu.

    Nam vitae finibus eros, eu eleifend erat. Maecenas hendrerit magna quis molestie dictum. Ut consequat quam eu massa auctor pulvinar. Pellentesque vitae eros ornare urna accumsan tempor. Maecenas porta id quam at sodales. Donec quis accumsan leo, vel viverra nibh. Vestibulum congue blandit nulla, sed rhoncus libero eleifend ac. In risus lorem, rutrum et tincidunt a, interdum a lectus. Pellentesque aliquet pulvinar mauris, ut ultrices nibh ultricies nec. Mauris mi mauris, facilisis nec metus non, egestas luctus ligula. Quisque ac ligula at felis mollis blandit id nec risus. Nam sollicitudin lacus sed sapien fringilla ullamcorper. Etiam dui quam, posuere sit amet velit id, aliquet molestie ante. Integer cursus eget sapien fringilla elementum. Integer molestie, mi ac scelerisque ultrices, nunc purus condimentum est, in posuere quam nibh vitae velit.
    """
    completion = await router.acompletion(
        "gpt-3.5-turbo",
        [
            {
                "role": "user",
                # Fixed speed (was random.random()*100) so the request body is
                # deterministic and the VCR cassette replays instead of
                # appending a new episode every run. This is a rate-limiting
                # test; the prompt content is irrelevant to what it asserts.
                "content": f"{pre_fill * 3}\n\nRecite the Declaration of independence at a speed of 50.0 words per minute.",
            }
        ],
        stream=True,
        temperature=0.0,
        stream_options={"include_usage": True},
    )

    async for chunk in completion:
        pass
    print("done", chunk)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY")
async def test_max_parallel_requests_rpm_rate_limiting():
    """
    - make sure requests > model limits are retried successfully.
    """
    from litellm import Router

    router = Router(
        routing_strategy="usage-based-routing-v2",
        enable_pre_call_checks=True,
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.0,
                    "rpm": 1,
                    "num_retries": 3,
                },
            }
        ],
    )
    await asyncio.gather(*[_handle_router_calls(router) for _ in range(3)])


@pytest.mark.asyncio
async def test_max_parallel_requests_tpm_rate_limiting_base_case():
    """
    - check error raised if defined tpm limit crossed.
    """
    from litellm import Router, token_counter

    _messages = [{"role": "user", "content": "Hey, how's it going?"}]
    router = Router(
        routing_strategy="usage-based-routing-v2",
        enable_pre_call_checks=True,
        model_list=[
            {
                "model_name": "gpt-4o-2024-08-06",
                "litellm_params": {
                    "model": "gpt-4o-2024-08-06",
                    "temperature": 0.0,
                    "tpm": 1,
                },
            }
        ],
        num_retries=0,
    )

    async def _exceed_limit():
        for _ in range(2):
            await router.acompletion(
                model="gpt-4o-2024-08-06",
                messages=_messages,
            )

    with pytest.raises(litellm.RateLimitError):
        await _exceed_limit()
