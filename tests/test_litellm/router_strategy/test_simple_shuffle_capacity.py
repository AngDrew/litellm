import asyncio
from unittest.mock import patch

import pytest

from litellm import Router


def _router() -> Router:
    return Router(
        routing_strategy="simple-shuffle",
        model_list=[
            {
                "model_name": "public-alias",
                "litellm_params": {
                    "model": "openai/first",
                    "api_base": "https://first.invalid",
                    "api_key": "test-key",
                    "max_parallel_requests": 1,
                    "weight": 2,
                },
                "model_info": {"id": "first"},
            },
            {
                "model_name": "public-alias",
                "litellm_params": {
                    "model": "openai/second",
                    "api_base": "https://second.invalid",
                    "api_key": "test-key",
                    "max_parallel_requests": 1,
                    "weight": 1,
                },
                "model_info": {"id": "second"},
            },
        ],
    )


def _deployments(router: Router) -> list[dict]:
    return router.get_model_list(model_name="public-alias")


async def _semaphore(router: Router, deployment: dict) -> asyncio.Semaphore:
    semaphore = router._get_client(
        deployment=deployment,
        kwargs={},
        client_type="max_parallel_requests",
    )
    assert isinstance(semaphore, asyncio.Semaphore)
    return semaphore


@pytest.mark.asyncio
async def test_simple_shuffle_skips_saturated_deployment() -> None:
    router = _router()
    first, second = _deployments(router)
    first_semaphore = await _semaphore(router, first)
    await first_semaphore.acquire()

    selected = await router._select_available_simple_shuffle_deployment(
        [first, second], "public-alias", {"model": "public-alias"}
    )

    assert selected["model_info"]["id"] == "second"
    first_semaphore.release()


@pytest.mark.asyncio
async def test_simple_shuffle_waits_for_any_saturated_deployment_to_free() -> None:
    router = _router()
    first, second = _deployments(router)
    first_semaphore = await _semaphore(router, first)
    second_semaphore = await _semaphore(router, second)
    await first_semaphore.acquire()
    await second_semaphore.acquire()

    selection = asyncio.create_task(
        router._select_available_simple_shuffle_deployment([first, second], "public-alias", {"model": "public-alias"})
    )
    await asyncio.sleep(0)
    assert not selection.done()

    second_semaphore.release()
    selected = await asyncio.wait_for(selection, timeout=0.1)
    assert selected["model_info"]["id"] == "second"
    first_semaphore.release()


@pytest.mark.asyncio
async def test_simple_shuffle_preserves_available_deployment_weights() -> None:
    router = _router()
    deployments = _deployments(router)

    with patch("litellm.router_strategy.simple_shuffle.random.choices", return_value=[0]) as choices:
        selected = await router._select_available_simple_shuffle_deployment(
            deployments, "public-alias", {"model": "public-alias"}
        )

    assert selected["model_info"]["id"] == "first"
    assert choices.call_args.kwargs["weights"] == [2 / 3, 1 / 3]
    assert router.routing_strategy == "simple-shuffle"
