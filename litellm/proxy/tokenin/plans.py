
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger

DEFAULT_PLAN_ID = "medium"

FLAT_TEAM_ID = "ac0b4e54-71a7-4e1f-bfaf-32fad13c09e9"
KEY_MODELS = ["all-team-models"]


class TokeninPlan(BaseModel):
    id: str
    name: str
    kind: str
    rpm_limit: int
    max_parallel_requests: int
    max_budget: float
    budget_duration: str | None = None
    weekly_value: float | None = None
    monthly_value: float | None = None


def load_plans() -> list[TokeninPlan]:
    from litellm.proxy.proxy_server import proxy_config

    config = proxy_config.get_config_state()
    raw = config.get("tokenin_plans") or []
    plans: list[TokeninPlan] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            plans.append(TokeninPlan(**entry))
        except Exception as e:
            verbose_proxy_logger.warning("tokenin: skipping malformed plan %r: %s", entry.get("id"), e)
    return plans


def get_plan_by_id(plan_id: str | None) -> TokeninPlan:
    plans = load_plans()
    if plan_id:
        for p in plans:
            if p.id == plan_id:
                return p
    for p in plans:
        if p.id == DEFAULT_PLAN_ID:
            return p
    if plans:
        return plans[0]
    raise RuntimeError("no tokenin_plans configured in proxy_server_config.yaml")


def is_payg(plan: TokeninPlan) -> bool:
    return plan.kind == "payg"
