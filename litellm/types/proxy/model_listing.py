"""Response types for the model listing/retrieve endpoints (/v1/models, /models)."""

from typing import Literal

from typing_extensions import NotRequired, TypedDict


class ModelInfoMetadata(TypedDict):
    fallbacks: list[str]


class ModelInfoResponse(TypedDict):
    """OpenAI-compatible model object. `mode`, `max_input_tokens`, and
    `max_output_tokens` are attached when the cost map knows them; `metadata`
    is present only when the endpoint is called with include_metadata=true.

    Any `supports_*` boolean the cost map defines for the model is also copied
    through verbatim (e.g. `supports_vision`, `supports_function_calling`,
    `supports_reasoning`) so clients can auto-detect model capabilities. The
    set is whatever the cost map holds for that model, so it is modeled here
    as extra keys rather than a fixed list.
    """

    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    mode: NotRequired[str]
    max_input_tokens: NotRequired[int]
    max_output_tokens: NotRequired[int]
    metadata: NotRequired[ModelInfoMetadata]
