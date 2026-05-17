import logging
import os
from litellm import acompletion
from litellm.utils import supports_response_schema
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from typing import Any

import gin

from pydantic import BaseModel, validate_call
from rich import print
from rich.logging import RichHandler

from sotopia.database import (
    BaseEnvironmentProfile,
    EnvironmentProfile,
    LLMEvalBaseModel,
    RelationshipProfile,
)
from sotopia.messages import ActionType, AgentAction, ScriptBackground
from sotopia.messages.message_classes import (
    ScriptInteraction,
    ScriptInteractionReturnType,
)
from sotopia.utils import format_docstring, truncate_chars



from sotopia.generation_utils.output_parsers import (
    OutputParser,
    PydanticOutputParser,
    StrOutputParser,
    OutputType,
    EnvResponse,
    ScriptOutputParser,
)

# Configure logger
log = logging.getLogger("sotopia.generation")
log.setLevel(logging.DEBUG)

# Create console handler with rich formatting
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)


def fill_template(template: str, **kwargs: str) -> str:
    """Fill template with kwargs, ignoring missing keys."""
    for k, v in kwargs.items():
        template = template.replace(f"{{{k}}}", v)
    return template


def _substituted_template_length(template: str, iv: dict[str, Any]) -> int:
    t = template
    for key, value in iv.items():
        t = t.replace(f"{{{key}}}", str(value))
    return len(t)


def _clamp_input_values_to_max_rendered(
    template: str,
    input_values: dict[str, Any],
    *,
    max_rendered_chars: int,
) -> dict[str, Any]:
    """当拼接后 user 文本将超过 ``max_rendered_chars`` 时，按优先级对大块字段做 ``truncate_chars``。"""
    if max_rendered_chars <= 0:
        return dict(input_values)

    # 不收缩 schema 指令，避免结构化输出解析失败；超长时依赖最终硬截断。
    protected = frozenset({"format_instructions"})

    iv: dict[str, Any] = {str(k): str(v) for k, v in input_values.items()}
    priority = (
        "history",
        "background",
        "goal",
        "examples",
        "inspiration_prompt",
        "agent_profile",
        "text",
    )
    safety = 0
    while _substituted_template_length(template, iv) > max_rendered_chars and safety < 200:
        safety += 1
        cur = _substituted_template_length(template, iv)
        over = cur - max_rendered_chars
        candidates = [k for k in priority if k in iv and k not in protected and len(iv[k]) > 512]
        tail = [k for k in iv if k not in candidates and k not in protected and len(iv[k]) > 512]
        tail.sort(key=lambda k: len(iv[k]), reverse=True)
        ordered = candidates + tail
        if not ordered:
            break
        key = ordered[0]
        target = max(512, len(iv[key]) - max(over + 2048, len(iv[key]) // 5))
        iv[key] = truncate_chars(iv[key], target)
    return iv


# Add handler to logger
log.addHandler(console_handler)

# subject to future OpenAI changes
DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_TEMPERATURE = 0.7


@validate_call
async def format_bad_output(
    ill_formed_output: str,
    output_parser: OutputParser[OutputType],
    model_name: str,
    use_fixed_model_version: bool = True,
    base_url: str | None = None,
) -> str:
    format_instructions = output_parser.get_format_instructions()
    template = """
    The previous output could not be parsed as JSON, OR it was a JSON Schema instead of an
    actual value object. Please rewrite it as a single VALID JSON OBJECT WITH FILLED-IN
    VALUES that conforms to the target schema (do NOT echo the schema; do NOT include
    ``description`` / ``type`` / ``properties`` / ``$defs`` keys at any level; do NOT wrap in
    markdown). For optional fields, supply concrete values (or empty strings / empty lists)
    rather than schema fragments.

    Original output (often the JSON schema mistakenly echoed back):
    {ill_formed_output}

    Target schema (reference only — fill in values, do not copy this back as the answer):
    {format_instructions}

    Output the cleaned JSON value object now:
    """

    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    content = template.format(**input_values)
    if isinstance(output_parser, PydanticOutputParser):
        response_format = _build_json_schema_response_format(
            output_parser.pydantic_object,
        )
        response = await acompletion(
            model=model_name,
            response_format=response_format,
            messages=[{"role": "user", "content": content}],
        )
    else:
        response = await acompletion(
            model=model_name,
            messages=[{"role": "user", "content": content}],
        )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Response content is None")
    return content


def _sanitize_schema_name(name: str) -> str:
    """
    Sanitize schema title to match OpenAI's naming pattern.

    OpenAI requires schema names to match: ^[a-zA-Z0-9_-]+$
    Replaces invalid characters with underscores.

    Args:
        name: Original schema title (may contain brackets, spaces, etc.)

    Returns:
        Sanitized name with only alphanumeric, underscore, and hyphen

    Example:
        >>> _sanitize_schema_name("Response[AgentAction]")
        'Response_AgentAction_'
    """
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def _is_response_format_unavailable_error(exc: BaseException) -> bool:
    """检测 API 不支持 structured output / response_format 的错误。"""
    msg = str(exc).lower()
    return (
        "response_format" in msg
        and ("unavailable" in msg or "not supported" in msg or "unsupported" in msg)
    ) or "response_format type is unavailable" in str(exc)


def _build_json_schema_response_format(
    pydantic_class: type[BaseModel],
) -> dict[str, Any]:
    """
    Build complete OpenAI response_format dict for structured output.

    Coordinates schema validation, fixing, and formatting into the
    structure expected by litellm's acompletion with json_schema mode.

    Args:
        schema: Raw JSON schema from model_json_schema()
        pydantic_class: Original Pydantic class (for name extraction)

    Returns:
        Complete response_format dict with type, json_schema, name, schema, strict
    """
    # 中文注释：先拿到 Pydantic schema，并清洗名称，避免不符合 OpenAI 命名约束
    #（例如包含 []、空格等字符）。
    # Sanitize the schema name
    schema = pydantic_class.model_json_schema()
    original_name = schema.get("title", pydantic_class.__name__)
    sanitized_name = _sanitize_schema_name(original_name)

    # Determine whether to use strict mode
    # LLMEvalBaseModel and its subclasses (like EvaluationForAgents) use dict[str, T]
    # which requires additionalProperties with a schema (for dynamic keys).
    # OpenAI's strict mode only allows additionalProperties: false, so we disable
    # strict mode for these models.
    #
    # ``AgentAction`` uses ``argument: str | dict[str, Any]``; the object branch yields
    # ``anyOf`` JSON Schema entries that violate strict ``additionalProperties: false``
    # on common OpenAI-compatible gateways (liteLLM -> proxy). Disable strict here too.
    #
    # ``BaseEnvironmentProfile.game_metadata`` is ``dict[str, Any] | None``; Pydantic v2
    # emits ``anyOf`` object branches without ``additionalProperties: false``, which
    # OpenAI rejects for ``json_schema`` strict mode (seen on repair path in
    # ``format_bad_output`` after a marginal parse failure). Disable strict for env profiles.
    use_strict = not issubclass(pydantic_class, LLMEvalBaseModel)
    if pydantic_class is AgentAction:
        use_strict = False
    if issubclass(pydantic_class, BaseEnvironmentProfile):
        use_strict = False
    # Generic opt-out: any Pydantic class that sets the class attribute
    # ``OPENAI_DISABLE_STRICT_JSON_SCHEMA = True`` (e.g. via a ``ClassVar``) will
    # be sent with ``strict=False``. This is the right escape hatch for models
    # whose schema would otherwise need every property required (forbidden by
    # strict OpenAI gateways) or whose ``anyOf``/dynamic dict branches lack
    # ``additionalProperties: false``. Used by ``LLMNegotiationAgentDraft``.
    if getattr(pydantic_class, "OPENAI_DISABLE_STRICT_JSON_SCHEMA", False):
        use_strict = False

    # 中文注释：返回 litellm/openai 期望的 response_format 结构，
    # 后续会直接传给 acompletion(..., response_format=...).
    # Build response format
    return {
        "type": "json_schema",
        "json_schema": {
            "name": sanitized_name,
            "schema": schema,
            "strict": use_strict,
        },
    }


@gin.configurable
@validate_call
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: float | None = DEFAULT_TEMPERATURE,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    context: dict[str, Any] | None = None,
) -> OutputType:
    """
    Generate text using LiteLLM with optional structured output support.

    This function handles template formatting, temperature management, schema validation,
    and API calls. It supports both structured (JSON schema) and standard text generation modes.

    Args:
        model_name: Model identifier (e.g., "gpt-4o-mini" or "custom/model@http://...")
        template: Template string with {variable} placeholders
        input_values: Dict mapping template variables to their values
        output_parser: Parser to convert raw output to desired type
        temperature: Temperature value (float or None). Defaults to DEFAULT_TEMPERATURE.
                     None means use provider default. With drop_params=True, litellm
                     automatically handles unsupported parameters.
        structured_output: If True, use JSON schema mode (requires PydanticOutputParser)
        bad_output_process_model: Model to use for reformatting bad outputs (if needed)
        use_fixed_model_version: Whether to use fixed model versioning
        context: Optional context dict passed to output parser

    Oversized user prompt (automatic):

        If ``SOTOPIA_MAX_RENDERED_USER_CHARS`` is set (default ``180000``; ``0`` / ``off`` / ``none``
        disables), ``agenerate`` **before** ``acompletion`` shrinks large ``input_values`` fields in a
        fixed priority (``history``, ``background``, ``goal``, …) using ``truncate_chars``, then applies
        a hard cap on the final rendered user string.
        Parsed output of type OutputType

    Example:
        >>> from sotopia.generation_utils.output_parsers import PydanticOutputParser
        >>> from pydantic import BaseModel
        >>> class Response(BaseModel):
        ...     text: str
        >>> result = await agenerate(
        ...     model_name="gpt-4o-mini",
        ...     template="Say hello to {name}",
        ...     input_values={"name": "Alice"},
        ...     output_parser=PydanticOutputParser(pydantic_object=Response),
        ...     structured_output=True,
        ... )
    """
    # 中文注释：构造“坏输出修复模型”，当主模型返回不可解析文本时用于二次修复。
    # Format template with input values
    bad_output_process_model = (
        bad_output_process_model or DEFAULT_BAD_OUTPUT_PROCESS_MODEL
    )
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()

    # 中文注释：统一做文档字符串去缩进，减少提示词格式噪声。
    # Process template
    template = format_docstring(template)

    raw_cap = os.environ.get("SOTOPIA_MAX_RENDERED_USER_CHARS", "180000").strip().lower()
    try:
        max_render = 0 if raw_cap in ("0", "", "none", "off", "disabled") else int(raw_cap)
    except ValueError:
        log.warning(
            "agenerate: invalid SOTOPIA_MAX_RENDERED_USER_CHARS=%r; using default 180000",
            os.environ.get("SOTOPIA_MAX_RENDERED_USER_CHARS"),
        )
        max_render = 180_000
    if max_render > 0:
        pre_len = _substituted_template_length(template, {k: str(v) for k, v in input_values.items()})
        input_values = _clamp_input_values_to_max_rendered(
            template,
            dict(input_values),
            max_rendered_chars=max_render,
        )
        post_len = _substituted_template_length(template, {k: str(v) for k, v in input_values.items()})
        if post_len < pre_len - 500:
            log.warning(
                "agenerate: auto-shrank template-bound fields %s -> %s chars (cap=%s via SOTOPIA_MAX_RENDERED_USER_CHARS)",
                pre_len,
                post_len,
                max_render,
            )

    # 中文注释：把模板变量替换成具体上下文。
    # Replace template variables
    for key, value in input_values.items():
        template = template.replace(f"{{{key}}}", str(value))

    if max_render > 0 and len(template) > max_render:
        log.warning(
            "agenerate: hard-capping rendered user message %s -> %s chars",
            len(template),
            max_render,
        )
        template = truncate_chars(template, max_render)

    # 中文注释：支持 custom/model@base_url 语法，
    # 便于接入 OpenAI-compatible 的第三方网关。
    if model_name.startswith("custom"):
        base_url, api_key = (
            model_name.split("@")[1],
            os.environ.get("CUSTOM_API_KEY", "EMPTY"),
        )
        model_name = model_name.split("@")[0].replace("custom/", "openai/")
    else:
        base_url = None
        api_key = None

    temperature_value: float | None = temperature

    supported_params: list[str] | None = None
    if base_url is None:
        supported_params = get_supported_openai_params(model=model_name)

    messages = [{"role": "user", "content": template}]
    if structured_output:
        # 中文注释：结构化输出分支，强制模型按 JSON Schema 返回。
        if not base_url:
            assert supported_params is not None
            assert (
                "response_format" in supported_params
            ), "response_format is not supported in this model"
            assert supports_response_schema(
                model=model_name
            ), "response_schema is not supported in this model"

        assert isinstance(
            output_parser, PydanticOutputParser
        ), "structured output only supported in PydanticOutputParser"

        response_format = _build_json_schema_response_format(
            output_parser.pydantic_object
        )

        # 中文注释：drop_params=True 会自动忽略不支持的参数，
        # 降低跨 provider 的兼容成本。
        # Build completion kwargs with structured output
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "response_format": response_format,
            "drop_params": True,  # litellm automatically drops unsupported params
            "base_url": base_url,
            "api_key": api_key,
        }
        if temperature_value is not None:
            completion_kwargs["temperature"] = temperature_value
        # if "deepseek" in model_name:
        #     completion_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = await acompletion(**completion_kwargs)
    else:
        # 中文注释：普通文本输出分支（不带 response_format）。
        # Build completion kwargs for standard output
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "drop_params": True,  # litellm automatically drops unsupported params
            "base_url": base_url,
            "api_key": api_key,
        }
        if temperature_value is not None:
            completion_kwargs["temperature"] = temperature_value
        # if "deepseek" in model_name:
        #     completion_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        response = await acompletion(**completion_kwargs)
    result = response.choices[0].message.content

    # 中文注释：只有 Pydantic 解析器支持 context（例如 action 校验上下文）。
    # Only PydanticOutputParser supports context parameter
    parse_kwargs = (
        {"context": context} if isinstance(output_parser, PydanticOutputParser) else {}
    )

    first_raw = result if isinstance(result, str) else ("" if result is None else str(result))
    repaired_raw: str | None = None
    try:
        parsed_result = output_parser.parse(result, **parse_kwargs)
    except Exception:
        # 中文注释：一次解析失败后，尝试“先修格式再解析”，提高鲁棒性。
        reformat_result = await format_bad_output(
            result,
            output_parser,
            bad_output_process_model or model_name,
            use_fixed_model_version,
            base_url=base_url,
        )
        repaired_raw = (
            reformat_result if isinstance(reformat_result, str) else str(reformat_result)
        )
        parsed_result = output_parser.parse(reformat_result, **parse_kwargs)

    # Include agent name in logs if available
    agent_name = input_values.get("agent", "")
    log_prefix = f" [{agent_name}]" if agent_name else ""
    log.debug(f"Model: {model_name}")
    # breakpoint()
    prompt = '\n'.join([f"{m['role']}: {m['content']}" for m in messages])
    log.debug(f"Prompt: {prompt}\n\n\n\n\n\nresponse: {result}\n")
    # log.info(f"Generated result{log_prefix}: {parsed_result}")
    try:
        from sotopia.settings.long_term_negotiation.model_trace import record_generation_step

        record_generation_step(
            step_kind="agenerate",
            model_name=model_name,
            messages=messages,
            raw_content=first_raw,
            parsed=parsed_result,
            input_values=dict(input_values),
            full_rendered_prompt=template,
            raw_model_content_repaired=repaired_raw,
            generation_meta={
                "temperature": temperature_value,
                "structured_output": bool(structured_output),
            },
        )
    except Exception:
        pass
    return parsed_result


@gin.configurable
@validate_call
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: float | None = DEFAULT_TEMPERATURE,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> EnvironmentProfile:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
        Examples:
        {examples}
        Inspirational prompt: {inspiration_prompt}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            inspiration_prompt=inspiration_prompt,
            examples=examples,
        ),
        output_parser=PydanticOutputParser(pydantic_object=EnvironmentProfile),
        temperature=temperature,
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[RelationshipProfile, str]:
    """
    Using langchain to generate the background
    """
    agent_profile = "\n".join(agents_profiles)
    return await agenerate(
        model_name=model_name,
        template="""Please generate relationship between two agents based on the agents' profiles below. Note that you generate
        {agent_profile}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            agent_profile=agent_profile,
        ),
        output_parser=PydanticOutputParser(pydantic_object=RelationshipProfile),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@gin.configurable
@validate_call
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    temperature: float | None = DEFAULT_TEMPERATURE,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    custom_template: str | None = None,
    structured_output: bool = False,
    agent_names: list[str] | None = None,
    sender: str | None = None,
) -> AgentAction:
    """
    Using langchain to generate an example episode
    """
    try:
        if custom_template:
            if script_like:
                raise ValueError(
                    "script_like and custom_template are mutually exclusive"
                )
            template = custom_template
        elif script_like:
            # model as playwright
            template = """
                Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.
                You can find {agent}'s background and goal in the 'Here is the context of the interaction' field.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                The script has proceeded to Turn #{turn_number}. Current available action types are
                {action_list}.
                Note: The script can be ended if 1. one agent have achieved social goals, 2. this conversation makes the agent uncomfortable, 3. the agent find it uninteresting/you lose your patience, 4. or for other reasons you think it should stop.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """
        else:
            # Normal case, model as agent
            template = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """
        # Build validation context if agent_names provided
        validation_context = None
        if agent_names is not None:
            validation_context = {
                "agent_names": agent_names,
                "available_action_types": action_types,
            }
            if sender is not None:
                validation_context["sender"] = sender

        return await agenerate(
            model_name=model_name,
            template=template,
            input_values=dict(
                agent=agent,
                turn_number=str(turn_number),
                history=history,
                action_list=" ".join(action_types),
                goal=goal,
            ),
            output_parser=PydanticOutputParser(pydantic_object=AgentAction),
            temperature=temperature,
            structured_output=structured_output,
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            context=validation_context,
        )
    except Exception as e:
        # structured output 不受支持时（如 DeepSeek），回退到非 structured 模式重试一次
        if structured_output and _is_response_format_unavailable_error(e):
            # log.warning(
            #     "structured output unavailable for %s; retrying without response_format",
            #     model_name,
            # )
            try:
                return await agenerate_action(
                    model_name=model_name,
                    agent=agent,
                    turn_number=turn_number,
                    history=history,
                    action_types=action_types,
                    goal=goal,
                    agent_names=agent_names,
                    sender=sender,
                    temperature=temperature,
                    structured_output=False,
                    custom_template=custom_template,
                    script_like=script_like,
                    bad_output_process_model=bad_output_process_model,
                    use_fixed_model_version=use_fixed_model_version,
                )
            except Exception as e2:
                # log.warning(f"Failed to generate action (fallback) due to {e2}")
                return AgentAction(action_type="none", argument="", to=[])
        # log.warning(f"Failed to generate action due to {e}")
        return AgentAction(action_type="none", argument="", to=[])


@gin.configurable
@validate_call
async def agenerate_script(
    model_name: str,
    background: ScriptBackground,
    temperature: float | None = DEFAULT_TEMPERATURE,
    agent_names: list[str] = [],
    agent_name: str = "",
    history: str = "",
    single_step: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[ScriptInteractionReturnType, str]:
    """
    Using langchain to generate an the script interactions between two agent
    The script interaction is generated in a single generation process.
    Note that in this case we do not require a json format response,
    so the failure rate will be higher, and it is recommended to use at least llama-2-70b.
    """
    try:
        if single_step:
            return await agenerate(
                model_name=model_name,
                template="""Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.

                Here are the conversation background and history:
                {background}
                {history}

                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.

                Here are the format instructions:
                {format_instructions}""",
                input_values=dict(
                    background=background.to_natural_language(),
                    history=history,
                    agent=agent_name,
                ),
                output_parser=ScriptOutputParser(  # type: ignore[call-arg,arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=True,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )

        else:
            return await agenerate(
                model_name=model_name,
                template="""
                Please write the script between two characters based on their social goals with a maximum of 20 turns.

                {background}
                Your action should follow the given format:
                {format_instructions}
                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.""",
                input_values=dict(
                    background=background.to_natural_language(),
                ),
                output_parser=ScriptOutputParser(  # type: ignore[call-arg,arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=False,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
    except Exception as e:
        # TODO raise(e) # Maybe we do not want to return anything?
        print(f"Exception in agenerate {e}")
        return_default_value: ScriptInteractionReturnType = (
            ScriptInteraction.default_value_for_return_type()
        )
        return (return_default_value, "")


@validate_call
def process_history(
    script: ScriptBackground | EnvResponse | dict[str, AgentAction],
) -> str:
    """
    Format the script background
    """
    result = ""
    if isinstance(script, ScriptBackground | EnvResponse):
        script = script.dict()
        result = "The initial observation\n\n"
    for key, value in script.items():
        if value:
            result += f"{key}: {value} \n"
    return result


@validate_call
async def agenerate_init_profile(
    model_name: str,
    basic_info: dict[str, str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please expand a fictional background for {name}. Here is the basic information:
            {name}'s age: {age}
            {name}'s gender identity: {gender_identity}
            {name}'s pronouns: {pronoun}
            {name}'s occupation: {occupation}
            {name}'s big 5 personality traits: {bigfive}
            {name}'s moral Foundation: think {mft} is more important than others
            {name}'s Schwartz portrait value: {schwartz}
            {name}'s decision-making style: {decision_style}
            {name}'s secret: {secret}
            Include the previous information in the background.
            Then expand the personal backgrounds with concrete details (e.g, look, family, hobbies, friends and etc.)
            For the personality and values (e.g., MBTI, moral foundation, and etc.),
            remember to use examples and behaviors in the person's life to demonstrate it.
            """,
        input_values=dict(
            name=basic_info["name"],
            age=basic_info["age"],
            gender_identity=basic_info["gender_identity"],
            pronoun=basic_info["pronoun"],
            occupation=basic_info["occupation"],
            bigfive=basic_info["Big_Five_Personality"],
            mft=basic_info["Moral_Foundation"],
            schwartz=basic_info["Schwartz_Portrait_Value"],
            decision_style=basic_info["Decision_making_Style"],
            secret=basic_info["secret"],
        ),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def convert_narratives(
    model_name: str,
    narrative: str,
    text: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    if narrative == "first":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a first-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with I, me, my, and mine.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    elif narrative == "second":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a second-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with you, your, and yours.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    else:
        raise ValueError(f"Narrative {narrative} is not supported.")


@validate_call
async def agenerate_goal(
    model_name: str,
    background: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    agent: str | None = None,
) -> str:
    """
    Using langchain to generate the background

    ``agent`` 仅用于 tracing / 元数据（写入 ``input_values``），不参与默认 goal 模板占位符。
    """
    iv: dict[str, Any] = dict(background=background)
    if agent:
        iv["agent"] = agent
    return await agenerate(
        model_name=model_name,
        template="""Please generate your goal based on the background:
            {background}
            """,
        input_values=iv,
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )
