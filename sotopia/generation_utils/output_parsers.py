import json
import re
from typing import Generic, Type, TypeVar, Optional, Any
from pydantic import BaseModel, Field
import json_repair

from sotopia.database import LLMBaseModel

OutputType = TypeVar("OutputType", bound=object)
T = TypeVar("T", bound=BaseModel)


class EnvResponse(LLMBaseModel):
    reasoning: str = Field(
        description="first reiterate agents' social goals and then reason about what agents say/do and whether that aligns with their goals."
    )
    p1_rate: int = Field(description="rating of participant 1, on the scale of 0 to 9")
    p2_rate: int = Field(description="rating of participant 2, on the scale of 0 to 9")


class OutputParser(LLMBaseModel, Generic[OutputType]):
    def parse(self, result: str) -> OutputType:
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        raise NotImplementedError


_SCHEMA_VALUE_KEYS: frozenset[str] = frozenset(
    {"type", "description", "title", "default", "$ref", "anyOf", "items", "properties"}
)


def _looks_like_schema_echo(obj: Any) -> bool:
    """启发式检测：LLM 把 JSON Schema 当 value 回写。

    现象 1（root-level）：``{"properties": {...}, "required": [...], "type": "object"}``
    现象 2（properties-only）：``{"first_name": {"type": "string", ...}, ...}`` —— 每个
    value 是 schema 片段（带 ``type`` / ``description`` / ``$ref`` / ``anyOf`` 等）。

    返回 ``True`` 时调用方应抛错，让 ``format_bad_output`` 用更短的 fix-prompt 重写。
    """
    if not isinstance(obj, dict) or not obj:
        return False
    if obj.keys() & {"properties", "$defs", "$ref"} and obj.get("type") == "object":
        return True
    schema_like = 0
    for v in obj.values():
        if isinstance(v, dict) and v.keys() & _SCHEMA_VALUE_KEYS:
            schema_like += 1
    # 至少 60% 的 value 像 schema 片段才认定是回写（避免误伤真有 ``type``/``description`` 字段的合法 payload）。
    return schema_like >= max(2, int(0.6 * len(obj)))


class PydanticOutputParser(OutputParser[T], Generic[T]):
    pydantic_object: Type[T]

    def parse(self, result: str, context: dict[str, Any] | None = None) -> T:
        # Strip markdown code blocks if present
        result = result.strip()
        # Remove the ```json and ``` if both are present
        result = re.sub(r"^```json\s*", "", result).strip(" \n")

        json_result = json_repair.loads(result)
        assert isinstance(json_result, dict)

        # Handle nested type-value structure
        def extract_value(obj: dict[str, Any] | list[Any] | str) -> Any:
            if isinstance(obj, dict):
                if "value" in obj:
                    return obj["value"]
                return {k: extract_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [extract_value(item) for item in obj]
            return obj

        json_result = extract_value(json_result)

        # Schema-echo guard：若 LLM 把 JSON Schema 当 value 回写，``model_validate_json`` 会抛
        # 一长串 ``string_type`` / ``int_type`` 错误。提前抛 ValueError 让 ``format_bad_output``
        # 接力做 repair（fix-prompt 通常更短，对小模型友好）。
        if _looks_like_schema_echo(json_result):
            raise ValueError(
                f"LLM returned a JSON schema instead of values for "
                f"{self.pydantic_object.__name__}; will trigger format_bad_output repair."
            )
        if isinstance(json_result, dict) and "properties" in json_result:
            inner = json_result["properties"]
            if context is not None:
                return self.pydantic_object.model_validate(inner, context=context)
            return self.pydantic_object.model_validate_json(json.dumps(inner))
        else:
            data = json_result

        # Use model_validate with context if provided, otherwise use model_validate_json for backward compatibility
        if context is not None:
            return self.pydantic_object.model_validate(data, context=context)
        else:
            # Fallback to JSON validation for backward compatibility
            # Type narrowing: check that json_result is a dict before accessing "properties"
            if isinstance(json_result, dict) and "properties" in json_result:
                return self.pydantic_object.model_validate_json(
                    json.dumps(json_result["properties"])
                )
            else:
                return self.pydantic_object.model_validate_json(result)

    def get_format_instructions(self) -> str:
        return json.dumps(self.pydantic_object.model_json_schema())


class EnvResponsePydanticOutputParser(PydanticOutputParser[EnvResponse]):
    def __init__(self, pydantic_object: Type[EnvResponse] = EnvResponse) -> None:
        super(EnvResponsePydanticOutputParser, self).__init__(
            pydantic_object=pydantic_object
        )

    def parse(self, text: str, context: dict[str, Any] | None = None) -> EnvResponse:
        # remove trailing commas before ) or ] from text
        text = re.sub(r",\s*(\)|\])", r"\1", text)
        response = super().parse(text, context=context)
        if isinstance(response, EnvResponse):
            return response
        else:
            raise ValueError(f"Expected EnvResponse, got {type(response)}")

    def get_format_instructions(self) -> str:
        format_instruction = super().get_format_instructions()
        return format_instruction


class StrOutputParser(OutputParser[str]):
    def parse(self, result: str) -> str:
        return result

    def get_format_instructions(self) -> str:
        return ""


class ScriptOutputParser(OutputParser[str]):
    def parse(self, result: str) -> str:
        return result

    def get_format_instructions(self) -> str:
        return ""


class ListOfIntOutputParser(OutputParser[list[int]]):
    number_of_int: Optional[int] = None
    range_of_int: Optional[tuple[int, int]] = None

    def __init__(
        self,
        number_of_int: Optional[int] = None,
        range_of_int: Optional[tuple[int, int]] = None,
    ):
        """
        Parse the output to a list of integers

        Args:
            number_of_int (int | None): The number of integers in the output. If None, the number of integers is not fixed.
        """
        super().__init__()
        self.number_of_int = number_of_int
        self.range_of_int = range_of_int

    def _get_description_text(self) -> str:
        # 中文注释：该描述会直接拼进提示词，强约束模型输出“仅空格分隔整数”。
        return f"a list of{' ' + str(self.number_of_int) if self.number_of_int else ''} intergers{' within the range of' + str(self.range_of_int) if self.range_of_int else ''} separated by spaces. Don't output anything else. Format example: 1 2 3 4 5"

    def get_format_instructions(self) -> str:
        return "Please output " + self._get_description_text()

    def parse(self, output: str) -> list[int]:
        try:
            # 中文注释：这里采用最严格解析：按空格切分并逐个 int 转换。
            output_loaded = output.split(" ")
            result = [int(x) for x in output_loaded]
            if self.number_of_int and len(result) != self.number_of_int:
                msg = f"Expect {self.number_of_int} integers, got {len(result)}"
                raise ValueError(msg)
            if self.range_of_int:
                # 中文注释：可选范围校验，避免模型给出超界值。
                for x in result:
                    if x < self.range_of_int[0] or x > self.range_of_int[1]:
                        msg = f"Expect integers within the range of {self.range_of_int}, got {result}"
                        raise ValueError(msg)
            return result
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            msg = f"Exception {e}: the output format is not correct. Expect {self._get_description_text()}, got {output}"
            raise ValueError(msg)

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "list[int]"
