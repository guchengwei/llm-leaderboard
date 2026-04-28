import json
import os
import time
import copy

from omegaconf import OmegaConf
from .openai_completion import OpenAICompletionsHandler
from ..model_style import ModelStyle
from ..utils import (
    combine_consecutive_user_prompts,
    func_doc_language_specific_pre_processing,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from overrides import override

try:
    from config_singleton import WandbConfigSingleton
except ImportError:
    WandbConfigSingleton = None


def _to_plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        return OmegaConf.to_container(value, resolve=True) or {}
    except Exception:
        return {}


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class DeepSeekAPIHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI_Completions
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_COMPATIBLE_API_KEY"),
        )
        self.cfg = self._load_cfg()
        self.generator_config = self._load_generator_config()

    @staticmethod
    def _load_cfg():
        if WandbConfigSingleton is None:
            return None
        try:
            return WandbConfigSingleton.get_instance().config
        except Exception:
            return None

    def _load_generator_config(self) -> dict:
        if self.cfg is None:
            return {}
        return _deep_merge_dicts(
            _to_plain_dict(getattr(self.cfg, "generator", {})),
            _to_plain_dict(getattr(getattr(self.cfg, "bfcl", {}), "generator_config", {})),
        )

    def _configured_api_model_name(self) -> str:
        if self.cfg is not None:
            try:
                model_name = str(self.cfg.model.pretrained_model_name_or_path)
                if model_name in {"deepseek-v4-pro", "deepseek-v4-flash"}:
                    return model_name
            except Exception:
                pass

        if "DeepSeek-V4-Pro" in self.model_name:
            return "deepseek-v4-pro"
        if "DeepSeek-V4-Flash" in self.model_name:
            return "deepseek-v4-flash"
        if "DeepSeek-V3" in self.model_name:
            return "deepseek-chat"
        if "DeepSeek-R1" in self.model_name:
            return "deepseek-reasoner"

        raise ValueError(f"Model name {self.model_name} not yet supported")

    def _build_generation_kwargs(self, *, include_tools: bool = False, tools=None) -> dict:
        kwargs = {}
        for key, value in self.generator_config.items():
            if value is None:
                continue
            if key == "parallel_tool_calls":
                continue
            kwargs[key] = value

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 8192

        thinking_cfg = kwargs.get("extra_body", {}).get("thinking") if isinstance(kwargs.get("extra_body"), dict) else None
        thinking_enabled = (
            isinstance(thinking_cfg, dict)
            and thinking_cfg.get("type", "enabled") == "enabled"
        )
        if thinking_enabled:
            # DeepSeek thinking mode ignores these parameters; omit them to keep requests clean.
            for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                kwargs.pop(key, None)
        else:
            kwargs.setdefault("temperature", self.temperature)

        if include_tools and tools:
            kwargs["tools"] = tools

        return kwargs

    def _add_reasoning_content_if_available(self, api_response: any, response_data: dict) -> None:
        if "FC" in self.model_name or self.is_fc_model:
            self._add_reasoning_content_if_available_FC(api_response, response_data)
        else:
            self._add_reasoning_content_if_available_prompting(api_response, response_data)

    # The deepseek API is unstable at the moment, and will frequently give empty responses, so retry on JSONDecodeError is necessary
    @retry_with_backoff(error_type=[RateLimitError, json.JSONDecodeError])
    def generate_with_backoff(self, **kwargs):
        """
        Per the DeepSeek API documentation:
        https://api-docs.deepseek.com/quick_start/rate_limit

        DeepSeek API does NOT constrain user's rate limit. We will try out best to serve every request.
        But please note that when our servers are under high traffic pressure, you may receive 429 (Rate Limit Reached) or 503 (Server Overloaded). When this happens, please wait for a while and retry.

        Thus, backoff is still useful for handling 429 and 503 errors.
        """
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        api_model_name = self._configured_api_model_name()
        kwargs = {
            "model": api_model_name,
            "messages": message,
            **self._build_generation_kwargs(include_tools=True, tools=tools),
        }

        return self.generate_with_backoff(**kwargs)

    @override
    def _query_prompting(self, inference_data: dict):
        """
        This method is intended to be used by the `DeepSeek-R1` models. If used for other models, you will need to modify the code accordingly.

        Reasoning models don't support temperature parameter
        https://api-docs.deepseek.com/guides/reasoning_model

        `DeepSeek-R1` should use `deepseek-reasoner` as the model name in the API
        https://api-docs.deepseek.com/quick_start/pricing
        """
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        api_model_name = self._configured_api_model_name()

        return self.generate_with_backoff(
            model=api_model_name,
            messages=message,
            **self._build_generation_kwargs(),
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        # 'deepseek-reasoner does not support successive user messages, so we need to combine them
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    @override
    def _parse_query_response_prompting(self, api_response: any) -> dict:
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available(api_response, response_data)
        return response_data


class DeepSeekV32APIHandler(DeepSeekAPIHandler):
    """
    Handler for DeepSeek V3.2 API with reasoning support.
    
    DeepSeek V3.2 supports the `reasoning` parameter to control thinking mode.
    See: https://api-docs.deepseek.com/guides/reasoning_model
    """
    
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        extra_body = self.generator_config.get("extra_body", {})
        self.thinking_param = extra_body.get("thinking") if isinstance(extra_body, dict) else None
    
    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        
        api_model_name = self._configured_api_model_name()
        if api_model_name not in {"deepseek-v4-pro", "deepseek-v4-flash"}:
            # Compatibility aliases used by older DeepSeek configs.
            api_model_name = "deepseek-reasoner" if self.thinking_param is not None else "deepseek-chat"

        kwargs = {
            "model": api_model_name,
            "messages": message,
            **self._build_generation_kwargs(include_tools=True, tools=tools),
        }

        return self.generate_with_backoff(**kwargs)


class DeepSeekV4APIHandler(DeepSeekV32APIHandler):
    """Handler for official DeepSeek V4 API in function-calling thinking mode."""

    @override
    def _parse_query_response_FC(self, api_response: any) -> dict:
        response_data = OpenAICompletionsHandler._parse_query_response_FC(self, api_response)
        message = api_response.choices[0].message
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            response_data["reasoning_content"] = message.reasoning_content
        return response_data
