# -*- coding: utf-8 -*-
"""System configuration service for `.env` based settings."""

from __future__ import annotations

import io
import logging
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests

from data_provider.base import canonical_stock_code
from src.config import (
    _MANAGED_LITELLM_KEY_PROVIDERS,
    SUPPORTED_LLM_CHANNEL_PROTOCOLS,
    Config,
    _get_litellm_provider,
    _uses_direct_env_provider,
    canonicalize_llm_channel_protocol,
    channel_allows_empty_api_key,
    get_configured_llm_models,
    normalize_agent_litellm_model,
    normalize_news_strategy_profile,
    normalize_llm_channel_model,
    parse_env_bool,
    resolve_news_window_days,
    resolve_llm_channel_protocol,
    setup_env,
)
from src.core.config_manager import ConfigManager
from src.core.config_registry import (
    build_schema_response,
    get_category_definitions,
    get_field_definition,
    get_registered_field_keys,
)
from src.services.name_to_code_resolver import resolve_name_to_code

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when one or more submitted fields fail validation."""

    def __init__(self, issues: List[Dict[str, Any]]):
        super().__init__("Configuration validation failed")
        self.issues = issues


class ConfigConflictError(Exception):
    """Raised when submitted config_version is stale."""

    def __init__(self, current_version: str):
        super().__init__("Configuration version conflict")
        self.current_version = current_version


class ConfigImportError(Exception):
    """Raised when an imported `.env` payload is invalid."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SystemConfigService:
    """Service layer for reading, validating, and updating runtime configuration."""

    _DISPLAY_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
        "AGENT_SKILL_DIR": ("AGENT_SKILL_DIR", "AGENT_STRATEGY_DIR"),
        "AGENT_SKILL_AUTOWEIGHT": ("AGENT_SKILL_AUTOWEIGHT", "AGENT_STRATEGY_AUTOWEIGHT"),
        "AGENT_SKILL_ROUTING": ("AGENT_SKILL_ROUTING", "AGENT_STRATEGY_ROUTING"),
    }
    _DISPLAY_VALUE_ALIASES: Dict[str, Dict[str, str]] = {
        "AGENT_ORCHESTRATOR_MODE": {
            "strategy": "specialist",
            "skill": "specialist",
        }
    }

    def __init__(self, manager: Optional[ConfigManager] = None):
        self._manager = manager or ConfigManager()

    def get_schema(self) -> Dict[str, Any]:
        """Return grouped schema metadata for UI rendering."""
        return build_schema_response()

    @staticmethod
    def _mask_sensitive_value(
        *,
        key: str,
        value: str,
        field_schema: Dict[str, Any],
        mask_token: str,
    ) -> Tuple[str, bool]:
        if not value:
            return value, False
        if not bool(field_schema.get("is_sensitive", False)):
            return value, False
        return SystemConfigService._build_masked_sensitive_value(value, mask_token), True

    @staticmethod
    def _build_masked_sensitive_value(value: str, mask_token: str) -> str:
        """Preserve comma-separated secret shape while masking every populated segment."""
        if "," not in value:
            return mask_token
        return ",".join(
            mask_token if segment.strip() else ""
            for segment in value.split(",")
        )

    @staticmethod
    def _is_masked_placeholder_value(
        value: str,
        mask_token: str,
        *,
        current_value: Optional[str] = None,
    ) -> bool:
        """Return True when *value* is a masked secret placeholder from the UI."""
        if value == mask_token:
            return True
        if "," not in value:
            return False
        masked_segments = [segment.strip() for segment in value.split(",") if segment.strip()]
        if not masked_segments or any(segment != mask_token for segment in masked_segments):
            return False
        if current_value is None:
            return True
        current_segments = [
            segment.strip()
            for segment in str(current_value).split(",")
            if segment.strip()
        ]
        return bool(current_segments) and len(masked_segments) == len(current_segments)

    def _read_persisted_config_map(self) -> Dict[str, str]:
        """Read persisted `.env` values without mutating runtime state."""
        raw_config = self._manager.read_config_map()
        return {
            str(key): "" if value is None else str(value)
            for key, value in raw_config.items()
        }

    @staticmethod
    def _load_runtime_config(*, reload_runtime: bool = False) -> Config:
        if reload_runtime:
            Config.reset_instance()
            setup_env(override=True)
        return Config.get_instance()

    @staticmethod
    def _reload_runtime_singletons() -> None:
        """Reset runtime singleton services after config reload."""
        from src.agent.tools.data_tools import reset_fetcher_manager
        from src.search_service import reset_search_service

        reset_fetcher_manager()
        reset_search_service()

    @classmethod
    def _normalize_display_value(cls, key: str, value: str) -> str:
        alias_map = cls._DISPLAY_VALUE_ALIASES.get(key.upper())
        if not alias_map:
            return value
        return alias_map.get(value.strip().lower(), value)

    @classmethod
    def _build_display_config_map(cls, raw_config_map: Dict[str, str]) -> Dict[str, str]:
        raw_upper = {key.upper(): value for key, value in raw_config_map.items()}
        aliased_keys = {
            alias
            for candidates in cls._DISPLAY_KEY_ALIASES.values()
            for alias in candidates
        }
        display_map: Dict[str, str] = {}

        for key, value in raw_upper.items():
            if key in aliased_keys:
                continue
            display_map[key] = cls._normalize_display_value(key, value)

        for canonical_key, candidates in cls._DISPLAY_KEY_ALIASES.items():
            canonical_env_key = candidates[0]
            if canonical_env_key in raw_upper:
                display_map[canonical_key] = cls._normalize_display_value(
                    canonical_key,
                    raw_upper[canonical_env_key],
                )
                continue

            selected_value: Optional[str] = None
            candidate_seen = False
            for candidate_key in candidates[1:]:
                if candidate_key not in raw_upper:
                    continue
                candidate_seen = True
                candidate_value = raw_upper[candidate_key]
                if candidate_value:
                    selected_value = candidate_value
                    break
            if candidate_seen:
                if selected_value is None:
                    for candidate_key in candidates[1:]:
                        if candidate_key in raw_upper:
                            selected_value = raw_upper[candidate_key]
                            break
                if selected_value is None:
                    selected_value = ""
                display_map[canonical_key] = cls._normalize_display_value(
                    canonical_key,
                    selected_value,
                )

        return display_map

    def get_config(self, include_schema: bool = True, mask_token: str = "******") -> Dict[str, Any]:
        """Return current config values with secret masking and setup status."""
        config_map = self._build_display_config_map(self._manager.read_config_map())
        registered_keys = set(get_registered_field_keys())
        all_keys = set(config_map.keys()) | registered_keys

        category_orders = {
            item["category"]: item["display_order"]
            for item in get_category_definitions()
        }

        schema_by_key: Dict[str, Dict[str, Any]] = {
            key: get_field_definition(key, config_map.get(key, ""))
            for key in all_keys
        }

        items: List[Dict[str, Any]] = []
        for key in all_keys:
            raw_value = config_map.get(key, "")
            field_schema = schema_by_key[key]
            display_value, is_masked = self._mask_sensitive_value(
                key=key,
                value=raw_value,
                field_schema=field_schema,
                mask_token=mask_token,
            )
            item: Dict[str, Any] = {
                "key": key,
                "value": display_value,
                "raw_value_exists": bool(raw_value),
                "is_masked": is_masked,
            }
            if include_schema:
                item["schema"] = field_schema
            items.append(item)

        items.sort(
            key=lambda item: (
                category_orders.get(schema_by_key[item["key"]].get("category", "uncategorized"), 999),
                schema_by_key[item["key"]].get("display_order", 9999),
                item["key"],
            )
        )

        return {
            "config_version": self._manager.get_config_version(),
            "mask_token": mask_token,
            "items": items,
            "updated_at": self._manager.get_updated_at(),
            "setup_status": self.get_setup_status(),
        }

    def validate(self, items: Sequence[Dict[str, str]], mask_token: str = "******") -> Dict[str, Any]:
        """Validate submitted items without writing to `.env`."""
        issues = self._collect_issues(items=items, mask_token=mask_token)
        valid = not any(issue["severity"] == "error" for issue in issues)
        return {
            "valid": valid,
            "issues": issues,
        }

    def get_setup_status(self) -> Dict[str, Any]:
        """Return the first-run setup completion status for the current config."""
        effective_map = self._read_persisted_config_map()
        llm_check = self._build_primary_llm_check(effective_map=effective_map)
        agent_check = self._build_agent_llm_check(effective_map=effective_map, llm_check=llm_check)
        stock_check = self._build_stock_list_check(effective_map=effective_map)
        notification_check = self._build_notification_check(effective_map=effective_map)
        storage_check = self._build_storage_check(effective_map=effective_map)

        checks = [
            llm_check,
            agent_check,
            stock_check,
            notification_check,
            storage_check,
        ]
        required_missing = [
            check["key"]
            for check in checks
            if check["required"] and check["status"] not in {"configured", "inherited"}
        ]
        next_step = required_missing[0] if required_missing else None
        return {
            "is_complete": not required_missing,
            "ready_for_smoke": not {"llm_primary", "stock_list", "storage"} & set(required_missing),
            "required_missing_keys": required_missing,
            "next_step_key": next_step,
            "checks": checks,
        }

    def run_setup_smoke(self, stock_input: str = "") -> Dict[str, Any]:
        """Run a low-risk first-run smoke check without generating a formal report."""
        setup_status = self.get_setup_status()
        if not setup_status["ready_for_smoke"]:
            missing_labels = [
                check["title"]
                for check in setup_status["checks"]
                if check["key"] in set(setup_status["required_missing_keys"])
            ]
            return self._smoke_payload(False, "基础配置尚未满足试跑条件", error_code="setup_incomplete", next_step="请先补齐 LLM、自选股或本地存储配置", resolved_stock_code=None, summary=f"仍缺少：{'、'.join(missing_labels)}", setup_status=setup_status)

        runtime_config = self._load_runtime_config(reload_runtime=True)
        stock_candidate = (stock_input or "").strip()
        if not stock_candidate:
            stock_candidate = (runtime_config.stock_list[0] if runtime_config.stock_list else "").strip()

        resolved_stock_code, resolved_stock_name = self._resolve_setup_stock(stock_candidate)
        if not resolved_stock_code:
            return self._smoke_payload(False, "无法解析试跑股票", error_code="stock_not_found", next_step="请在设置页先保存 1-3 只可识别的股票代码或名称", resolved_stock_code=None, summary=f"输入“{stock_candidate or '空值'}”未匹配到有效股票", setup_status=setup_status)

        from src.core.pipeline import StockAnalysisPipeline

        pipeline = StockAnalysisPipeline(
            config=runtime_config,
            query_id="setup-smoke",
            query_source="setup_wizard",
        )
        results = pipeline.run(
            stock_codes=[resolved_stock_code],
            dry_run=True,
            send_notification=False,
        )
        result = results[0] if results else None
        target_date = pipeline._resolve_resume_target_date(resolved_stock_code)
        if pipeline.db.has_today_data(resolved_stock_code, target_date):
            return self._smoke_payload(True, "首次试跑通过", error_code=None, next_step="现在可以回到首页发起正式分析", resolved_stock_code=resolved_stock_code, summary=f"已完成 {resolved_stock_code} 的轻量数据抓取校验，未生成正式报告。", setup_status=setup_status)

        error_message = getattr(result, "error_message", None) or "轻量试跑失败，请检查数据源或股票代码配置"
        return self._smoke_payload(False, "首次试跑失败", error_code="dry_run_failed", next_step="请确认股票代码可用、网络可访问并重新试跑", resolved_stock_code=resolved_stock_code, summary=error_message, setup_status=setup_status)

    def export_desktop_env(self) -> Dict[str, Any]:
        """Return the raw active `.env` content for desktop-only backup."""
        if self._manager.env_path.exists():
            content = self._manager.env_path.read_text(encoding="utf-8")
        else:
            content = ""

        return {
            "content": content,
            "config_version": self._manager.get_config_version(),
            "updated_at": self._manager.get_updated_at(),
        }

    def import_desktop_env(
        self,
        *,
        config_version: str,
        content: str,
        reload_now: bool = True,
    ) -> Dict[str, Any]:
        """Merge imported `.env` assignments into the active config."""
        current_version = self._manager.get_config_version()
        if current_version != config_version:
            raise ConfigConflictError(current_version=current_version)

        updates = self._parse_imported_env_content(content)
        return self.update(
            config_version=config_version,
            items=updates,
            mask_token="__DSA_IMPORT_LITERAL_MASK__",
            reload_now=reload_now,
        )

    def discover_llm_channel_models(
        self,
        *,
        name: str,
        protocol: str,
        base_url: str,
        api_key: str,
        models: Sequence[str] = (),
        timeout_seconds: float = 20.0,
        mask_token: str = "******",
    ) -> Dict[str, Any]:
        """Discover available models from an OpenAI-compatible `/models` endpoint."""
        channel_name = name.strip() or "channel"
        existing_models = [str(m).strip() for m in models if str(m).strip()]
        api_key = self._resolve_request_api_key(
            channel_name=channel_name,
            protocol=protocol,
            api_key=api_key,
            mask_token=mask_token,
            models=existing_models,
        )
        validation_issues, resolved_protocol = self._validate_llm_channel_connection(
            channel_name=channel_name,
            protocol_value=protocol,
            base_url_value=base_url,
            api_key_value=api_key,
            model_values=existing_models,
            field_prefix="discover_channel",
            require_base_url=True,
        )
        if not resolved_protocol and existing_models:
            resolved_protocol = resolve_llm_channel_protocol(
                protocol,
                base_url=base_url,
                models=existing_models,
                channel_name=channel_name,
            )
        errors = [issue for issue in validation_issues if issue["severity"] == "error"]
        if errors:
            return {
                "success": False,
                "message": "LLM channel configuration is invalid",
                "error": errors[0]["message"],
                "resolved_protocol": resolved_protocol or None,
                "models": [],
                "latency_ms": None,
            }

        if resolved_protocol not in {"openai", "deepseek"}:
            return {
                "success": False,
                "message": "Model discovery is not supported for this protocol",
                "error": (
                    f"LLM channel '{channel_name}' protocol '{resolved_protocol}' "
                    "does not support /models discovery yet"
                ),
                "resolved_protocol": resolved_protocol or None,
                "models": [],
                "latency_ms": None,
            }

        api_keys = [segment.strip() for segment in api_key.split(",") if segment.strip()]
        selected_api_key = api_keys[0] if api_keys else ""
        request_headers = {"Accept": "application/json"}
        if selected_api_key:
            request_headers["Authorization"] = f"Bearer {selected_api_key}"

        models_url = self._build_llm_models_url(base_url)

        try:
            started_at = time.perf_counter()
            response = requests.get(
                models_url,
                headers=request_headers,
                timeout=max(5.0, float(timeout_seconds)),
                allow_redirects=False,
            )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
        except requests.RequestException as exc:
            logger.warning("LLM channel model discovery failed for %s: %s", channel_name, exc)
            return {
                "success": False,
                "message": "Failed to discover models",
                "error": str(exc),
                "resolved_protocol": resolved_protocol or None,
                "models": [],
                "latency_ms": None,
            }

        if 300 <= response.status_code < 400:
            return {
                "success": False,
                "message": "Model discovery request was redirected",
                "error": "Redirect responses are not allowed for model discovery",
                "resolved_protocol": resolved_protocol or None,
                "models": [],
                "latency_ms": latency_ms,
            }

        if not response.ok:
            return {
                "success": False,
                "message": "Model discovery request failed",
                "error": self._extract_llm_discovery_error(response),
                "resolved_protocol": resolved_protocol or None,
                "models": [],
                "latency_ms": latency_ms,
            }

        try:
            payload = response.json()
        except ValueError:
            return {
                "success": False,
                "message": "Model discovery returned invalid JSON",
                "error": "The /models endpoint did not return valid JSON",
                "resolved_protocol": resolved_protocol or None,
                "models": [],
                "latency_ms": latency_ms,
            }

        models = self._extract_discovered_llm_models(payload)
        if not models:
            return {
                "success": False,
                "message": "Model discovery returned no models",
                "error": "The /models endpoint did not return any model IDs",
                "resolved_protocol": resolved_protocol or None,
                "models": [],
                "latency_ms": latency_ms,
            }

        return {
            "success": True,
            "message": "LLM channel model discovery succeeded",
            "error": None,
            "resolved_protocol": resolved_protocol or None,
            "models": models,
            "latency_ms": latency_ms,
        }

    def test_llm_channel(
        self,
        *,
        name: str,
        protocol: str,
        base_url: str,
        api_key: str,
        models: Sequence[str],
        enabled: bool = True,
        timeout_seconds: float = 20.0,
        mask_token: str = "******",
    ) -> Dict[str, Any]:
        """Run a minimal completion call against one channel definition."""
        raw_models = [str(model).strip() for model in models if str(model).strip()]
        channel_name = name.strip() or "channel"
        api_key = self._resolve_request_api_key(
            channel_name=channel_name,
            protocol=protocol,
            api_key=api_key,
            mask_token=mask_token,
            models=raw_models,
        )
        stages = [
            self._stage("validation", "配置校验", "running", "正在检查渠道定义"),
            self._stage("model_discovery", "模型发现", "pending", "尚未开始"),
            self._stage("chat", "聊天接口", "pending", "尚未开始"),
            self._stage("response_parse", "响应解析", "pending", "尚未开始"),
        ]
        validation_issues = self._validate_llm_channel_definition(
            channel_name=channel_name,
            protocol_value=protocol,
            base_url_value=base_url,
            api_key_value=api_key,
            model_values=raw_models,
            enabled=enabled,
            field_prefix="test_channel",
            require_complete=True,
        )
        errors = [issue for issue in validation_issues if issue["severity"] == "error"]
        if errors:
            stages[0] = self._stage("validation", "配置校验", "failed", errors[0]["message"])
            return self._llm_test_payload(False, "LLM channel configuration is invalid", error=errors[0]["message"], error_type="invalid_config", resolved_model=None, latency_ms=None, next_step="请先补全渠道的 API Key、协议和模型配置", stages=stages)
        stages[0] = self._stage("validation", "配置校验", "success", "渠道配置格式有效")

        resolved_protocol = resolve_llm_channel_protocol(protocol, base_url=base_url, models=raw_models, channel_name=name)
        resolved_models = [normalize_llm_channel_model(model, resolved_protocol, base_url) for model in raw_models]
        resolved_model = resolved_models[0]
        api_keys = [segment.strip() for segment in api_key.split(",") if segment.strip()]
        selected_api_key = api_keys[0] if api_keys else ""

        stages[1] = self._stage("model_discovery", "模型发现", "success", f"将使用已配置模型 {resolved_model} 进行连通性测试")

        call_kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "temperature": self._normalize_test_llm_temperature(
                resolved_model,
                self._get_runtime_llm_temperature(),
            ),
            "max_tokens": 256,
            "timeout": max(5.0, float(timeout_seconds)),
        }
        if selected_api_key:
            call_kwargs["api_key"] = selected_api_key
        if base_url.strip():
            call_kwargs["api_base"] = base_url.strip()

        try:
            import litellm
            from src.agent.llm_adapter import LLMToolAdapter

            LLMToolAdapter._register_custom_model_pricing()

            started_at = time.perf_counter()
            response = litellm.completion(**call_kwargs)
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            stages[2] = self._stage("chat", "聊天接口", "success", f"接口返回成功{f'，耗时 {latency_ms} ms' if latency_ms else ''}")
            content = ""
            if response and getattr(response, "choices", None):
                choice = response.choices[0]
                content_blocks = getattr(choice, "content_blocks", None) or getattr(getattr(choice, "message", None), "content_blocks", None)
                if content_blocks:
                    text_parts = []
                    for block in content_blocks:
                        if getattr(block, "type", None) == "text":
                            text = getattr(block, "text", "") or ""
                            if text:
                                text_parts.append(text)
                        elif hasattr(block, "content") and block.content:
                            text_parts.append(block.content)
                    content = "".join(text_parts).strip()
                else:
                    message = getattr(choice, "message", None)
                    if message:
                        content = str(message.content or "").strip()

            if not content:
                stages[3] = self._stage("response_parse", "响应解析", "failed", "返回内容为空，无法确认模型输出")
                return self._llm_test_payload(False, "LLM channel returned an empty response", error="Empty response", error_type="empty_response", resolved_model=resolved_model, latency_ms=latency_ms, next_step="请检查模型权限、额度或切换到另一个可用模型后重试", stages=stages)
            stages[3] = self._stage("response_parse", "响应解析", "success", "已成功解析模型返回内容")
            return self._llm_test_payload(True, "LLM channel test succeeded", error=None, error_type=None, resolved_model=resolved_model, latency_ms=latency_ms, next_step=None, stages=stages)
        except Exception as exc:
            sanitized_error = self._sanitize_error_text(str(exc), secrets=[selected_api_key])
            error_type = self._classify_llm_error(sanitized_error)
            logger.warning("LLM channel test failed for %s: %s", channel_name, sanitized_error)
            stages[2] = self._stage("chat", "聊天接口", "failed", sanitized_error)
            stages[3] = self._stage("response_parse", "响应解析", "skipped", "接口请求未完成，未进入解析阶段")
            return self._llm_test_payload(False, "LLM channel test failed", error=sanitized_error, error_type=error_type, resolved_model=resolved_model, latency_ms=None, next_step=self._suggest_llm_next_step(error_type), stages=stages)

    def update(
        self,
        config_version: str,
        items: Sequence[Dict[str, str]],
        mask_token: str = "******",
        reload_now: bool = True,
    ) -> Dict[str, Any]:
        """Validate and persist updates into `.env`, then reload runtime config."""
        current_version = self._manager.get_config_version()
        if current_version != config_version:
            raise ConfigConflictError(current_version=current_version)

        issues = self._collect_issues(items=items, mask_token=mask_token)
        errors = [issue for issue in issues if issue["severity"] == "error"]
        if errors:
            raise ConfigValidationError(issues=errors)

        current_map = self._manager.read_config_map()
        submitted_keys: Set[str] = set()
        updates: List[Tuple[str, str]] = []
        sensitive_keys: Set[str] = set()
        for item in items:
            key = item["key"].upper()
            value = item["value"]
            field_schema = get_field_definition(key, value)
            normalized_value = self._normalize_value_for_storage(value, field_schema)
            submitted_keys.add(key)
            if bool(field_schema.get("is_sensitive", False)):
                sensitive_keys.add(key)
                if self._is_masked_placeholder_value(
                    normalized_value,
                    mask_token,
                    current_value=current_map.get(key),
                ) and current_map.get(key):
                    normalized_value = mask_token
            updates.append((key, normalized_value))

        updated_keys, skipped_masked_keys, new_version = self._manager.apply_updates(
            updates=updates,
            sensitive_keys=sensitive_keys,
            mask_token=mask_token,
        )

        warnings: List[str] = []
        reload_triggered = False
        if reload_now:
            try:
                Config.reset_instance()
                self._reload_runtime_singletons()
                setup_env(override=True)
                config = Config.get_instance()
                warnings.extend(config.validate())
                reload_triggered = True
            except Exception as exc:  # pragma: no cover - defensive branch
                logger.error("Configuration reload failed: %s", exc, exc_info=True)
                warnings.append("Configuration updated but reload failed")

        warnings.extend(
            self._build_explainability_warnings(
                submitted_keys=submitted_keys,
                reload_now=reload_now,
            )
        )

        return {
            "success": True,
            "config_version": new_version,
            "applied_count": len(updated_keys),
            "skipped_masked_count": len(skipped_masked_keys),
            "reload_triggered": reload_triggered,
            "updated_keys": updated_keys,
            "warnings": warnings,
        }

    def _build_explainability_warnings(
        self,
        *,
        submitted_keys: Set[str],
        reload_now: bool,
    ) -> List[str]:
        """Append user-facing runtime explainability warnings for key settings."""
        warnings: List[str] = []
        if not submitted_keys:
            return warnings

        current_map = self._manager.read_config_map()

        if submitted_keys & {"NEWS_MAX_AGE_DAYS", "NEWS_STRATEGY_PROFILE"}:
            raw_profile = current_map.get("NEWS_STRATEGY_PROFILE", "short")
            profile = normalize_news_strategy_profile(raw_profile)
            try:
                max_age = max(1, int(current_map.get("NEWS_MAX_AGE_DAYS", "3") or "3"))
            except (TypeError, ValueError):
                max_age = 3
            effective_days = resolve_news_window_days(
                news_max_age_days=max_age,
                news_strategy_profile=profile,
            )
            warnings.append(
                (
                    "新闻窗口已按策略计算："
                    f"NEWS_STRATEGY_PROFILE={profile}, "
                    f"NEWS_MAX_AGE_DAYS={max_age}, "
                    f"effective_days={effective_days} "
                    "(effective_days=min(profile_days, NEWS_MAX_AGE_DAYS))."
                )
            )

        if "MAX_WORKERS" in submitted_keys:
            try:
                max_workers = max(1, int(current_map.get("MAX_WORKERS", "3") or "3"))
            except (TypeError, ValueError):
                max_workers = 3
            if reload_now:
                warnings.append(
                    (
                        f"MAX_WORKERS={max_workers} 已保存。任务队列空闲时会自动应用；"
                        "若当前存在运行中任务，将在队列空闲后生效。"
                    )
                )
            else:
                warnings.append(
                    (
                        f"MAX_WORKERS={max_workers} 已写入 .env，但本次未触发运行时重载"
                        "（reload_now=false）；重载后才会应用。"
                    )
                )

        startup_only_run_keys = submitted_keys & {
            "RUN_IMMEDIATELY",
        }
        if startup_only_run_keys:
            warnings.append(
                (
                    f"{', '.join(sorted(startup_only_run_keys))} 已写入 .env。"
                    "它属于启动期单次运行配置：当前已运行的 WebUI/API 进程不会因为本次保存立即触发分析；"
                    "请重启当前进程后，在非 schedule 模式下按新值生效。"
                )
            )

        startup_only_schedule_keys = submitted_keys & {
            "SCHEDULE_ENABLED",
            "SCHEDULE_TIME",
            "SCHEDULE_RUN_IMMEDIATELY",
        }
        if startup_only_schedule_keys:
            warnings.append(
                (
                    f"{', '.join(sorted(startup_only_schedule_keys))} 已写入 .env。"
                    "这些属于启动期调度配置：当前已运行的 WebUI/API 进程不会因为本次保存立即触发分析，"
                    "也不会自动重建 scheduler；请重启当前进程，并以 schedule 模式重新启动后生效。"
                )
            )

        return warnings

    def apply_simple_updates(
        self,
        updates: Sequence[Tuple[str, str]],
        mask_token: str = "******",
    ) -> None:
        """Apply raw key updates without validation (internal service use only)."""
        self._manager.apply_updates(
            updates=updates,
            sensitive_keys=set(),
            mask_token=mask_token,
        )

    @staticmethod
    def _parse_imported_env_content(content: str) -> List[Dict[str, str]]:
        """Parse raw `.env` text into update items using current dotenv semantics."""
        normalized_content = content.replace("\ufeff", "")
        if not normalized_content.strip():
            raise ConfigImportError("未识别到有效 .env 配置")

        from dotenv import dotenv_values

        parsed = dotenv_values(stream=io.StringIO(normalized_content))
        updates: List[Dict[str, str]] = []
        for key, value in parsed.items():
            if key is None:
                continue
            updates.append(
                {
                    "key": str(key).upper(),
                    "value": "" if value is None else str(value),
                }
            )

        if not updates:
            raise ConfigImportError("未识别到有效 .env 配置")

        return updates

    def _collect_issues(self, items: Sequence[Dict[str, str]], mask_token: str) -> List[Dict[str, Any]]:
        """Collect field-level and cross-field validation issues."""
        current_map = self._manager.read_config_map()
        effective_map = dict(current_map)
        issues: List[Dict[str, Any]] = []
        updated_map: Dict[str, str] = {}

        for item in items:
            key = item["key"].upper()
            value = item["value"]
            field_schema = get_field_definition(key, value)
            is_sensitive = bool(field_schema.get("is_sensitive", False))

            if is_sensitive and self._is_masked_placeholder_value(
                value,
                mask_token,
                current_value=current_map.get(key),
            ) and current_map.get(key):
                continue

            updated_map[key] = value
            effective_map[key] = value
            issues.extend(self._validate_value(key=key, value=value, field_schema=field_schema))

        issues.extend(self._validate_cross_field(effective_map=effective_map, updated_keys=set(updated_map.keys())))
        return issues

    @staticmethod
    def _legacy_provider_api_key_candidates(protocol: str) -> List[str]:
        normalized_protocol = canonicalize_llm_channel_protocol(protocol or "")
        if not normalized_protocol:
            return []

        candidates: List[str] = []
        seen: Set[str] = set()

        def _add_candidate(env_key: str) -> None:
            normalized_key = (env_key or "").strip().upper()
            if normalized_key and normalized_key not in seen:
                seen.add(normalized_key)
                candidates.append(normalized_key)

        provider_aliases = {normalized_protocol}
        litellm_provider = _get_litellm_provider(normalized_protocol)
        if litellm_provider:
            provider_aliases.add(str(litellm_provider).strip().lower())

        explicit_candidates = {
            "gemini": ("GEMINI_API_KEYS", "GEMINI_API_KEY"),
            "vertex_ai": ("GEMINI_API_KEYS", "GEMINI_API_KEY"),
            "anthropic": ("ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY"),
            "openai": ("OPENAI_API_KEYS", "AIHUBMIX_KEY", "OPENAI_API_KEY"),
            "deepseek": ("DEEPSEEK_API_KEYS", "DEEPSEEK_API_KEY"),
        }

        for provider_name in provider_aliases:
            normalized_provider = str(provider_name or "").strip().lower()
            if not normalized_provider:
                continue
            for env_key in explicit_candidates.get(normalized_provider, ()):
                _add_candidate(env_key)
            if _uses_direct_env_provider(normalized_provider):
                env_prefix = normalized_provider.upper().replace("-", "_")
                _add_candidate(f"{env_prefix}_API_KEYS")
                _add_candidate(f"{env_prefix}_API_KEY")

        return candidates

    def _resolve_masked_request_api_key(
        self,
        *,
        channel_name: str,
        protocol: str,
        current_map: Dict[str, str],
        models: Sequence[str] = (),
    ) -> str:
        prefix = f"LLM_{channel_name.upper()}"
        channel_api_key = (current_map.get(f"{prefix}_API_KEYS") or "").strip() or (current_map.get(f"{prefix}_API_KEY") or "").strip()
        if channel_api_key:
            return channel_api_key

        provider_candidates: List[str] = []
        seen_candidates: Set[str] = set()

        def _add_candidates(candidate_keys: Sequence[str]) -> None:
            for env_key in candidate_keys:
                normalized_key = (env_key or "").strip().upper()
                if normalized_key and normalized_key not in seen_candidates:
                    seen_candidates.add(normalized_key)
                    provider_candidates.append(normalized_key)

        for model in models:
            raw_model = str(model).strip()
            if "/" not in raw_model:
                continue
            provider_name = raw_model.split("/", 1)[0]
            _add_candidates(self._api_key_candidates_for_provider(provider_name))

        _add_candidates(self._api_key_candidates_for_provider(channel_name))

        resolved_protocol = resolve_llm_channel_protocol(
            current_map.get(f"{prefix}_PROTOCOL") or protocol,
            base_url=current_map.get(f"{prefix}_BASE_URL") or "",
            models=[
                model.strip()
                for model in (current_map.get(f"{prefix}_MODELS") or "").split(",")
                if model.strip()
            ],
            channel_name=channel_name,
        )
        _add_candidates(self._legacy_provider_api_key_candidates(resolved_protocol))

        for env_key in provider_candidates:
            resolved_api_key = (current_map.get(env_key) or "").strip()
            if resolved_api_key:
                return resolved_api_key

        return ""

    def _resolve_request_api_key(
        self,
        *,
        channel_name: str,
        protocol: str,
        api_key: str,
        mask_token: str,
        models: Sequence[str] = (),
    ) -> str:
        if not self._is_masked_placeholder_value(api_key, mask_token):
            return api_key
        current_map = self._read_persisted_config_map()
        return (
            self._resolve_masked_request_api_key(
                channel_name=channel_name,
                protocol=protocol,
                current_map=current_map,
                models=models,
            )
            or api_key
        )

    @classmethod
    def _api_key_candidates_for_provider(cls, provider_name: str) -> List[str]:
        normalized_provider = canonicalize_llm_channel_protocol(provider_name or "")
        if normalized_provider in _MANAGED_LITELLM_KEY_PROVIDERS:
            return cls._legacy_provider_api_key_candidates(normalized_provider)
        if normalized_provider == "ollama":
            return []

        raw_provider = str(provider_name or "").strip().lower().replace("-", "_")
        if not raw_provider:
            return []
        env_prefix = raw_provider.upper().replace("-", "_")
        return [f"{env_prefix}_API_KEYS", f"{env_prefix}_API_KEY"]

    @staticmethod
    def _sanitize_error_text(text: str, secrets: Sequence[str]) -> str:
        sanitized = text or ""
        for secret in secrets:
            if secret:
                sanitized = sanitized.replace(secret, "******")
        sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer ******", sanitized, flags=re.IGNORECASE)
        return sanitized

    @staticmethod
    def _classify_llm_error(message: str, status_code: Optional[int] = None) -> str:
        lowered = (message or "").lower()
        if status_code in {401, 403} or "unauthorized" in lowered or "invalid api key" in lowered or "authentication" in lowered: return "auth_error"
        if status_code == 404 or "not found" in lowered or "does not exist" in lowered or "unknown model" in lowered: return "model_not_found"
        if "timeout" in lowered or "timed out" in lowered: return "timeout"
        if "empty response" in lowered: return "empty_response"
        if "json" in lowered or "parse" in lowered or "schema" in lowered: return "response_parse_error"
        if status_code and status_code >= 400: return "api_error"
        return "network_error"

    @staticmethod
    def _suggest_llm_next_step(error_type: str) -> str:
        suggestions = {"invalid_config": "请先补全渠道配置并保存后再测试", "auth_error": "请检查 API Key 是否正确、是否有额度，或更换可用账号后重试", "timeout": "请检查网络连通性、代理或 Base URL 是否可访问", "model_not_found": "请确认模型名拼写正确，或先通过“获取模型”重新选择可用模型", "empty_response": "请切换到另一个模型，或确认当前模型支持聊天接口", "response_parse_error": "请确认该渠道兼容 OpenAI Chat 接口返回格式", "api_error": "请根据返回错误检查渠道权限、地域限制或请求参数", "network_error": "请检查 Base URL、网络连接或本地服务是否已启动"}
        return suggestions.get(error_type, "请检查渠道配置后重试")

    @staticmethod
    def _setup_check(key: str, title: str, category: str, required: bool, status: str, message: str, next_action: Optional[str] = None) -> Dict[str, Any]:
        return {"key": key, "title": title, "category": category, "required": required, "status": status, "message": message, "next_action": next_action}

    @staticmethod
    def _stage(key: str, title: str, status: str, detail: str) -> Dict[str, str]:
        return {"key": key, "title": title, "status": status, "detail": detail}

    @staticmethod
    def _llm_test_payload(success: bool, message: str, *, error: Optional[str], error_type: Optional[str], resolved_model: Optional[str], latency_ms: Optional[int], next_step: Optional[str], stages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {"success": success, "message": message, "error": error, "error_type": error_type, "resolved_model": resolved_model, "latency_ms": latency_ms, "next_step": next_step, "stages": stages}

    @staticmethod
    def _smoke_payload(success: bool, message: str, *, error_code: Optional[str], next_step: Optional[str], resolved_stock_code: Optional[str], summary: str, setup_status: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": success, "message": message, "error_code": error_code, "next_step": next_step, "resolved_stock_code": resolved_stock_code, "summary": summary, "setup_status": setup_status}

    def _build_primary_llm_check(self, *, effective_map: Dict[str, str]) -> Dict[str, Any]:
        active_model, source = self._resolve_primary_model_from_map(effective_map)
        if active_model:
            if source == "llm_channels":
                message = f"当前可用主模型：{active_model}（来自已启用渠道）"
            elif source == "legacy_env":
                message = f"当前可用主模型：{active_model}（来自 legacy API Key）"
            elif source == "litellm_config":
                message = f"已检测到 LiteLLM 路由配置，共声明 {len(self._collect_yaml_models_from_map(effective_map))} 个模型"
            else:
                message = f"当前可用主模型：{active_model}"
            return self._setup_check("llm_primary", "LLM 主渠道", "ai_model", True, "configured", message)
        return self._setup_check("llm_primary", "LLM 主渠道", "ai_model", True, "needs_action", "尚未检测到可用的主模型配置", "请先配置至少一个可用的模型渠道或主模型")

    def _build_agent_llm_check(
        self,
        *,
        effective_map: Dict[str, str],
        llm_check: Dict[str, Any],
    ) -> Dict[str, Any]:
        if llm_check["status"] not in {"configured", "inherited"}:
            return self._setup_check("llm_agent", "Agent 渠道", "agent", True, "needs_action", "Agent 当前还没有可继承的主模型", "请先完成主模型配置")

        primary_model, _ = self._resolve_primary_model_from_map(effective_map)
        configured_agent_model = (effective_map.get("AGENT_LITELLM_MODEL") or "").strip()
        if not configured_agent_model:
            return self._setup_check("llm_agent", "Agent 渠道", "agent", True, "inherited", f"未单独配置，默认继承主模型：{primary_model}")

        available_models = set(
            self._collect_yaml_models_from_map(effective_map)
            or self._collect_llm_channel_models_from_map(effective_map)
        )
        normalized_agent_model = normalize_agent_litellm_model(
            configured_agent_model,
            configured_models=available_models,
        )
        if normalized_agent_model and (
            normalized_agent_model in available_models
            or self._has_runtime_source_for_model(normalized_agent_model, effective_map)
        ):
            return self._setup_check("llm_agent", "Agent 渠道", "agent", True, "configured", f"当前 Agent 主模型：{normalized_agent_model}")
        return self._setup_check("llm_agent", "Agent 渠道", "agent", True, "needs_action", "Agent 主模型当前不可用", "请检查 AGENT_LITELLM_MODEL 是否仍对应已启用渠道或匹配的 API Key")

    @staticmethod
    def _build_stock_list_check(*, effective_map: Dict[str, str]) -> Dict[str, Any]:
        stock_count = len(
            [
                stock.strip()
                for stock in (effective_map.get("STOCK_LIST") or "").split(",")
                if stock.strip()
            ]
        )
        if stock_count > 0:
            return SystemConfigService._setup_check("stock_list", "自选股", "base", True, "configured", f"已配置 {stock_count} 只股票")
        return SystemConfigService._setup_check("stock_list", "自选股", "base", True, "needs_action", "当前自选股列表为空", "请至少添加 1 只股票用于首次试跑")

    @staticmethod
    def _build_notification_check(*, effective_map: Dict[str, str]) -> Dict[str, Any]:
        configured = any(
            [
                bool((effective_map.get("FEISHU_WEBHOOK_URL") or "").strip()),
                bool((effective_map.get("TELEGRAM_BOT_TOKEN") or "").strip() and (effective_map.get("TELEGRAM_CHAT_ID") or "").strip()),
                bool(
                    (effective_map.get("EMAIL_SENDER") or "").strip()
                    and (effective_map.get("EMAIL_PASSWORD") or "").strip()
                    and any(receiver.strip() for receiver in (effective_map.get("EMAIL_RECEIVERS") or "").split(","))
                ),
                bool((effective_map.get("PUSHPLUS_TOKEN") or "").strip()),
                bool((effective_map.get("SERVERCHAN3_SENDKEY") or "").strip()),
                bool(any(url.strip() for url in (effective_map.get("CUSTOM_WEBHOOK_URLS") or "").split(","))),
                bool((effective_map.get("DISCORD_WEBHOOK_URL") or "").strip()),
                bool((effective_map.get("SLACK_WEBHOOK_URL") or "").strip()),
                bool((effective_map.get("PUSHOVER_USER_KEY") or "").strip() and (effective_map.get("PUSHOVER_API_TOKEN") or "").strip()),
            ]
        )
        return SystemConfigService._setup_check(
            "notification",
            "通知渠道",
            "notification",
            False,
            "configured" if configured else "optional",
            "已检测到可用通知配置" if configured else "通知为可选项，未配置也不影响首次跑通",
            None if configured else "需要时可稍后再配置飞书、Telegram、邮件等通知",
        )

    @staticmethod
    def _build_storage_check(*, effective_map: Dict[str, str]) -> Dict[str, Any]:
        db_path = Path(
            (effective_map.get("DATABASE_PATH") or "./data/stock_analysis.db").strip()
            or "./data/stock_analysis.db"
        ).expanduser()
        parent_path = db_path.parent if str(db_path.parent) else Path(".")
        try:
            if db_path.exists() and db_path.is_dir():
                raise ValueError("DATABASE_PATH 指向了目录而不是文件")
            if parent_path.exists() and not parent_path.is_dir():
                raise ValueError("DATABASE_PATH 的父路径不是目录")

            existing_parent = parent_path
            while not existing_parent.exists() and existing_parent != existing_parent.parent:
                existing_parent = existing_parent.parent
            if not existing_parent.exists():
                raise ValueError("DATABASE_PATH 不存在可写父目录")
            if not os.access(existing_parent, os.W_OK | os.X_OK):
                raise PermissionError(f"目录不可写：{existing_parent}")
            if db_path.exists() and not os.access(db_path, os.W_OK):
                raise PermissionError(f"数据库文件不可写：{db_path}")

            detail = (
                f"SQLite 路径可写：{db_path}"
                if not db_path.exists()
                else f"SQLite 文件可写：{db_path}"
            )
            return SystemConfigService._setup_check("storage", "数据库 / 本地存储", "system", True, "configured", detail)
        except Exception as exc:
            return SystemConfigService._setup_check("storage", "数据库 / 本地存储", "system", True, "needs_action", f"本地数据库不可用：{exc}", "请检查 DATABASE_PATH 指向的目录是否可写")

    @classmethod
    def _resolve_primary_model_from_map(cls, effective_map: Dict[str, str]) -> Tuple[str, str]:
        explicit_primary_model = (effective_map.get("LITELLM_MODEL") or "").strip()
        available_models = cls._collect_yaml_models_from_map(effective_map)
        if available_models:
            available_model_set = set(available_models)
            if explicit_primary_model:
                if explicit_primary_model in available_model_set:
                    return explicit_primary_model, "litellm_config"
                return "", ""
            return available_models[0], "litellm_config"

        channel_models = cls._collect_llm_channel_models_from_map(effective_map)
        if explicit_primary_model:
            if explicit_primary_model in set(channel_models) or cls._has_runtime_source_for_model(
                explicit_primary_model,
                effective_map,
            ):
                return explicit_primary_model, "explicit"
            return "", ""

        if channel_models:
            return channel_models[0], "llm_channels"

        gemini_keys = [
            key.strip()
            for key in ((effective_map.get("GEMINI_API_KEYS") or "").split(","))
            if key.strip()
        ]
        if not gemini_keys and (effective_map.get("GEMINI_API_KEY") or "").strip():
            gemini_keys = [(effective_map.get("GEMINI_API_KEY") or "").strip()]
        if gemini_keys:
            return f"gemini/{(effective_map.get('GEMINI_MODEL') or 'gemini-3-flash-preview').strip()}", "legacy_env"

        anthropic_keys = [
            key.strip()
            for key in ((effective_map.get("ANTHROPIC_API_KEYS") or "").split(","))
            if key.strip()
        ]
        if not anthropic_keys and (effective_map.get("ANTHROPIC_API_KEY") or "").strip():
            anthropic_keys = [(effective_map.get("ANTHROPIC_API_KEY") or "").strip()]
        if anthropic_keys:
            return (
                f"anthropic/{(effective_map.get('ANTHROPIC_MODEL') or 'claude-3-5-sonnet-20241022').strip()}",
                "legacy_env",
            )

        deepseek_keys = [
            key.strip()
            for key in ((effective_map.get("DEEPSEEK_API_KEYS") or "").split(","))
            if key.strip()
        ]
        if not deepseek_keys and (effective_map.get("DEEPSEEK_API_KEY") or "").strip():
            deepseek_keys = [(effective_map.get("DEEPSEEK_API_KEY") or "").strip()]
        if deepseek_keys:
            return "deepseek/deepseek-chat", "legacy_env"

        openai_keys = [
            key.strip()
            for key in ((effective_map.get("OPENAI_API_KEYS") or "").split(","))
            if key.strip()
        ]
        if not openai_keys:
            legacy_openai_key = (effective_map.get("AIHUBMIX_KEY") or "").strip() or (effective_map.get("OPENAI_API_KEY") or "").strip()
            if legacy_openai_key:
                openai_keys = [legacy_openai_key]
        if openai_keys:
            openai_model = (effective_map.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
            return (openai_model if "/" in openai_model else f"openai/{openai_model}"), "legacy_env"

        return "", ""

    @staticmethod
    def _resolve_setup_stock(raw_value: str) -> Tuple[Optional[str], Optional[str]]:
        text = (raw_value or "").strip()
        if not text:
            return None, None
        if re.fullmatch(r"^[A-Za-z0-9.\-]+$", text):
            return canonical_stock_code(text), None
        resolved_code = resolve_name_to_code(text)
        if not resolved_code:
            return None, None
        return canonical_stock_code(resolved_code), text

    @staticmethod
    def _validate_value(key: str, value: str, field_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate a single field value against schema metadata."""
        issues: List[Dict[str, Any]] = []
        data_type = field_schema.get("data_type", "string")
        validation = field_schema.get("validation", {}) or {}
        is_required = field_schema.get("is_required", False)

        # Empty values are valid for non-required fields (skip type validation)
        if not value.strip() and not is_required:
            return issues

        if ("\n" in value or "\r" in value) and data_type != "json":
            issues.append(
                {
                    "key": key,
                    "code": "invalid_value",
                    "message": "Value cannot contain newline characters",
                    "severity": "error",
                    "expected": "single-line value",
                    "actual": "contains newline",
                }
            )
            return issues

        if data_type == "integer":
            try:
                numeric = int(value)
            except ValueError:
                return [
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be an integer",
                        "severity": "error",
                        "expected": "integer",
                        "actual": value,
                    }
                ]
            issues.extend(SystemConfigService._validate_numeric_range(key, numeric, validation))

        elif data_type == "number":
            try:
                numeric = float(value)
            except ValueError:
                return [
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be a number",
                        "severity": "error",
                        "expected": "number",
                        "actual": value,
                    }
                ]
            issues.extend(SystemConfigService._validate_numeric_range(key, numeric, validation))

        elif data_type == "boolean":
            if value.strip().lower() not in {"true", "false"}:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be true or false",
                        "severity": "error",
                        "expected": "true|false",
                        "actual": value,
                    }
                )

        elif data_type == "time":
            pattern = validation.get("pattern") or r"^([01]\d|2[0-3]):[0-5]\d$"
            if not re.match(pattern, value.strip()):
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_format",
                        "message": "Value must be in HH:MM format",
                        "severity": "error",
                        "expected": "HH:MM",
                        "actual": value,
                    }
                )

        elif data_type == "json":
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_json",
                        "message": "Value must be valid JSON",
                        "severity": "error",
                        "expected": "valid JSON",
                        "actual": value[:120],
                    }
                )
            else:
                if key == "AGENT_EVENT_ALERT_RULES_JSON":
                    try:
                        from src.agent.events import parse_event_alert_rules, validate_event_alert_rule

                        rule_index = 0
                        for rule_index, rule in enumerate(parse_event_alert_rules(parsed), start=1):
                            validate_event_alert_rule(rule)
                    except ValueError as exc:
                        issues.append(
                            {
                                "key": key,
                                "code": "invalid_event_rule",
                                "message": f"Rule validation failed: {exc}",
                                "severity": "error",
                                "expected": "supported EventMonitor rule fields and enum values",
                                "actual": f"rule #{rule_index or 1}",
                            }
                        )

        if "enum" in validation and value and value not in validation["enum"]:
            issues.append(
                {
                    "key": key,
                    "code": "invalid_enum",
                    "message": "Value is not in allowed options",
                    "severity": "error",
                    "expected": ",".join(validation["enum"]),
                    "actual": value,
                }
            )

        if validation.get("item_type") == "url":
            delimiter = validation.get("delimiter", ",")
            values = [item.strip() for item in value.split(delimiter)] if validation.get("multi_value") else [value.strip()]
            allowed_schemes = tuple(validation.get("allowed_schemes", ["http", "https"]))
            invalid_values = [
                item for item in values
                if item and not SystemConfigService._is_valid_url(item, allowed_schemes=allowed_schemes)
            ]
            if invalid_values:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_url",
                        "message": "Value must contain valid URLs with scheme and host",
                        "severity": "error",
                        "expected": ",".join(allowed_schemes) + " URL(s)",
                        "actual": ", ".join(invalid_values[:3]),
                    }
                )

        return issues

    @staticmethod
    def _normalize_value_for_storage(value: str, field_schema: Dict[str, Any]) -> str:
        """Normalize submitted values before persisting to the single-line .env file."""
        if field_schema.get("data_type", "string") != "json":
            return value

        if not value.strip():
            return value

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value

        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _validate_numeric_range(key: str, numeric_value: float, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        min_value = validation.get("min")
        max_value = validation.get("max")

        if min_value is not None and numeric_value < min_value:
            issues.append(
                {
                    "key": key,
                    "code": "out_of_range",
                    "message": "Value is lower than minimum",
                    "severity": "error",
                    "expected": f">={min_value}",
                    "actual": str(numeric_value),
                }
            )
        if max_value is not None and numeric_value > max_value:
            issues.append(
                {
                    "key": key,
                    "code": "out_of_range",
                    "message": "Value is greater than maximum",
                    "severity": "error",
                    "expected": f"<={max_value}",
                    "actual": str(numeric_value),
                }
            )
        return issues

    @staticmethod
    def _is_valid_url(value: str, allowed_schemes: Tuple[str, ...]) -> bool:
        """Return True when *value* looks like a valid absolute URL."""
        parsed = urlparse(value)
        return parsed.scheme in allowed_schemes and bool(parsed.netloc)

    @staticmethod
    def _is_safe_base_url(value: str) -> bool:
        """Block link-local and cloud metadata addresses to prevent SSRF.

        Allows localhost / private-LAN addresses (e.g. Ollama on 192.168.x.x)
        but blocks 169.254.x.x (AWS/Azure/GCP/Alibaba instance-metadata service)
        and other known metadata hostnames.
        """
        import ipaddress

        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if not host:
            return True
        # Known cloud metadata hostnames
        _BLOCKED_HOSTS = frozenset({
            "169.254.169.254",
            "metadata.google.internal",
            "100.100.100.200",
        })
        if host in _BLOCKED_HOSTS:
            return False
        # Numeric IPs: block link-local range (169.254.0.0/16)
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_link_local:
                return False
        except ValueError:
            pass  # hostname, not an IP — already checked against blocklist above
        return True

    @staticmethod
    def _build_llm_models_url(base_url: str) -> str:
        """Convert a channel base URL into a `/models` endpoint."""
        parsed = urlparse(base_url.strip())
        normalized = (parsed.path or "").rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        if normalized.endswith("/models"):
            models_path = normalized or "/models"
        else:
            models_path = f"{normalized}/models" if normalized else "/models"
        return urlunparse(parsed._replace(path=models_path, params="", query="", fragment=""))

    @staticmethod
    def _get_runtime_llm_temperature() -> float:
        """Return the current configured LLM temperature for ad-hoc channel tests."""
        config = Config._load_from_env()
        try:
            return float(getattr(config, "llm_temperature", 0.7))
        except (TypeError, ValueError):
            return 0.7

    @staticmethod
    def _normalize_test_llm_temperature(model: str, temperature: float, default: float = 0.7) -> float:
        """Apply provider-required temperature overrides for ad-hoc channel tests."""
        normalized_model = (model or "").strip().lower()
        if "kimi-k2.6" in normalized_model:
            return 1.0
        try:
            return float(temperature)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extract_llm_discovery_error(response: requests.Response) -> str:
        """Extract a concise error message from a failed model discovery response."""
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = str(
                    error_payload.get("message")
                    or error_payload.get("code")
                    or ""
                ).strip()
                if message:
                    return message

            message = str(payload.get("message") or payload.get("detail") or "").strip()
            if message:
                return message

        text = response.text.strip()
        if text:
            return text[:200]
        return f"HTTP {response.status_code}"

    @staticmethod
    def _extract_discovered_llm_models(payload: Any) -> List[str]:
        """Normalize common `/models` response shapes into a unique model ID list."""
        raw_models: List[Any] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                raw_models = payload["data"]
            elif isinstance(payload.get("models"), list):
                raw_models = payload["models"]
        elif isinstance(payload, list):
            raw_models = payload

        models: List[str] = []
        seen: Set[str] = set()
        for entry in raw_models:
            if isinstance(entry, str):
                model_id = entry.strip()
            elif isinstance(entry, dict):
                model_id = str(
                    entry.get("id") or entry.get("model") or entry.get("name") or ""
                ).strip()
            else:
                model_id = ""

            if not model_id or model_id in seen:
                continue

            seen.add(model_id)
            models.append(model_id)

        return models

    @staticmethod
    def _validate_cross_field(effective_map: Dict[str, str], updated_keys: Set[str]) -> List[Dict[str, Any]]:
        """Validate dependencies across multiple keys."""
        issues: List[Dict[str, Any]] = []

        token_value = (effective_map.get("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id_value = (effective_map.get("TELEGRAM_CHAT_ID") or "").strip()
        if token_value and not chat_id_value and (
            "TELEGRAM_BOT_TOKEN" in updated_keys or "TELEGRAM_CHAT_ID" in updated_keys
        ):
            issues.append(
                {
                    "key": "TELEGRAM_CHAT_ID",
                    "code": "missing_dependency",
                    "message": "TELEGRAM_CHAT_ID is required when TELEGRAM_BOT_TOKEN is set",
                    "severity": "error",
                    "expected": "non-empty TELEGRAM_CHAT_ID",
                    "actual": chat_id_value,
                }
            )

        feishu_relevant_keys = {
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_WEBHOOK_URL",
            "FEISHU_WEBHOOK_SECRET",
            "FEISHU_WEBHOOK_KEYWORD",
            "FEISHU_STREAM_ENABLED",
            "FEISHU_FOLDER_TOKEN",
        }
        has_feishu_app_id = bool((effective_map.get("FEISHU_APP_ID") or "").strip())
        has_feishu_app_secret = bool((effective_map.get("FEISHU_APP_SECRET") or "").strip())
        has_feishu_app_credentials = has_feishu_app_id or has_feishu_app_secret
        has_feishu_webhook = bool((effective_map.get("FEISHU_WEBHOOK_URL") or "").strip())
        has_feishu_folder_token = bool((effective_map.get("FEISHU_FOLDER_TOKEN") or "").strip())
        has_feishu_full_cloud_doc_credentials = (
            has_feishu_app_id
            and has_feishu_app_secret
            and has_feishu_folder_token
        )
        # Match runtime semantics: Config.from_env only enables stream mode
        # when the value is exactly "true" (case-insensitive).
        feishu_stream_enabled = (
            (effective_map.get("FEISHU_STREAM_ENABLED") or "false")
            .strip()
            .lower()
            == "true"
        )
        if (
            has_feishu_app_credentials
            and not has_feishu_full_cloud_doc_credentials
            and not has_feishu_webhook
            and not (feishu_stream_enabled and has_feishu_app_id and has_feishu_app_secret)
            and (updated_keys & feishu_relevant_keys)
        ):
            issues.append(
                {
                    "key": "FEISHU_WEBHOOK_URL",
                    "code": "feishu_mode_mismatch",
                    "message": (
                        "仅配置 FEISHU_APP_ID / FEISHU_APP_SECRET 不会开启飞书群 Webhook 推送；"
                        "如需通知推送请填写 FEISHU_WEBHOOK_URL，若要使用应用机器人请同时开启 "
                        "FEISHU_STREAM_ENABLED 并完成应用发布与权限配置。"
                    ),
                    "severity": "warning",
                    "expected": "FEISHU_WEBHOOK_URL or FEISHU_STREAM_ENABLED=true",
                    "actual": "app credentials only",
                }
            )

        issues.extend(
            SystemConfigService._validate_llm_channel_map(
                effective_map=effective_map,
                updated_keys=updated_keys,
            )
        )
        issues.extend(SystemConfigService._validate_llm_runtime_selection(effective_map=effective_map))

        return issues

    @staticmethod
    def _validate_llm_channel_map(effective_map: Dict[str, str], updated_keys: Set[str]) -> List[Dict[str, Any]]:
        """Validate channel-style LLM configuration stored in `.env`."""
        issues: List[Dict[str, Any]] = []
        if SystemConfigService._uses_litellm_yaml(effective_map):
            return issues

        raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
        if not raw_channels:
            return issues

        normalized_names: List[str] = []
        seen_names: Set[str] = set()
        for raw_name in raw_channels.split(","):
            name = raw_name.strip()
            if not name:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                issues.append(
                    {
                        "key": "LLM_CHANNELS",
                        "code": "invalid_channel_name",
                        "message": f"LLM channel name '{name}' may only contain letters, numbers, and underscores",
                        "severity": "error",
                        "expected": "letters/numbers/underscores",
                        "actual": name,
                    }
                )
                continue

            normalized_upper = name.upper()
            if normalized_upper in seen_names:
                issues.append(
                    {
                        "key": "LLM_CHANNELS",
                        "code": "duplicate_channel_name",
                        "message": f"LLM channel '{name}' is declared more than once",
                        "severity": "error",
                        "expected": "unique channel names",
                        "actual": raw_channels,
                    }
                )
                continue

            seen_names.add(normalized_upper)
            normalized_names.append(name)

        for name in normalized_names:
            prefix = f"LLM_{name.upper()}"
            protocol_value = (effective_map.get(f"{prefix}_PROTOCOL") or "").strip()
            base_url_value = (effective_map.get(f"{prefix}_BASE_URL") or "").strip()
            api_key_value = (
                (effective_map.get(f"{prefix}_API_KEYS") or "").strip()
                or (effective_map.get(f"{prefix}_API_KEY") or "").strip()
            )
            models_value = [
                model.strip()
                for model in (effective_map.get(f"{prefix}_MODELS") or "").split(",")
                if model.strip()
            ]
            enabled = parse_env_bool(effective_map.get(f"{prefix}_ENABLED"), default=True)
            issues.extend(
                SystemConfigService._validate_llm_channel_definition(
                    channel_name=name,
                    protocol_value=protocol_value,
                    base_url_value=base_url_value,
                    api_key_value=api_key_value,
                    model_values=models_value,
                    enabled=enabled,
                    field_prefix=prefix,
                    require_complete=enabled,
                )
            )

        return issues

    @staticmethod
    def _has_usable_api_key_value(value: str) -> bool:
        """Return True when the raw env value contains at least one usable key."""
        return bool([segment.strip() for segment in str(value or "").split(",") if segment.strip()])

    @classmethod
    def _collect_llm_channel_models_from_map(cls, effective_map: Dict[str, str]) -> List[str]:
        """Collect normalized model names from channel-style env values."""
        raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
        if not raw_channels:
            return []

        models: List[str] = []
        seen: Set[str] = set()
        for raw_name in raw_channels.split(","):
            name = raw_name.strip()
            if not name:
                continue

            prefix = f"LLM_{name.upper()}"
            enabled = parse_env_bool(effective_map.get(f"{prefix}_ENABLED"), default=True)
            if not enabled:
                continue

            base_url_value = (effective_map.get(f"{prefix}_BASE_URL") or "").strip()
            protocol_value = (effective_map.get(f"{prefix}_PROTOCOL") or "").strip()
            raw_models = [
                model.strip()
                for model in (effective_map.get(f"{prefix}_MODELS") or "").split(",")
                if model.strip()
            ]
            validation_issues = cls._validate_llm_channel_definition(
                channel_name=name,
                protocol_value=protocol_value,
                base_url_value=base_url_value,
                api_key_value=(
                    (effective_map.get(f"{prefix}_API_KEYS") or "").strip()
                    or (effective_map.get(f"{prefix}_API_KEY") or "").strip()
                ),
                model_values=raw_models,
                enabled=enabled,
                field_prefix=prefix,
                require_complete=True,
            )
            if any(issue["severity"] == "error" for issue in validation_issues):
                continue
            resolved_protocol = resolve_llm_channel_protocol(protocol_value, base_url=base_url_value, models=raw_models, channel_name=name)
            for model in raw_models:
                normalized_model = normalize_llm_channel_model(model, resolved_protocol, base_url_value)
                if not normalized_model or normalized_model in seen:
                    continue
                seen.add(normalized_model)
                models.append(normalized_model)

        return models

    @staticmethod
    def _uses_litellm_yaml(effective_map: Dict[str, str]) -> bool:
        """Return True when a valid LiteLLM YAML config takes precedence over channels."""
        config_path = (effective_map.get("LITELLM_CONFIG") or "").strip()
        if not config_path:
            return False
        return bool(Config._parse_litellm_yaml(config_path))

    @staticmethod
    def _collect_yaml_models_from_map(effective_map: Dict[str, str]) -> List[str]:
        """Collect declared router model names from LiteLLM YAML config."""
        config_path = (effective_map.get("LITELLM_CONFIG") or "").strip()
        if not config_path:
            return []
        return get_configured_llm_models(Config._parse_litellm_yaml(config_path))

    @classmethod
    def _has_legacy_key_for_provider(cls, provider: str, effective_map: Dict[str, str]) -> bool:
        """Return True when legacy env config can still back the provider."""
        normalized_provider = canonicalize_llm_channel_protocol(provider)
        if normalized_provider in {"gemini", "vertex_ai"}:
            return cls._has_usable_api_key_value(
                (effective_map.get("GEMINI_API_KEYS") or "").strip()
                or (effective_map.get("GEMINI_API_KEY") or "").strip()
            )
        if normalized_provider == "anthropic":
            return cls._has_usable_api_key_value(
                (effective_map.get("ANTHROPIC_API_KEYS") or "").strip()
                or (effective_map.get("ANTHROPIC_API_KEY") or "").strip()
            )
        if normalized_provider == "deepseek":
            return cls._has_usable_api_key_value(
                (effective_map.get("DEEPSEEK_API_KEYS") or "").strip()
                or (effective_map.get("DEEPSEEK_API_KEY") or "").strip()
            )
        if normalized_provider == "openai":
            return cls._has_usable_api_key_value(
                (effective_map.get("OPENAI_API_KEYS") or "").strip()
                or (effective_map.get("AIHUBMIX_KEY") or "").strip()
                or (effective_map.get("OPENAI_API_KEY") or "").strip()
            )
        return False

    @classmethod
    def _has_runtime_source_for_model(cls, model: str, effective_map: Dict[str, str]) -> bool:
        """Whether the selected model still has a backing runtime source."""
        normalized_model = (model or "").strip()
        if not normalized_model:
            return False

        provider = _get_litellm_provider(normalized_model)
        if not provider:
            return False

        for env_key in cls._api_key_candidates_for_provider(provider):
            if cls._has_usable_api_key_value((effective_map.get(env_key) or "").strip()):
                return True

        provider_base_url = ""
        if canonicalize_llm_channel_protocol(provider) == "openai":
            provider_base_url = (effective_map.get("OPENAI_BASE_URL") or "").strip()
        return channel_allows_empty_api_key(provider, provider_base_url)

    @staticmethod
    def _validate_llm_runtime_selection(effective_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate selected primary/fallback/vision models against configured channels."""
        issues: List[Dict[str, Any]] = []

        available_models = (
            SystemConfigService._collect_yaml_models_from_map(effective_map)
            or SystemConfigService._collect_llm_channel_models_from_map(effective_map)
        )
        available_model_set = set(available_models)
        if not available_model_set:
            raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
            if not raw_channels:
                return issues

            configured_agent_model_raw = (effective_map.get("AGENT_LITELLM_MODEL") or "").strip()
            configured_agent_model = normalize_agent_litellm_model(
                configured_agent_model_raw,
                configured_models=available_model_set,
            )
            primary_model = (effective_map.get("LITELLM_MODEL") or "").strip()
            if primary_model and not SystemConfigService._has_runtime_source_for_model(primary_model, effective_map):
                issues.append(
                    {
                        "key": "LITELLM_MODEL",
                        "code": "missing_runtime_source",
                        "message": (
                            "A primary model is selected, but no usable runtime source was found. "
                            "Enable at least one channel with available models, or provide the "
                            "matching provider API key so the model can be resolved."
                        ),
                        "severity": "error",
                        "expected": "enabled channel model or matching legacy API key",
                        "actual": primary_model,
                    }
                )

            if (
                configured_agent_model_raw
                and configured_agent_model
                and not SystemConfigService._has_runtime_source_for_model(
                    configured_agent_model,
                    effective_map,
                )
            ):
                issues.append(
                    {
                        "key": "AGENT_LITELLM_MODEL",
                        "code": "missing_runtime_source",
                        "message": (
                            "An Agent primary model is selected, but no usable runtime source was found. "
                            "Enable at least one channel with available models, or provide the "
                            "matching provider API key so the model can be resolved."
                        ),
                        "severity": "error",
                        "expected": "enabled channel model or matching legacy API key",
                        "actual": configured_agent_model,
                    }
                )

            fallback_models = [
                model.strip()
                for model in (effective_map.get("LITELLM_FALLBACK_MODELS") or "").split(",")
                if model.strip()
            ]
            invalid_fallbacks = [
                model for model in fallback_models
                if not SystemConfigService._has_runtime_source_for_model(model, effective_map)
            ]
            if invalid_fallbacks:
                issues.append(
                    {
                        "key": "LITELLM_FALLBACK_MODELS",
                        "code": "missing_runtime_source",
                        "message": (
                            "Some fallback models do not have an enabled channel "
                            "or matching API key available"
                        ),
                        "severity": "error",
                        "expected": "enabled channel models or matching legacy API keys",
                        "actual": ", ".join(invalid_fallbacks[:3]),
                    }
                )

            vision_model = (effective_map.get("VISION_MODEL") or "").strip()
            if vision_model and not SystemConfigService._has_runtime_source_for_model(vision_model, effective_map):
                issues.append(
                    {
                        "key": "VISION_MODEL",
                        "code": "missing_runtime_source",
                        "message": (
                            "A Vision model is selected, but there is no enabled channel "
                            "or matching API key available for it"
                        ),
                        "severity": "warning",
                        "expected": "enabled channel model or matching legacy API key",
                        "actual": vision_model,
                    }
                )

            return issues

        primary_model = (effective_map.get("LITELLM_MODEL") or "").strip()
        if primary_model and primary_model not in available_model_set and not _uses_direct_env_provider(primary_model):
            issues.append(
                {
                    "key": "LITELLM_MODEL",
                    "code": "unknown_model",
                    "message": (
                        "The selected primary model is not declared by the current enabled channels "
                        "or advanced model routing config. "
                        f"Available models: {', '.join(available_models[:6])}"
                    ),
                    "severity": "error",
                    "expected": "one configured channel model",
                    "actual": primary_model,
                }
            )

        configured_agent_model_raw = (effective_map.get("AGENT_LITELLM_MODEL") or "").strip()
        configured_agent_model = normalize_agent_litellm_model(
            configured_agent_model_raw,
            configured_models=available_model_set,
        )
        if (
            configured_agent_model_raw
            and configured_agent_model
            and configured_agent_model not in available_model_set
            and not _uses_direct_env_provider(configured_agent_model)
        ):
            issues.append(
                {
                    "key": "AGENT_LITELLM_MODEL",
                    "code": "unknown_model",
                    "message": (
                        "The selected Agent primary model is not declared by the current enabled channels "
                        "or advanced model routing config. "
                        f"Available models: {', '.join(available_models[:6])}"
                    ),
                    "severity": "error",
                    "expected": "one configured channel model",
                    "actual": configured_agent_model,
                }
            )

        fallback_models = [
            model.strip()
            for model in (effective_map.get("LITELLM_FALLBACK_MODELS") or "").split(",")
            if model.strip()
        ]
        invalid_fallbacks = [
            model for model in fallback_models
            if model not in available_model_set and not _uses_direct_env_provider(model)
        ]
        if invalid_fallbacks:
            issues.append(
                {
                    "key": "LITELLM_FALLBACK_MODELS",
                    "code": "unknown_model",
                    "message": (
                        "Fallback models include entries that are not declared by the current enabled channels "
                        "or advanced model routing config"
                    ),
                    "severity": "error",
                    "expected": ",".join(available_models[:6]),
                    "actual": ", ".join(invalid_fallbacks[:3]),
                }
            )

        vision_model = (effective_map.get("VISION_MODEL") or "").strip()
        if vision_model and vision_model not in available_model_set and not _uses_direct_env_provider(vision_model):
            issues.append(
                {
                    "key": "VISION_MODEL",
                    "code": "unknown_model",
                    "message": (
                        "The selected Vision model is not declared by the current enabled channels "
                        "or advanced model routing config"
                    ),
                    "severity": "warning",
                    "expected": ",".join(available_models[:6]),
                    "actual": vision_model,
                }
            )

        return issues

    @staticmethod
    def _validate_llm_channel_definition(
        *,
        channel_name: str,
        protocol_value: str,
        base_url_value: str,
        api_key_value: str,
        model_values: Sequence[str],
        enabled: bool,
        field_prefix: str,
        require_complete: bool,
    ) -> List[Dict[str, Any]]:
        """Validate one normalized LLM channel definition."""
        if not require_complete:
            return []

        issues, resolved_protocol = SystemConfigService._validate_llm_channel_connection(
            channel_name=channel_name,
            protocol_value=protocol_value,
            base_url_value=base_url_value,
            api_key_value=api_key_value,
            model_values=model_values,
            field_prefix=field_prefix,
            require_base_url=False,
        )
        models_key = f"{field_prefix}_MODELS" if field_prefix != "test_channel" else "models"

        if not model_values:
            issues.append(
                {
                    "key": models_key,
                    "code": "missing_models",
                    "message": f"LLM channel '{channel_name}' requires at least one model",
                    "severity": "error",
                    "expected": "comma-separated model list",
                    "actual": "",
                }
            )
        elif not resolved_protocol:
            unresolved = [model for model in model_values if "/" not in model]
            if unresolved:
                issues.append(
                    {
                        "key": models_key,
                        "code": "missing_protocol",
                        "message": (
                            f"LLM channel '{channel_name}' uses bare model names. "
                            "Set PROTOCOL or add provider/model prefixes."
                        ),
                        "severity": "error",
                        "expected": "protocol or provider/model",
                        "actual": ", ".join(unresolved[:3]),
                    }
                )

        return issues

    @staticmethod
    def _validate_llm_channel_connection(
        *,
        channel_name: str,
        protocol_value: str,
        base_url_value: str,
        api_key_value: str,
        model_values: Sequence[str] = (),
        field_prefix: str,
        require_base_url: bool,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Validate connection-level fields shared by test and discovery flows."""
        issues: List[Dict[str, Any]] = []
        protocol_key = f"{field_prefix}_PROTOCOL" if field_prefix != "test_channel" else "protocol"
        base_url_key = f"{field_prefix}_BASE_URL" if field_prefix != "test_channel" else "base_url"
        api_key_key = f"{field_prefix}_API_KEY" if field_prefix != "test_channel" else "api_key"

        normalized_protocol = canonicalize_llm_channel_protocol(protocol_value)
        if normalized_protocol and normalized_protocol not in SUPPORTED_LLM_CHANNEL_PROTOCOLS:
            issues.append(
                {
                    "key": protocol_key,
                    "code": "invalid_protocol",
                    "message": (
                        f"Unsupported LLM channel protocol '{protocol_value}'. "
                        f"Supported: {', '.join(SUPPORTED_LLM_CHANNEL_PROTOCOLS)}"
                    ),
                    "severity": "error",
                    "expected": ",".join(SUPPORTED_LLM_CHANNEL_PROTOCOLS),
                    "actual": protocol_value,
                }
            )

        if require_base_url and not base_url_value.strip():
            issues.append(
                {
                    "key": base_url_key,
                    "code": "missing_base_url",
                    "message": f"LLM channel '{channel_name}' requires a base URL to discover models",
                    "severity": "error",
                    "expected": "http(s)://host/v1",
                    "actual": "",
                }
            )
        elif base_url_value and not SystemConfigService._is_valid_url(
            base_url_value,
            allowed_schemes=("http", "https"),
        ):
            issues.append(
                {
                    "key": base_url_key,
                    "code": "invalid_url",
                    "message": "LLM channel base URL must be a valid absolute URL",
                    "severity": "error",
                    "expected": "http(s)://host",
                    "actual": base_url_value,
                }
            )
        elif base_url_value and not SystemConfigService._is_safe_base_url(base_url_value):
            issues.append(
                {
                    "key": base_url_key,
                    "code": "ssrf_blocked",
                    "message": "LLM channel base URL points to a restricted address (cloud metadata services are not allowed)",
                    "severity": "error",
                    "expected": "publicly reachable or local LLM endpoint",
                    "actual": base_url_value,
                }
            )

        resolved_protocol = resolve_llm_channel_protocol(
            protocol_value,
            base_url=base_url_value,
            models=list(model_values) if model_values else None,
            channel_name=channel_name,
        )
        # Validate parsed key segments so that inputs like "," or " , " are
        # treated as empty (they produce zero usable keys after split+strip).
        _parsed_api_keys = [seg.strip() for seg in api_key_value.split(",") if seg.strip()]
        if not _parsed_api_keys and not channel_allows_empty_api_key(resolved_protocol, base_url_value):
            issues.append(
                {
                    "key": api_key_key,
                    "code": "missing_api_key",
                    "message": f"LLM channel '{channel_name}' requires an API key",
                    "severity": "error",
                    "expected": "non-empty API key",
                    "actual": api_key_value,
                }
            )
        return issues, resolved_protocol
