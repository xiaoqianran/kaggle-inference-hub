from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompts import MODE_LABELS, build_system_prompt, build_user_prompt


class PromptPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptPipelineSettings:
    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    concurrency: int = 4
    max_tokens: int = 900
    temperature: float = 0.35

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.base_url.strip() and self.model.strip())


class PromptPipeline:
    def __init__(self, settings: PromptPipelineSettings):
        self.settings = settings
        self._semaphore = asyncio.Semaphore(max(1, int(settings.concurrency)))

    def public_config(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "configured": self.settings.configured,
            "provider_model": self.settings.model if self.settings.configured else None,
            "concurrency": max(1, int(self.settings.concurrency)),
            "modes": [{"id": key, "label": label} for key, label in MODE_LABELS.items()],
        }

    async def process(
        self,
        source: str,
        *,
        target_model: str,
        mode: str = "enhance",
        translate_to_english: bool = True,
    ) -> dict[str, Any]:
        source = source.strip()
        if not source:
            raise PromptPipelineError("Prompt is empty")
        if mode not in MODE_LABELS:
            raise PromptPipelineError(f"Unknown prompt mode: {mode}")
        if not self.settings.enabled:
            raise PromptPipelineError("AI Prompt Pipeline is disabled")
        if not self.settings.configured:
            raise PromptPipelineError("AI Prompt Pipeline is not configured")

        started = time.perf_counter()
        payload = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(target_model, mode, translate_to_english),
                },
                {"role": "user", "content": build_user_prompt(source)},
            ],
            "temperature": float(self.settings.temperature),
            "max_tokens": int(self.settings.max_tokens),
            "stream": False,
        }

        async with self._semaphore:
            data = await self._chat_completions(payload)

        processed = self._extract_content(data)
        if not processed:
            raise PromptPipelineError("AI returned an empty prompt")

        return {
            "original": source,
            "processed": processed,
            "target_model": target_model,
            "mode": mode,
            "translate_to_english": translate_to_english,
            "provider_model": self.settings.model,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    async def _chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "kaggle-inference-hub/0.3",
        }
        if self.settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await asyncio.to_thread(self._post_json, url, headers, payload)
            except (OSError, ValueError, PromptPipelineError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                await asyncio.sleep(0.5 * (2**attempt))
        raise PromptPipelineError(str(last_error) if last_error else "AI request failed")

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except HTTPError as exc:
            body = exc.read(1200).decode("utf-8", errors="replace")
            raise PromptPipelineError(f"AI HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise PromptPipelineError(f"AI connection failed: {exc.reason}") from exc

    @classmethod
    def _extract_content(cls, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PromptPipelineError("AI response does not contain choices[0].message.content") from exc

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            content = "\n".join(parts)
        if not isinstance(content, str):
            raise PromptPipelineError("AI response content is not text")
        return cls._clean_output(content)

    @staticmethod
    def _clean_output(text: str) -> str:
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```[^\n]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        text = re.sub(r"^(?:final\s+prompt|prompt|optimized\s+prompt)\s*:\s*", "", text, flags=re.I)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            text = text[1:-1].strip()
        return text
