"""Streaming regex parser for LLM tagged output.

Parses the custom tag format into typed segment objects:
  [role mood="x" gesture="y"] text [/role]  → Segment
  [choices] 1. ... [/choices]               → Choices
  [scene location="id"]                     → SceneChange
  Untagged text                             → Segment(role="narrator")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    role: str
    text: str
    mood: str = "neutral"
    gesture: str | None = None
    nonverbals: list[str] = field(default_factory=list)


@dataclass
class Choices:
    options: list[str]


@dataclass
class SceneChange:
    location_id: str


ParsedElement = Union[Segment, Choices, SceneChange]


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches [role ...attrs...] but NOT [/role] or [choices] or [scene ...]
_OPEN_TAG = re.compile(
    r'\[(?P<role>(?!choices\b|scene\b|/)\w+)'
    r'(?P<attrs>(?:\s+\w+="[^"]*")*)\s*\]'
)
_CLOSE_TAG = re.compile(r'\[/(?P<role>\w+)\]')
_CHOICES_BLOCK = re.compile(r'\[choices\](.*?)\[/choices\]', re.DOTALL)
_SCENE_TAG = re.compile(r'\[scene\s+location="(?P<loc>[^"]+)"\s*\]')
_ATTR_PAIR = re.compile(r'(\w+)="([^"]*)"')
_OPTION_LINE = re.compile(r'\d+\.\s*"([^"]+)"')
_NONVERBAL = re.compile(r'\((\w+)\)')

_NONVERBAL_MAP: dict[str, str] = {
    "laughs": "[laugh]",
    "sighs": "[sigh]",
    "coughs": "[cough]",
    "chuckles": "[chuckle]",
}


def _parse_attrs(attr_str: str) -> dict[str, str]:
    return dict(_ATTR_PAIR.findall(attr_str))


def _extract_nonverbals(text: str) -> list[str]:
    return [
        _NONVERBAL_MAP.get(m, f"[{m}]")
        for m in _NONVERBAL.findall(text)
    ]


# ---------------------------------------------------------------------------
# Full-text parser
# ---------------------------------------------------------------------------


def parse_segments(text: str) -> list[ParsedElement]:
    """Parse a complete LLM response into a list of typed elements."""
    elements: list[ParsedElement] = []
    pos = 0

    while pos < len(text):
        # Try scene tag
        m_scene = _SCENE_TAG.match(text, pos)
        if m_scene:
            elements.append(SceneChange(location_id=m_scene.group("loc")))
            pos = m_scene.end()
            continue

        # Try choices block
        m_choices = _CHOICES_BLOCK.match(text, pos)
        if m_choices:
            options = _OPTION_LINE.findall(m_choices.group(1))
            elements.append(Choices(options=options))
            pos = m_choices.end()
            continue

        # Try tagged segment [role ...] ... [/role]
        m_open = _OPEN_TAG.match(text, pos)
        if m_open:
            role = m_open.group("role")
            attrs = _parse_attrs(m_open.group("attrs"))
            close_pat = re.compile(rf'\[/{re.escape(role)}\]')
            m_close = close_pat.search(text, m_open.end())
            if m_close:
                body = text[m_open.end():m_close.start()].strip()
                elements.append(Segment(
                    role=role,
                    text=body,
                    mood=attrs.get("mood", "neutral"),
                    gesture=attrs.get("gesture"),
                    nonverbals=_extract_nonverbals(body),
                ))
                pos = m_close.end()
                continue

        # Accumulate untagged text as narrator
        # Find the next tag start
        next_tag = re.search(r'\[', text[pos:])
        if next_tag:
            chunk = text[pos:pos + next_tag.start()].strip()
            if chunk:
                elements.append(Segment(
                    role="narrator",
                    text=chunk,
                    nonverbals=_extract_nonverbals(chunk),
                ))
            pos += next_tag.start()
            # If we're stuck on a bracket that didn't match any pattern, skip it
            if pos < len(text) and not (
                _OPEN_TAG.match(text, pos)
                or _CHOICES_BLOCK.match(text, pos)
                or _SCENE_TAG.match(text, pos)
            ):
                # Check if it's a closing tag we can skip
                m_close_orphan = _CLOSE_TAG.match(text, pos)
                if m_close_orphan:
                    pos = m_close_orphan.end()
                else:
                    pos += 1
        else:
            chunk = text[pos:].strip()
            if chunk:
                elements.append(Segment(
                    role="narrator",
                    text=chunk,
                    nonverbals=_extract_nonverbals(chunk),
                ))
            break

    return elements


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------


class StreamingParser:
    """Feed chunks of text and emit complete parsed elements."""

    def __init__(self) -> None:
        self._buffer: str = ""

    def feed(self, chunk: str) -> list[ParsedElement]:
        """Append *chunk* to the internal buffer, return any complete elements."""
        self._buffer += chunk
        elements: list[ParsedElement] = []

        while True:
            consumed = self._try_consume(elements)
            if not consumed:
                break

        return elements

    def flush(self) -> list[ParsedElement]:
        """Flush remaining buffer content as narrator text."""
        elements: list[ParsedElement] = []
        text = self._buffer.strip()
        if text:
            elements.append(Segment(
                role="narrator",
                text=text,
                nonverbals=_extract_nonverbals(text),
            ))
            self._buffer = ""
        return elements

    def _try_consume(self, out: list[ParsedElement]) -> bool:
        buf = self._buffer

        # Skip leading whitespace
        stripped = buf.lstrip()
        if not stripped:
            return False

        # Scene tag
        m = _SCENE_TAG.match(stripped)
        if m:
            # Emit any preceding narrator text
            prefix = buf[:len(buf) - len(stripped)]
            if prefix.strip():
                out.append(Segment(
                    role="narrator",
                    text=prefix.strip(),
                    nonverbals=_extract_nonverbals(prefix),
                ))
            out.append(SceneChange(location_id=m.group("loc")))
            self._buffer = stripped[m.end():]
            return True

        # Choices block
        m = _CHOICES_BLOCK.match(stripped)
        if m:
            prefix = buf[:len(buf) - len(stripped)]
            if prefix.strip():
                out.append(Segment(
                    role="narrator",
                    text=prefix.strip(),
                    nonverbals=_extract_nonverbals(prefix),
                ))
            options = _OPTION_LINE.findall(m.group(1))
            out.append(Choices(options=options))
            self._buffer = stripped[m.end():]
            return True

        # Tagged segment
        m_open = _OPEN_TAG.match(stripped)
        if m_open:
            role = m_open.group("role")
            close_pat = re.compile(rf'\[/{re.escape(role)}\]')
            m_close = close_pat.search(stripped, m_open.end())
            if m_close:
                prefix = buf[:len(buf) - len(stripped)]
                if prefix.strip():
                    out.append(Segment(
                        role="narrator",
                        text=prefix.strip(),
                        nonverbals=_extract_nonverbals(prefix),
                    ))
                attrs = _parse_attrs(m_open.group("attrs"))
                body = stripped[m_open.end():m_close.start()].strip()
                out.append(Segment(
                    role=role,
                    text=body,
                    mood=attrs.get("mood", "neutral"),
                    gesture=attrs.get("gesture"),
                    nonverbals=_extract_nonverbals(body),
                ))
                self._buffer = stripped[m_close.end():]
                return True
            # Opening tag but no close yet — wait for more data
            return False

        # If buffer starts with '[' but doesn't match, might be incomplete tag — wait
        if stripped.startswith("["):
            # But if we have a lot of text after '[', it might be just a stray bracket
            # Only wait if buffer is short (likely incomplete tag)
            if len(stripped) < 200:
                return False
            # Skip the bracket
            if buf[:len(buf) - len(stripped)]:
                pass  # fall through to narrator handling below

        # Emit text up to the next '[' as narrator
        bracket_pos = stripped.find("[", 1) if stripped.startswith("[") else stripped.find("[")
        if bracket_pos > 0:
            prefix_offset = len(buf) - len(stripped)
            narrator_text = buf[:prefix_offset] + stripped[:bracket_pos]
            narrator_text = narrator_text.strip()
            if narrator_text:
                out.append(Segment(
                    role="narrator",
                    text=narrator_text,
                    nonverbals=_extract_nonverbals(narrator_text),
                ))
            self._buffer = stripped[bracket_pos:]
            return True

        # No bracket found — can't emit yet, wait for more data
        return False
