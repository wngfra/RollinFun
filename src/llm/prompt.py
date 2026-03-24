"""Assemble LLM context for Ollama chat API."""

from __future__ import annotations


def assemble_context(
    *,
    system_prompt: str,
    act_title: str | None = None,
    beat_id: str | None = None,
    objectives: list[str] | None = None,
    completed_beats: list[str] | None = None,
    relationships: dict[str, str] | None = None,
    discovered_secrets: list[str] | None = None,
    current_location: str | None = None,
    memory_summary: str | None = None,
    rag_results: list[str] | None = None,
    recent_turns: list[dict[str, str]] | None = None,
    user_input: str,
) -> list[dict[str, str]]:
    """Build the full message list for an Ollama /api/chat call.

    Returns a list of ``{"role": ..., "content": ...}`` dicts.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Build per-turn context injection (prepended to user message)
    injection_parts: list[str] = []

    # Story progress
    if act_title or beat_id:
        lines = ["[STORY PROGRESS]"]
        if act_title:
            lines.append(f"Act: {act_title}" + (f" | Beat: {beat_id}" if beat_id else ""))
        if objectives:
            lines.append("Objectives: " + "; ".join(objectives))
        if completed_beats:
            lines.append("Completed: " + ", ".join(completed_beats))
        if relationships:
            rel_strs = [f"{k}: {v}" for k, v in relationships.items()]
            lines.append("Relationships: " + "; ".join(rel_strs))
        if discovered_secrets:
            lines.append("Discovered secrets: " + ", ".join(discovered_secrets))
        if current_location:
            lines.append(f"Location: {current_location}")
        injection_parts.append("\n".join(lines))

    # Memory summary
    if memory_summary:
        injection_parts.append(f"[MEMORY]\n{memory_summary}")

    # RAG results
    if rag_results:
        injection_parts.append("[RECALLED]\n" + "\n".join(rag_results))

    # Recent turns
    if recent_turns:
        recent = recent_turns[:]
        messages.extend(recent)

    # User input with injection prefix
    if injection_parts:
        prefix = "\n\n".join(injection_parts) + "\n\n"
        messages.append({"role": "user", "content": prefix + user_input})
    else:
        messages.append({"role": "user", "content": user_input})

    return messages
