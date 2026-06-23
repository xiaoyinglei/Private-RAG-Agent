"""LLM prompt templates for persistent memory operations."""

# ── Selector prompt ──

MEMORY_SELECT_PROMPT = """You are a memory selector. Given a user task and a list of available memories,
select the {max_memories} most relevant memories that would help complete the task.

Memory types:
- user: Who the user is (role, expertise, preferences) — always relevant
- feedback: Guidance on how to work — relevant for code generation tasks
- project: Ongoing work, goals, constraints — relevant for project-specific tasks
- reference: External resource pointers — relevant when the task matches the topic

User task:
{task}

Available memories (index):
{index_entries}

Return ONLY the memory names, one per line, that are relevant to the task.
Order by relevance (most relevant first).
If none are relevant, return "NONE"."""

# ── Extractor prompt ──

MEMORY_EXTRACT_PROMPT = """You are a memory extractor. Analyze the conversation and extract durable facts
worth remembering for future sessions.

Memory types:
- user: Who the user is (role, expertise, preferences)
- feedback: Guidance the user gave on how to work
- project: Ongoing work, goals, constraints not derivable from the code
- reference: Pointers to external resources (URLs, dashboards, tickets)

Rules:
1. Only extract facts that are durable (not one-time requests)
2. Only extract facts that are non-obvious (not derivable from the code)
3. Each memory must have a clear name (kebab-case), description (one line), and type
4. If no durable facts found, return "NO_MEMORIES"
5. Maximum {max_memories} memories per extraction

Conversation transcript:
{transcript}

Existing memories (avoid duplicates):
{existing_index}

Extracted memories in this format (one per block):

---MEMORY---
name: <kebab-case-name>
description: <one-line summary>
type: <user|feedback|project|reference>
content:
<memory content in markdown>
---END---"""

# ── Similarity check prompt ──

MEMORY_SIMILARITY_PROMPT = """Are these two memories about the same fact or topic?
Reply with ONLY "YES" or "NO".

Memory 1 ({name1}):
{content1}

Memory 2 ({name2}):
{content2}"""

# ── Merge prompt ──

MEMORY_MERGE_PROMPT = """You are a memory merger. Two memories cover overlapping information.
Merge them into a single memory that preserves all unique facts.

Memory 1 ({name1}):
{content1}

Memory 2 ({name2}):
{content2}

Return the merged memory in this format:

---MEMORY---
name: <merged-name>
description: <merged description>
type: <type>
content:
<merged content in markdown>
---END---"""

# ── Consolidator prompt ──

MEMORY_CONSOLIDATE_PROMPT = """You are a memory consolidator. You have {count} memory files.
Your job is to merge duplicates, remove outdated information, and keep the set clean.

Memories:
{memories_text}

For each memory, decide one of:
- KEEP: still relevant and unique (no change needed)
- MERGE: combine with another specific memory (specify which)
- DELETE: outdated, trivial, or fully covered by another memory

Return your decisions in this format (one per block):

---DECISION---
action: <KEEP|MERGE|DELETE>
name: <memory-name>
merge_with: <other-memory-name> (only for MERGE)
reason: <brief explanation>
---END---"""


__all__ = [
    "MEMORY_CONSOLIDATE_PROMPT",
    "MEMORY_EXTRACT_PROMPT",
    "MEMORY_MERGE_PROMPT",
    "MEMORY_SELECT_PROMPT",
    "MEMORY_SIMILARITY_PROMPT",
]
