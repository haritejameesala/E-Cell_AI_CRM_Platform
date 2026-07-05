# Prompt Templates

This directory stores the prompt templates used by the AI CRM system.

- `system.md`
- `customer_agent.md`
- `ticket_summary.md`
- `factual_router.md`

These prompts correspond to the prompt logic implemented in `src/agents.py`.

The current application still embeds the prompts directly in code. These files document the same prompts separately for maintainability and to align with the suggested project structure.

Runtime behavior is unchanged: `src/agents.py` does not read these files yet.
