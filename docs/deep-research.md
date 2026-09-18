# Deep Research

Brave Deep Research MCP (to be open sourced on a future date) provides multi-step, iterative web research via a streaming MCP tool. Disabled by default. To enable it locally:

| Environment variable           | Default                 | Description                       |
| ------------------------------ | ----------------------- | --------------------------------- |
| `deep_research_enabled`        | `False`                 | Enable the deep research MCP tool |
| `deep_research_url`            | `http://localhost:3011` | URL of the deep-research-service  |
| `deep_research_max_iterations` | `3`                     | Maximum research iterations       |
| `deep_research_max_queries`    | `30`                    | Maximum total search queries      |
| `deep_research_max_seconds`    | `300`                   | Time budget in seconds            |

1. Start the [deep-research-service](https://github.com/brave/brave-deepresearch) on port 3011 (or set `deep_research_url`).
2. Set `deep_research_enabled=True` in your environment.
3. Start ai-gateway as usual — the `deep_research` tool will appear in the tool list.
