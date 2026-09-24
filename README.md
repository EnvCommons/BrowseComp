# BrowseComp

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/BrowseComp)

## Description

BrowseComp is an environment for evaluating web search reasoning capabilities. Based on OpenAI's [simple-evals](https://github.com/openai/simple-evals) benchmark, it contains 1,266 encrypted research questions that require multi-hop reasoning and cannot be answered without current web information. The environment provides built-in web search and URL fetching tools powered by OpenReward's backdated search corpus (backsearch), with the cutoff set to the UTC date each session starts.

## Capabilities

- Multi-hop web search reasoning
- Information retrieval and synthesis
- Research question answering
- Confidence calibration

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT)

## Tasks

There is one split in this environment:

- **test**: 1,266 encrypted research questions

Questions require multi-hop reasoning across multiple web searches. Example: "What was the name of the 1995 film starring the actress who played Victoria that married the final scorer of the World Cup 1998 winner?"

## Reward Structure

This is a sparse reward environment with LLM-based grading:

1. Agent receives a research question
2. Agent uses `web_search` and `web_fetch` tools to gather information
3. Agent submits answer with explanation, exact_answer, and confidence
4. An LLM grader (gpt-5-mini) evaluates semantic equivalence
5. Binary reward: 1.0 if correct, 0.0 if incorrect

## Data

Data is sourced from [OpenAI's BrowseComp benchmark](https://openai.com/index/browsecomp/). Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the backsearch corpus as of the session's start date (returns titles, URLs, snippets) |
| `web_fetch` | Fetch an archived capture of a URL, extracting what the prompt asks for |
| `submit_answer` | Submit answer with explanation, exact_answer, and confidence |

Note that the `web_fetch` and `web_search` tools require an OpenReward API key, but are optional. If you want to use a different provider for search you can exclude these tools and use external tools instead.

## Time Horizon

Multi-turn. Agents can perform multiple web searches before submitting a final answer.

## Environment Difficulty

| Model | Accuracy |
|-------|----------|
| Gemini 3.1 Pro (search, Python, browse) | 85.9% |
| Claude Opus 4.6 | 84.0% |
| Kimi K2.5 (agent swarm) | 78.4% |
| MiniMax M2.5 | 76.3% |
| GLM-5 (with ctx management) | 75.9% |

This benchmark requires persistent multi-hop web navigation to find hard-to-find, entangled information.

## Other Environment Requirements

- OpenAI API key required for LLM-based grading
- OpenReward API key required for web search (falls back to the server's `OPENREWARD_API_KEY`)

Pass via `secrets={"openai_api_key": "...", "api_key": "..."}`.

## Safety

Agents in BrowseComp perform web searches and answer research questions. The environment does not present direct safety risks. Note: Do not share decrypted questions publicly per OpenAI's request.

## Citation

```bibtex
@article{wei2025browsecomp,
  title={BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents},
  author={Wei, Jason and Sun, Zhiqing and Papay, Spencer and McKinney, Scott and Han, Jeffrey and Fulford, Isa and Chung, Hyung Won and Passos, Alex Tachard and Fedus, William and Glaese, Amelia},
  journal={arXiv preprint arXiv:2504.12516},
  year={2025},
  url={https://arxiv.org/abs/2504.12516}
}
```
