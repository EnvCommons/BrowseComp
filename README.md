# BrowseComp - OpenReward Env

OpenReward evaluation environment for **BrowseComp**, a web search reasoning benchmark from OpenAI's simple-evals.

## Overview

BrowseComp is a challenging benchmark containing **1,266 encrypted research questions** designed to test an agent's ability to use web search effectively. Questions require multi-hop reasoning and often cannot be answered without current web information.

### Key Features

- **Single-turn evaluation**: Agent receives question → researches → submits answer
- **Encrypted dataset**: Questions/answers encrypted with per-task passwords to prevent test set leakage
- **LLM-based grading**: Semantic comparison using gpt-5-mini (not exact string matching)
- **Built-in web tools**: Environment provides `web_search` and `fetch_url` tools powered by Tavily
- **Confidence tracking**: Agents report confidence levels for calibration analysis

### Baseline Performance

- **GPT-4o without browsing**: 0.9% accuracy
- **GPT-4o with browsing**: 2.5% accuracy (2.8x improvement)

This benchmark is intentionally difficult - even state-of-the-art models with web access achieve low scores.

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/EnvCommons/BrowseComp.git
cd BrowseComp

# Install dependencies
pip install -r requirements.txt

# Download data
curl -o browse_comp_test_set.csv https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv

# Run server
python server.py
```

### Docker

```bash
# Build image
docker build -t browsecomp:latest .

# Run container
docker run -p 8080:8080 browsecomp:latest
```

## Usage

### Quick Start

```bash
# Set API keys (required for grading and web search)
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."

# Start server
python server.py

# In another terminal, run test agent
python test_agent.py
```

### Example Agent Interaction

```python
import asyncio
import json
from openai import AsyncOpenAI
from openreward import OpenReward

async def run_browsecomp():
    or_client = OpenReward()
    oai_client = AsyncOpenAI()

    # Connect to environment
    env = or_client.environments.get(name="EnvCommons/BrowseComp")
    tasks = await env.list_tasks(split="test")

    # Get first task
    task = tasks[0]

    async with env.session(
        task=task,
        secrets={
            "openai_api_key": "sk-...",
            "tavily_api_key": "tvly-..."
        }
    ) as session:
        # Get question prompt
        prompt = await session.get_prompt()
        print(prompt[0].text)

        # Agent uses web_search tool to find information
        search_result = await session.call_tool("web_search", {
            "query": "search query here"
        })

        # Agent uses fetch_url tool to get full page content
        fetch_result = await session.call_tool("fetch_url", {
            "url": "https://example.com/relevant-page"
        })

        # Agent submits final answer
        result = await session.call_tool("submit_answer", {
            "explanation": "Based on my research across multiple sources...",
            "exact_answer": "The precise answer",
            "confidence": 0.85
        })

        print(f"Reward: {result.reward}")
        print(result.blocks[0].text)

asyncio.run(run_browsecomp())
```

## Environment Details

### Splits
- **test**: 1,266 tasks

### Tools
- **web_search**: Search the web for information (powered by Tavily)
  - `query` (string): Search query
  - Returns: Search results with titles, URLs, and snippets
- **fetch_url**: Fetch full content from a specific URL (powered by Tavily)
  - `url` (string): URL to fetch
  - Returns: Full text content from the page (truncated to 8000 chars)
- **submit_answer**: Submit final answer with explanation and confidence
  - `explanation` (string): Research summary and reasoning (2-4 sentences)
  - `exact_answer` (string): Precise, concise answer
  - `confidence` (float): Confidence level 0.0-1.0

### Secrets
- **openai_api_key** (required): Used for LLM-based grading with gpt-5-mini
- **tavily_api_key** (required): Used for web search and URL fetching

### Rewards
- **1.0**: Correct answer (semantically equivalent to reference)
- **0.0**: Incorrect answer

## Architecture

### Design Pattern
- Extends `Environment` (single-turn, no sandbox)
- Follows AIME2025 reference implementation pattern
- Environment class + server wrapper in `server.py`

### Data Flow
1. **Load**: CSV decrypted at module import time using SHA256 + XOR
2. **Prompt**: Agent receives question with instructions
3. **Research**: Agent uses web_search and fetch_url tools (Tavily-powered, environment tools)
4. **Submit**: Agent calls submit_answer with findings
5. **Grade**: Environment uses gpt-5-mini to evaluate semantic match
6. **Reward**: Return 1.0 (correct) or 0.0 (incorrect) with feedback

### Path Handling
```python
# Production: /orwd_data/browsecomp/browse_comp_test_set.csv
# Local dev: ./browse_comp_test_set.csv (fallback)
```

The environment automatically detects which path to use via `constants.py`.

## Data Requirements

See [DATA_UPLOAD.md](DATA_UPLOAD.md) for instructions on:
- Downloading the encrypted dataset
- Uploading to OpenReward cloud storage
- Verification steps

**Quick summary**:
- Download from: `https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv`
- Production path: `/orwd_data/browsecomp/browse_comp_test_set.csv`
- Size: 1.1 MB, 1,266 rows

## Example Task

### Question
```
"What was the name of the 1995 film starring the actress who played
Victoria that married the final scorer of the World Cup 1998 winner?"
```

### Expected Agent Flow
1. Search: "World Cup 1998 final scorer"
   - Find: Emmanuel Petit scored the final goal
2. Search: "Emmanuel Petit wife actress"
   - Find: Married to Agathe de La Fontaine
3. Search: "Agathe de La Fontaine 1995 films Victoria"
   - Find: Played Victoria in "French Kiss" (1995)
4. Submit answer: "French Kiss"

### Grading
- LLM judge compares submitted answer against reference answer
- Allows semantic equivalence (e.g., "French Kiss" = "French Kiss (1995 film)")
- Temperature 0.0 for deterministic grading

## File Structure

```
BrowseComp/
├── server.py                    # Environment class + server wrapper
├── decrypt.py                   # SHA256 + XOR decryption utilities
├── constants.py                 # Path handling logic
├── test_agent.py                # Agent test runner
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── DATA_UPLOAD.md              # Data upload instructions
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
└── browse_comp_test_set.csv    # Dataset (local dev only)
```

## Development

### Running Tests

```bash
# Syntax check
python -m py_compile *.py

# Test decryption
python -c "from decrypt import decrypt_task; import pandas as pd; df = pd.read_csv('browse_comp_test_set.csv'); print(decrypt_task(df.iloc[0]['problem'], df.iloc[0]['answer'], df.iloc[0]['canary']))"

# Test local server
python server.py  # Should start on http://0.0.0.0:8080

# Test with agent
export OPENAI_API_KEY="sk-..."
python test_agent.py
```

### Docker Testing

```bash
# Build
docker build -t browsecomp:test .

# Run
docker run -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY browsecomp:test

# Test connection
curl http://localhost:8080/health  # (if health endpoint exists)
```

## Tool Architecture

All tools (`web_search`, `fetch_url`, `submit_answer`) are **environment tools** powered by Tavily and OpenAI APIs:

- **web_search**: Uses Tavily's search API to find relevant web pages
- **fetch_url**: Uses Tavily's extract API to get full content from URLs
- **submit_answer**: Uses gpt-5-mini for semantic grading

Agents call all tools through the environment session:
```python
# All tools are environment tools
tools = await environment.list_tools(format="openai")

# Call any tool through the session
result = await session.call_tool(tool_name, parameters)
```

## Security & Privacy

### Encrypted Dataset
- Questions/answers encrypted using per-task canary passwords
- SHA256 key derivation + XOR cipher
- **Do not share decrypted questions publicly** (per OpenAI's request)

### Secrets Management
- OpenAI API key required via `secrets` parameter
- Never falls back to environment variables
- Fails fast if key missing

## Troubleshooting

### "CSV not found" error
```bash
# Check file exists
ls -lh browse_comp_test_set.csv

# Download if missing
curl -o browse_comp_test_set.csv https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
```

### "openai_api_key required" or "tavily_api_key required" error
```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."

# Or pass in session creation
async with env.session(
    task=task,
    secrets={
        "openai_api_key": "sk-...",
        "tavily_api_key": "tvly-..."
    }
) as session:
    ...
```

### Grading failures
- Check gpt-5-mini model access
- Verify API key has sufficient credits
- Check rate limits

### Docker build issues
- Ensure Docker has internet access (downloads uv installer)
- Check base image pulls correctly: `docker pull ubuntu:22.04`

## Citation

If you use BrowseComp in your research, please cite:

```bibtex
@misc{openai-simple-evals,
  title={Simple Evals},
  author={OpenAI},
  year={2024},
  url={https://github.com/openai/simple-evals}
}
```

## License

MIT License

## References

- **OpenAI simple-evals**: https://github.com/openai/simple-evals
- **BrowseComp evaluation**: https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
- **Dataset**: https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv
- **OpenReward Platform**: https://openreward.ai
- **OpenReward Docs**: https://docs.openreward.org

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/EnvCommons/BrowseComp/issues
- **OpenReward Discord**: https://discord.gg/openreward
- **Documentation**: https://docs.openreward.org
