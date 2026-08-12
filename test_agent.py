"""
Test agent for BrowseComp environment.

Demonstrates how to use the BrowseComp environment with OpenAI's Responses API
and web_search tool for testing locally.
"""

import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def main():
    """
    Test the BrowseComp environment with a sample task.

    This script:
    1. Connects to local BrowseComp environment server
    2. Gets available tools (submit_answer from env + web_search from OpenAI)
    3. Runs the agent on the first task
    4. Displays results and reward
    """
    # Configuration
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    ENV_NAME = "local/BrowseComp"
    SPLIT = "test"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Optional: whichever the server's OPENREWARD_SEARCH_BACKEND needs.
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    OPENREWARD_API_KEY = os.getenv("OPENREWARD_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable required. "
            "Set with: export OPENAI_API_KEY='sk-...'"
        )


    # Initialize clients
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Connect to local environment server
    print(f"Connecting to environment: {ENV_NAME}")
    environment = or_client.environments.get(
        name=ENV_NAME,
        base_url="http://localhost:8080"
    )

    # Get tasks and tools
    tasks = await environment.list_tasks(split=SPLIT)
    print(f"Found {len(tasks)} tasks")

    # Get environment tools (web_search, web_fetch, submit_answer)
    tools = await environment.list_tools(format="openai")


    # Test first task
    print(f"\n{'='*80}")
    print("Testing first task")
    print('='*80)

    task = tasks[0]
    print(task)

    finished = False

    async with environment.session(
        task=task,
        secrets={
            "openai_api_key": OPENAI_API_KEY,
            **({"tavily_api_key": TAVILY_API_KEY} if TAVILY_API_KEY else {}),
            **({"api_key": OPENREWARD_API_KEY} if OPENREWARD_API_KEY else {}),
        }
    ) as session:
        # Get prompt
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text

        print(f"\nPrompt preview (first 200 chars):")
        print(prompt_text[:200] + "...")

        # Initialize conversation
        input_list = [{"role": "user", "content": prompt_text}]

        turn = 0
        max_turns = 20  # Prevent infinite loops

        while not finished and turn < max_turns:
            turn += 1
            print(f"\n--- Turn {turn} ---")

            # Call model with tools
            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            # Process response output
            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    # All tools are environment tools now (web_search, web_fetch, submit_answer)
                    print(f"🛠️  Tool call: {item.name}")

                    # Show arguments for non-verbose tools
                    if item.name == "submit_answer":
                        print(f"   Arguments: {item.arguments}")
                    elif item.name == "web_search":
                        args = json.loads(str(item.arguments))
                        print(f"   Query: {args.get('query', '')}")
                    elif item.name == "web_fetch":
                        args = json.loads(str(item.arguments))
                        print(f"   URL: {args.get('url', '')}")

                    # Call environment tool
                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments))
                    )

                    reward = tool_result.reward
                    finished = tool_result.finished

                    # Add tool result to conversation
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result.blocks[0].text if tool_result.blocks else ""
                    })

                    # Show result summary
                    if item.name == "submit_answer":
                        print(f"\n📊 Result:")
                        print(f"   Reward: {reward:.1f}")
                        print(f"   Finished: {finished}")

                        if tool_result.blocks:
                            print(f"\n📝 Feedback:")
                            print("   " + "\n   ".join(tool_result.blocks[0].text.split("\n")))
                    else:
                        # For web_search and web_fetch, show abbreviated output
                        if tool_result.blocks:
                            output_preview = tool_result.blocks[0].text[:150].replace("\n", " ")
                            print(f"   Result: {output_preview}...")

                    if finished:
                        print("\n✅ Episode finished!")
                        break

                elif item.type == "message":
                    # Model's text response (before tool calls)
                    if hasattr(item, 'content') and item.content:
                        content_text = item.content[0].text if item.content else ""
                        if content_text:
                            print(f"💬 Model: {content_text[:100]}...")

            # Check if we got a response without function calls (shouldn't happen)
            if not any(i.type == "function_call" for i in response.output):
                print("⚠️  Model didn't call any tools. Breaking loop.")
                break

        if turn >= max_turns:
            print(f"\n⚠️  Reached maximum turns ({max_turns})")

    print(f"\n{'='*80}")
    print("Test completed!")
    print('='*80)


if __name__ == "__main__":
    asyncio.run(main())
