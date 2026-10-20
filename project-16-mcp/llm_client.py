import asyncio
import json

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
client = OpenAI()


async def main():
    server = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            openai_tools = []

            for tool in tools.tools:
                openai_tools.append({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema
                })

            print("Tools available to AI:")
            print([tool["name"] for tool in openai_tools])

            response = client.responses.create(
                model="gpt-4.1-mini",
                input="How many public repositories does DrinSadikuu have on GitHub?",
                tools=openai_tools
            )

            tool_outputs = []

            for item in response.output:
                if item.type == "function_call":
                    print("AI selected tool:", item.name)
                    print("Arguments:", item.arguments)

                    arguments = json.loads(item.arguments)

                    result = await session.call_tool(
                        item.name,
                        arguments=arguments
                    )

                    print("Tool result:", result.content)

                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(
                            [content.model_dump() for content in result.content]
                        )
                    })

            if tool_outputs:
                final_response = client.responses.create(
                    model="gpt-4.1-mini",
                    previous_response_id=response.id,
                    input=tool_outputs,
                    tools=openai_tools
                )

                print("AI answer:", final_response.output_text)
            else:
                print("AI answer:", response.output_text)


asyncio.run(main())