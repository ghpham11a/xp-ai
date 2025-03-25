import os
import asyncio
import PIL
import requests

from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

from io import BytesIO
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image

from tavily import TavilyClient

load_dotenv()

async def main():

    #### CONFIGURE LLM ####
    openai_model_client = OpenAIChatCompletionClient(
        model="gpt-4o-2024-08-06",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    #### CREATE TOOLS ####

     # Tavily Search.
    async def tavily_search(query: str) -> str:

        client = TavilyClient(os.environ["TAVILY_API_KEY"])
        response = client.search(
            query=query
        )
        if len(response["results"]) > 0:
            return response["results"][0]["content"]
        else:
            return "Sorry, I couldn't find any relevant information."
    
    #### AGGREGATE TOOLS ####
    tools = [tavily_search]

    #### CREATE AGENTS ####
    agent = AssistantAgent(
        name="assistant",
        model_client=openai_model_client,
        tools=tools,
        system_message="Use tools to solve tasks.",
    )

    # It is important to note that on_messages() will update the internal state of the 
    # agent – it will add the messages to the agent’s history. So you should call 
    # this method with new messages. You should not repeatedly call this method 
    # with the same messages or the complete history.

    async def assistant_on_text_message(content: str) -> None:

        # Option 1: read each message from the stream (as shown in the previous example).
        # response = await agent.on_messages(
        #     [TextMessage(content=content, source="user")],
        #     cancellation_token=CancellationToken(),
        # )
        # print(response.inner_messages)
        # print(response.chat_message)

        # Option 2: use Console to print all messages as they appear.
        await Console(
            agent.on_messages_stream(
                [TextMessage(content=content, source="user")],
                cancellation_token=CancellationToken(),
            ),
            output_stats=False,
        )

    async def assistant_on_image_message(content: str, content_url: str) -> None:
        pil_image = PIL.Image.open(BytesIO(requests.get(content_url).content))
        img = Image(pil_image)
        multi_modal_message = MultiModalMessage(content=[content, img], source="user")
        response = await agent.on_messages([multi_modal_message], CancellationToken())
        print(response.chat_message.content)

    await assistant_on_text_message("Find information on AutoGen")

    # await assistant_on_image_message("Can you describe the content of this image?", "https://picsum.photos/300/200")

asyncio.run(main())



