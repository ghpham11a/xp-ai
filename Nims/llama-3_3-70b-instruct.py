import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.getenv("LLAMA_3_3_70B_INSTRUCT_KEY")
)

completion = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")