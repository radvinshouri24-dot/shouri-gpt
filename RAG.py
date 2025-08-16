from openai import OpenAI
from api2 import *


def currect_rag_response(In):
    return rag_response(In)

def rag_response(In):
    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key,
    )

    completion = client.chat.completions.create(
      extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
      },
      extra_body={},
      model="google/gemini-2.5-flash-lite",
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": In
            }
          ]
        }
      ]
    )
    response = completion.choices[0].message.content
    return response
