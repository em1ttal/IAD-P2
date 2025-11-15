import requests
import json, re

OPENROUTER_API_KEY = "TODO: Add your OpenRouter API key here"

response = requests.post(
  "https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
  },
  json={
    "model": "openrouter/polaris-alpha",
    "messages": [
      {"role": "user", "content": "What is the weather like in London?"},
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "weather",
        "strict": True,
        "schema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City or location name",
            },
            "temperature": {
              "type": "number",
              "description": "Temperature in Celsius",
            },
            "conditions": {
              "type": "string",
              "description": "Weather conditions description",
            },
          },
          "required": ["location", "temperature", "conditions"],
          "additionalProperties": False,
        },
      },
    },
  },
)
data = response.json()
weather_info = data["choices"][0]["message"]["content"]

try:
    decision = json.loads(weather_info)
    print(decision)
except json.JSONDecodeError as e:
    print("JSONDecodeError", e)