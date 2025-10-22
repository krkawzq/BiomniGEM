from litellm import completion

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = completion(
    model="gpt-4o-mini",
    messages=messages,
)

print(response["choices"][0]["message"]["content"])