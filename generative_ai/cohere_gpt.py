import cohere

co = cohere.Client('PjvfZ9sPIgqHTUhfMOTJVRCG6aWbpb48w7zpihdV')


prompt = input()

response = co.generate(
    model='command',
    prompt=prompt,
    max_tokens=200,
    temperature=1)

output = response.generations[0].text

print(output)
