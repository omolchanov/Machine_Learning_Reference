import pprint

import os
from openai import OpenAI

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=OPENROUTER_API_KEY
)

MODEL = 'nousresearch/deephermes-3-mistral-24b-preview:free'
memory = []


def ai_chat():
    while True:
        user_input = input('👤 You: ')

        if user_input.lower() == 'exit':
            print('Bye!')
            break

        memory.append({'role': 'user', 'content': user_input})

        try:
            res = client.chat.completions.create(model=MODEL, messages=memory)
            reply = res.choices[0].message.content

            print('🤖 AI: ', reply)

            memory.append({'role': 'assistant', 'content': reply})
            print('\n')
            pprint.pp(memory)
            print('\n')

        except Exception as e:
            print(str(e))


if __name__ == '__main__':
    ai_chat()
