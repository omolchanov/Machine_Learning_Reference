prompt_types = ['Zero-shot', 'Few-shot', 'Chain-of-thought', 'Self-consistency', 'ReAct']


def create_prompt(role, task, format, examples="", prompt_type="Zero-shot"):
    template = """
        Тип промта: {prompt_type}
        Ты: {role}.
        Задача: {task}
        Формат ответа: {format}
        Примеры: {examples}
    """

    return template.format(
        prompt_type=prompt_type,
        role=role,
        task=task,
        format=format,
        examples=examples
    )


for _, p in enumerate(prompt_types):
    prompt = create_prompt(
        role="QA-инженер",
        task="Напиши 5 тестов для функции is_even(n)",
        format="JSON",
        prompt_type=p,
        examples='[{"input":2,"expected":true}]',
    )

    print(prompt)
