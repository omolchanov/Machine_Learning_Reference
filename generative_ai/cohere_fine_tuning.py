# Guidelines:
# https://txt.cohere.com/programmatic-custom-model/


import cohere

co = cohere.Client('PjvfZ9sPIgqHTUhfMOTJVRCG6aWbpb48w7zpihdV')


def create_generative_model_shakespear():

    dataset = co.create_dataset(
        name='my_dataset',
        data=open('../assets/gpt/Shakespeare_data.csv', 'rb'),
        dataset_type='prompt-completion-finetune-input'
    ).await_validation()

    print(dataset.validation_status)

    co.create_custom_model(name='my_model', dataset=dataset, model_type='GENERATIVE')
    ft = co.get_custom_model_by_name('my_model')

    while True:

        prompt = input('>>> ')

        if prompt == 'exit':
            break

        if prompt == '':
            continue

        response = co.generate(
          model=ft.model_id,
          prompt=prompt
        )

        print('Answer: {}'.format(response.generations[0].text))


def create_generative_model_daily_dialogue():
    dataset = co.create_dataset(
        name='dialogue_dataset',
        data=open('../assets/gpt/dialogues_train.csv', 'rb'),
        dataset_type='prompt-completion-finetune-input'
    ).await_validation()

    model_name = 'dialogue_dataset_model'

    co.create_custom_model(name=model_name, dataset=dataset, model_type='GENERATIVE')
    ft = co.get_custom_model_by_name(model_name)

    while True:

        prompt = input('>>> ')

        if prompt == 'exit':
            break

        if prompt == '':
            continue

        response = co.generate(
            model=ft.model_id,
            prompt=prompt
        )

        print('Answer: {}'.format(response.generations[0].text))


if __name__ == '__main__':
    create_generative_model_shakespear()
    create_generative_model_daily_dialogue()
