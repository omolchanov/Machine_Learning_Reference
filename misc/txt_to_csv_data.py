from csv import writer


def move_data_from_txt_to_csv():
    with open('../assets/gpt/dialogues_train.txt', 'r', encoding='utf8') as txt_f:
        lines = [line for line in txt_f]

        with open('../assets/gpt/dialogues_train.csv', 'a', encoding='utf8', newline='') as csv_f:
            w = writer(csv_f)

            for l in lines:
                replicas = l.split(' __eou__')[:-1]

                prompts = []
                completions = []

                for i, r in enumerate(replicas):
                    if not i % 2:
                        prompts.append(replicas[i].strip())
                    else:
                        completions.append(replicas[i].strip())

                for i, p in enumerate(prompts):

                    try:
                        row = [p, completions[i]]
                        print(row)
                        w.writerow(row)
                    except:
                        continue

            csv_f.close()

        txt_f.close()
