# %%

import json
import os

TARGET_FOLDER = 'data'

os.makedirs(TARGET_FOLDER + 'RU', exist_ok = True)
os.makedirs(TARGET_FOLDER + 'EN', exist_ok = True)

count_ru = 0
count_en = 0

for name in os.listdir(TARGET_FOLDER):
    with open(TARGET_FOLDER + '/' + name, 'r', encoding = 'utf-8') as file:
        data = json.load(file)
    lang = [article['language'] for article in data]
    lang = 'RU' if lang.count('RU') > lang.count('EN') else 'EN'

    count = count_ru if lang == 'RU' else count_en

    with open(TARGET_FOLDER + lang + '/data%i.json' % count, 'w', encoding = 'utf-8') as file:
        json.dump(data, file, ensure_ascii = False)

    if lang == 'RU':
        count_ru += 1
    else:
        count_en += 1

# %%
