# %%

import requests
import json
import os

from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector

ELEMENTS_PER_QUERY = 5
TARGET_FOLDER = 'data'

QDRANT_HOST = 'scisearch.ru'
QDRANT_PORT = ???

QDRANT_LSEARCH_NAME = ???
QDRANT_COLLECTION = ???

with open('queries.txt', 'r', encoding = 'utf-8') as file:
    queries = list(i.replace('\n', '') for i in file.readlines() if i)

api_url = ???

articles = []

for query in queries:
    payload = {'elemsPerPage' : ELEMENTS_PER_QUERY, 'numPage' : 1, 'query' : query}
    response = requests.post(api_url, json = payload)
    response.raise_for_status()
    data = response.json()

    if not data['message'] == 'Поиск по статьям выполнен успешно':
        raise Exception('Поиск по статьям не удался (%s)' % query)

    articles += data['articles']

    if len(data['articles']) < ELEMENTS_PER_QUERY:
        raise Exception('Для запроса "%s" найти статьи не удалось' % query)
    print('"%s" OK!' % query)

# %%

def extract(id : str):
    response = requests.get(api_url + id)
    response.raise_for_status()
    data = response.json()

    if not data['message'] == 'Статья успешно получена':
        raise Exception('Получить статью не удалось')
    return data['article']

nums = [7, 14, 15, 25, 26, 28, 29, 30, 32, 33, 38, 45, 46, 47, 49, 67, 80, 81, 82, 84, 92, 95, 96]
count = 0
os.makedirs(TARGET_FOLDER, exist_ok = True)

client = QdrantClient(host = QDRANT_HOST, port = QDRANT_PORT)
api_url = 'https://scisearch.ru/api/articles/'

for article in articles:
    if count == len(nums):
        break
    try:
        rec_s = client.retrieve(
            collection_name = QDRANT_COLLECTION,
            ids = [article['articleId']],
            with_vectors = True
        )[0].vector[QDRANT_LSEARCH_NAME]

        query_vector = NamedVector(name = QDRANT_LSEARCH_NAME, vector = rec_s)

        results = client.search(
            collection_name = QDRANT_COLLECTION,
            query_vector = query_vector,
            limit = 2000,
            score_threshold = 0.1
        )
        res = [result.id for result in results]
        result = [article]

        with ThreadPoolExecutor(max_workers = 10) as executor:
            result += list(executor.map(extract, res))

        path = TARGET_FOLDER + '/data%i.json' % nums[count]
        count += 1

        with open(path, 'w', encoding = 'utf-8') as file:
            json.dump(result, file, ensure_ascii = False)

    except Exception as e:
        print(f'{count} got an error -', e)
        count += 1

# %%

nums = list(range(102))
for file in os.listdir(TARGET_FOLDER):
    num = int(file[4:-5])
    nums.remove(num)
print(nums)

# %%
