import json
import math
import os
import sys
import time
from hashlib import md5
from functools import lru_cache

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../')))
from modules import text2embedding
from modules.vectorstores import FAISS
from modules.vectorstores.docstore.document import Document
from config.text_embedding_config import MODEL_LIST

CACHED_VS_NUM = 100
search_words_path = "../datasets/关键词"
dir_path = "../datasets/texts"
vs_path = "../../vector_store"
model_base_path = "/aigc/data/wangjing/models/embeddings/seq"


def get_embeddings(model_name, model_version):
    args = {
        "task_type": "sequence-text2embedding",
        "infer_arch": "Pytorch",
        "device": "CPU",
        "model_name": model_name,
        "model_version": model_version,
    }
    text2embedding.init_model(args)

    return text2embedding


def encrypt_md5(string):
    new_md5 = md5()
    new_md5.update(string.encode(encoding='utf-8'))

    return new_md5.hexdigest()


# will keep CACHED_VS_NUM of vector store caches
@lru_cache(CACHED_VS_NUM)
def load_vector_store(index_path, embeddings, index_name):
    return FAISS.load_local(index_path, embeddings, index_name)


def search_from_vector_store(self, query, top_k=10):
    vector_store = load_vector_store(self.vs_root_path, self.embeddings)
    related_docs_with_score = vector_store.similarity_search_with_score(query, k=top_k)

    return related_docs_with_score


def get_documents():
    all_documents = {}
    dirs = os.listdir(dir_path)
    for file_name in sorted(dirs, key=lambda x: int(x.split("_")[0])):
        file_path = os.path.join(dir_path, file_name)
        label_name = f"{file_name[:-6]}"

        docs = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                line_content = json.loads(line)
                if len(line_content["text"].strip()) > 0:
                    doc = Document(
                        page_content=line_content["text"],
                        metadata={
                            "filename": label_name,
                        }
                    )
                    docs.append(doc)

        file.close()

        all_documents[label_name] = docs

    return all_documents


def create_embeddings(model_name, model_version, batch_size=16):
    embeddings = get_embeddings(model_name, model_version)
    index_name = f"index_{model_version}"

    all_documents = get_documents()
    for _, docs in all_documents.items():
        start_time = time.time()
        for i in range(math.floor(len(docs) / batch_size)):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size]
            if f"{index_name}.faiss" in os.listdir(vs_path):
                vector_store = load_vector_store(vs_path, embeddings, index_name)
                vector_store.add_documents(batch_docs)
            else:
                vector_store = FAISS.from_documents(batch_docs, embeddings)

            vector_store.save_local(vs_path, index_name)
        cost_time = round(time.time() - start_time, 3)
        print(f"index_name = {index_name}, docs_count = {len(docs)}, cost_time = {cost_time}",  flush=True)


def get_model_name_and_version():
    model_list = []
    for key, model_info in MODEL_LIST["Pytorch"].items():
        for model_version in model_info["model_list"]:
            model_list.append({
                "model_name": key,
                "model_version": model_version,
            })
    return model_list


def create_index():
    model_list = get_model_name_and_version()
    for model_info in model_list:
        create_embeddings(model_info["model_name"], model_info["model_version"])


def get_search_words():
    search_words_dict = {}
    search_words_files = os.listdir(search_words_path)
    for search_words_file in search_words_files:
        search_words_file_path = os.path.join(search_words_path, search_words_file)
        with open(search_words_file_path) as file:
            lines = file.readlines()
            search_words_dict[search_words_file[:-4]] = [line.replace("\n", "") for line in lines]

    return search_words_dict


def get_dir_size(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size


def search():
    result_file_path = "./gte_embeddings_result.txt"
    with open(result_file_path, 'w+', encoding='utf-8-sig') as file:

        top_k_arr = [1, 3, 5, 10]
        search_words = get_search_words()

        model_list = get_model_name_and_version()

        for model_info in model_list:
            embeddings = get_embeddings(model_info['model_name'], model_info['model_version'])
            index_name = f"index_{model_info['model_version']}"
            model_path = os.path.join(model_base_path, model_info['model_version'])
            model_size = format(get_dir_size(model_path) / 1000 / 1000 / 2, ".2f")

            for top_k in top_k_arr:
                correct_count, error_count, total_count = 0, 0, 0
                for key, search_word in search_words.items():
                    vector_store = load_vector_store(vs_path, embeddings, index_name)
                    for query in search_word:
                        if len(query) > 0:
                            result_lists = vector_store.similarity_search_with_score(query, k=top_k)
                            for doc in result_lists:
                                if doc[0].metadata['filename'].strip() == key.strip():
                                    correct_count += 1
                                else:
                                    error_count += 1
                                # print(f"query_txt = {key}, query = {query}, result_label = {doc[0].metadata['filename']}, score = {doc[1]}")
                                total_count += 1

                correct_rate = format(correct_count / total_count * 100, ".2f")
                metric = f"model_name = {model_info['model_version']}, top_k = {top_k}, total_count = {total_count}, correct_count = {correct_count}, error_count = {error_count}, correct_rate = {correct_rate}%, model_size = {model_size}MB"
                print(metric, flush=True)
                file.write(metric+"\n")
                file.flush()
            print("\n", flush=True)
            file.write("\n")
            # sys.exit(0)


if __name__ == "__main__":
    # create index
    create_index()

    # search
    # search()
