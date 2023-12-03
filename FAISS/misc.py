# def get_query_embedding(query, model, tokenizer):
#     # Generate embeddings for the query
#     inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.pooler_output.numpy()

# get embedding for the query
# if model_name == "bert-base-uncased":
#     query_embedding = get_embedding2(query, model, tokenizer)
# elif model_name == "gpt2":
#     query_embedding = get_embedding2(query, model, tokenizer)
#     print(query_embedding.shape)
#     # query_embedding = np.mean(query_embedding, axis=1, keepdims=True)
#     # query_embedding = query_embedding.squeeze(1)

# if model_name == "bert-base-uncased":
#     embedding = get_embedding2(text_to_embed, model, tokenizer)
#     # print(embedding.shape)
# elif model_name == "gpt2":
#     embedding = get_embedding2(text_to_embed, model, tokenizer)
#     # embedding = np.mean(embedding, axis=1, keepdims=True)
#     # embedding = embedding.squeeze(1)

# def build_faiss_index(embedding_list, index_file_path, data):
#     embeddings = np.array(embedding_list).astype("float32")
#     embeddings = np.squeeze(embeddings, axis=1)
#
#     # set up an index
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)  # IndexFlatIP for inner product (for cosine similarity)
#
#     # Step 3: Pass the index to IndexIDMap
#     index = faiss.IndexIDMap(index)
#     # faiss.normalize_L2(embeddings)
#
#     # Step 4: Add vectors and their IDs
#     index.add_with_ids(embeddings, data.index_id.values)
#
#     # write index to a file
#     faiss.write_index(index, index_file_path)
#
#     print(f"FAISS index is built and saved to {index_file_path}")