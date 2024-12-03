from transformers import pipeline, GPT2TokenizerFast
import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Απαντήστε στην ερώτηση βασιζόμενοι μόνο στο ακόλουθο περιεχόμενο:

{context}

---

Απάντηση στην ερώτηση βάσει του παραπάνω περιεχομένου: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Το κείμενο της ερώτησης.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Truncate context
    encoded_context = tokenizer(context_text, truncation=True, max_length=900, return_tensors="pt")
    truncated_context = tokenizer.decode(encoded_context["input_ids"][0], skip_special_tokens=True)

    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=truncated_context, question=query_text)

    # Create Hugging Face pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
        max_new_tokens=200,
        pad_token_id=50256
    )
    model = HuggingFacePipeline(pipeline=hf_pipeline)

    # Generate response
    try:
        response_text = model(prompt)
    except Exception as e:
        print(f"Error during model generation: {str(e)}")
        return None

    # Extract sources
    sources = [doc.metadata.get("id", "Unknown source") for doc, _ in results]
    print(f"Απάντηση: {response_text}\nΠηγές: {sources}")
    return response_text

if __name__ == "__main__":
    main()
