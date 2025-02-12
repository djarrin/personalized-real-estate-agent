from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def create_vector_db(listings, db_path):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []
    
    for listing in listings:
        content = f"""
        Address: {listing["Address"]}
        Neighborhood: {listing["Neighborhood"]}
        Price: {listing["Price"]}
        Bedrooms: {listing["Bedrooms"]}
        Bathrooms: {listing["Bathrooms"]}
        House Size: {listing["House_Size"]}
        Description: {listing["Description"]}
        """
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=listing))
    
    vector_db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=db_path)
    vector_db.persist()
    print(f"Database saved to {db_path}")

def query_vector_db(vector_db, structured_preferences):
    """Retrieve relevant listings using vector database search."""
        
    metadata = structured_preferences.get("metadata", {})  # Extract structured search criteria
    
    query_text = ' '.join([
        f'Neighborhood: {metadata.get("Neighborhood", "")} Importance: High',
        f'Price: {metadata.get("Price", "")} Importance: High',
        f'Bedrooms: {metadata.get("Bedrooms", "")} Importance: Medium',
        f'Bathrooms: {metadata.get("Bathrooms", "")} Importance: Medium',
        f'Features: {", ".join(metadata.get("Features", []))} Importance: Medium'
    ])
        
    # Convert query text into an embedding
    query_embedding = OpenAIEmbeddings().embed_query(query_text)
    
    # Perform similarity search with top-k results
    results = vector_db.similarity_search_by_vector(query_embedding, k=5)
    
    # Convert search results into structured list format
    listings = [
        {
            "Address": result.metadata.get("Address", "Unknown"),
            "Neighborhood": result.metadata.get("Neighborhood", "Unknown"),
            "Price": result.metadata.get("Price", "Unknown"),
            "Bedrooms": result.metadata.get("Bedrooms", "Unknown"),
            "Bathrooms": result.metadata.get("Bathrooms", "Unknown"),
            "House_Size": result.metadata.get("House_Size", "Unknown"),
            "Description": result.metadata.get("Description", "Unknown"),
            "Neighborhood_Description": result.metadata.get("Neighborhood_Description", "Unknown"),
            "Features": result.metadata.get("Features", [])
        }
        for result in results
    ]
    
    return listings