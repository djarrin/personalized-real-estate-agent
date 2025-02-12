import os
import json
import time
import argparse
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

def load_env():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")

def generate_home_listings(prompt, index):
    system_prompt = """
    You are a real estate listing assistant specializing in crafting compelling and informative home descriptions. Your task is to generate real estate listings in **valid JSON format** based on user-provided details.
    
    Each listing should strictly follow this JSON format:

    {
        "Address": "[Address"]
        "Neighborhood": "[Neighborhood Name]",
        "Price": "$[Price]",
        "Bedrooms": [Number of Bedrooms],
        "Bathrooms": [Number of Bathrooms],
        "House_Size": "[Size in sqft]",
        "Description": "[Compelling property description]",
        "Neighborhood_Description": "[Engaging neighborhood description]"
    }

    **Rules:**
    - Return output in **pure JSON format** without any extra text.
    - Ensure values are properly formatted.
    - Use double quotes `" "` for JSON keys and values.

    Address: Generate a fake address with the Neighborhood as the city. Ex. 225 W 38th Street, {Neighborhood}
    
    Description:
    Provide a captivating yet concise property description, emphasizing key selling points such as architectural style, unique features, energy efficiency, or location benefits. Use persuasive language to make the home appealing to potential buyers. Highlight amenities, natural lighting, interior design elements, and lifestyle advantages.
    
    Neighborhood Description:
    Offer a brief, engaging description of the neighborhood. Highlight aspects such as community atmosphere, parks, restaurants, transportation, and local attractions. Focus on features that would be appealing to potential buyers considering the area.
    
    Maintain a professional and inviting tone, ensuring the listing feels aspirational yet grounded in reality. Avoid exaggeration while emphasizing the home’s best features.
    """
    print(f"Generating listing {index + 1}...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=256
    )
    return json.loads(response.choices[0].message.content.strip())

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

def collect_user_preferences():
    questions = [
        "What neighborhood are you looking in?",
        "What is your budget range?",
        "How many bedrooms do you need?",
        "How many bathrooms do you need?",
        "What size house are you looking for?",
        "Are there any specific features you want?"
    ]
    answers = {}
    for question in questions:
        answers[question] = input(question + " ")
    return answers

def collect_structured_preferences(answers):
    """Extract structured preferences from user responses and return in vector database query format."""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_bedrooms",
                "description": "Extract the exact number of bedrooms as an integer.",
                "parameters": {"type": "object", "properties": {"Bedrooms": {"type": "integer"}}, "required": ["Bedrooms"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_bathrooms",
                "description": "Extract the exact number of bathrooms as an integer.",
                "parameters": {"type": "object", "properties": {"Bathrooms": {"type": "integer"}}, "required": ["Bathrooms"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_neighborhood",
                "description": "Extract the neighborhood or city name from user input.",
                "parameters": {"type": "object", "properties": {"Neighborhood": {"type": "string"}}, "required": ["Neighborhood"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_price",
                "description": "Extract the maximum budget as an integer.",
                "parameters": {"type": "object", "properties": {"Price": {"type": "integer"}}, "required": ["Price"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_house_size",
                "description": "Extracts the house size in square feet.",
                "parameters": {"type": "object", "properties": {"House_Size": {"type": "string"}}, "required": ["House_Size"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_features",
                "description": "Extracts specific features desired in the home.",
                "parameters": {"type": "object", "properties": {"Features": {"type": "array", "items": {"type": "string"}}}, "required": ["Features"]}
            }
        }
    ]
    
    messages = [{"role": "system", "content": "Extract structured real estate search criteria."}]
    
    for question, answer in answers.items():
        messages.append({"role": "user", "content": f"Q: {question}\nA: {answer}"})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,  # Pass ALL tools at once
        tool_choice="auto"
    )
    
    # Extract structured data from tool calls
    structured_criteria = {
        "Neighborhood": "",
        "Price": None,
        "Bedrooms": None,
        "Bathrooms": None,
        "House_Size": "",
        "Features": []
    }
    
    tool_calls = response["choices"][0].get("message", {}).get("tool_calls", [])
    
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_response = json.loads(tool_call["function"]["arguments"])

        if tool_name == "extract_bedrooms":
            structured_criteria["Bedrooms"] = tool_response.get("Bedrooms")
        elif tool_name == "extract_bathrooms":
            structured_criteria["Bathrooms"] = tool_response.get("Bathrooms")
        elif tool_name == "extract_neighborhood":
            structured_criteria["Neighborhood"] = tool_response.get("Neighborhood")
        elif tool_name == "extract_price":
            structured_criteria["Price"] = tool_response.get("Price")
        elif tool_name == "extract_house_size":
            structured_criteria["House_Size"] = tool_response.get("House_Size")
        elif tool_name == "extract_features":
            structured_criteria["Features"] = tool_response.get("Features", [])

    return {"page_content": json.dumps(structured_criteria), "metadata": structured_criteria}

property_description_prompt = PromptTemplate(
    input_variables=["neighborhood", "price", "bedrooms", "bathrooms", "features",
                     "listing_address", "listing_price", "listing_bedrooms", 
                     "listing_bathrooms", "listing_features", "listing_description"],
    template="""
    You are a real estate assistant providing personalized property descriptions. 

    The buyer is looking for a property in {neighborhood} with a budget of around {price}. 
    They prefer {bedrooms} bedrooms and {bathrooms} bathrooms, and they are particularly 
    interested in features like {features}.

    Below is a property listing:

    - Address: {listing_address}
    - Price: {listing_price}
    - Bedrooms: {listing_bedrooms}
    - Bathrooms: {listing_bathrooms}
    - Features: {listing_features}
    - Description: {listing_description}

    ### Task:
    1. Rewrite the property description so it highlights aspects that match the buyer’s preferences.
    2. Keep all factual details intact.
    3. Maintain a professional and engaging tone.

    ### Output:
    """
)

def generate_personalized_description(llm, user_prefs, listing):
    prompt = property_description_prompt.format(
        neighborhood=user_prefs.get("Neighborhood", "N/A"),
        price=user_prefs.get("Price", "N/A"),
        bedrooms=user_prefs.get("Bedrooms", "N/A"),
        bathrooms=user_prefs.get("Bathrooms", "N/A"),
        features=", ".join(user_prefs.get("Features", [])),
        listing_address=listing.get("Address", "N/A"),
        listing_price=listing.get("Price", "N/A"),
        listing_bedrooms=listing.get("Bedrooms", "N/A"),
        listing_bathrooms=listing.get("Bathrooms", "N/A"),
        listing_features=", ".join(listing.get("Features", [])),
        listing_description=listing.get("Description", "N/A")
    )
    response = llm.invoke(prompt) 
    return response

def main():
    load_env()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    default_db_path = "vector_db"
    db_path = input(f"Enter path to vector database (default: {default_db_path}): ") or default_db_path
    
    generate_db = input("Would you like to generate a new database? (y/n, default: n): ").strip().lower() == "y"
    
    if generate_db:
        prompts = [
            "Generate a listing for a 4-bedroom, 2 bathroom house, with 4,000 sqft in Baltimore, priced at $400,000. The home has a parking pad and is near the park.",
            "Generate a listing for a 2-bedroom, 1 bathroom house, with 1,200 sqft in Washington DC, priced at $600,000. The home has a back patio and has street parking.",
            "Generate a listing for a 6-bedroom, 3 bathroom house, with 8,000 sqft in Dallas, priced at $500,000. The home has a large backyard and has three garages.",
            "Generate a listing for a 1-bedroom, 1 bathroom apartment, with 400 sqft in New York, priced at $800,000. The apartment has bathroom kitchen and features a window.",
            "Generate a listing for a 2-bedroom, 1 bathroom house, with 1,100 sqft in Baltimore, priced at $205,000. The home has a view of the harbor and is walking distance to the stadiums.",
            "Generate a listing for a 2-bedroom, 1 bathroom house, with 1,100 sqft in San Francisco, priced at $812,000. The home has is close to public transit and is near a park.",
            "Generate a listing for a 3-bedroom, 1 bathroom apartment, with 1,100 sqft in Chicago, priced at $700,000. The home is an apartment and has a view of the river.",
            "Generate a listing for a 3-bedroom, 2 bathroom house, with 2,100 sqft in New York, priced at $500,000. The home is in upstate New York and is near a wings restaurant.",
            "Generate a listing for a 3-bedroom, 2 bathroom house, with 1,100 sqft in Baltimore, priced at $190,000. The home near Canton square and has many dining options near it.",
            "Generate a listing for a 2-bedroom, 2 bathroom house, with 1,700 sqft in Dallas, priced at $195,000. The home is near the PGA headquarters and offers many opportunities to golf.",
            "Generate a listing for a 4-bedroom, 2 bathroom house, with 1,900 sqft in Eldersburg, priced at $450,000. The home is in a great school district.",
            "Generate a listing for a 4-bedroom, 2 bathroom house, with 1,700 sqft in Phoenix, priced at $400,000. The home is in the desert, has great views and is in a valley.",
            "Generate a listing for a 6-bedroom, 2 bathroom house, with 6,700 sqft in Dallas, priced at $600,000. The home is in the suberbs of Dallas and is only a 25 min drive from the city.",
            "Generate a listing for a 1-bedroom, 1 bathroom apartment, with 900 sqft in New York, priced at $800,000. The home is an apartment and is a 15 min walk from the park.",
            "Generate a listing for a 1-bedroom, 1 bathroom apartment, with 1,200 sqft in Miami, priced at $800,000. The home is an apartment and is a 15 min walk from the beach.",
        ]
        listings = [generate_home_listings(p, i) for i, p in enumerate(prompts)]
        create_vector_db(listings, db_path)
    
    vector_db = Chroma(persist_directory=db_path, embedding_function=OpenAIEmbeddings())
    user_answers = collect_user_preferences()
    structured_preferences = collect_structured_preferences(user_answers)
    results = query_vector_db(vector_db, structured_preferences)
    
    if results:
        personalized_desc = generate_personalized_description(llm, structured_preferences, results[0])
        print("\nPersonalized Property Description:\n", personalized_desc.content)

if __name__ == "__main__":
    main()
