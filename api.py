from openai import OpenAI

base_url = "your_base_url"
key = "your_key"

client = OpenAI(
    api_key=key, 
    base_url=base_url
)

def call_openai_api(prompt, model="gpt-4o-mini", max_tokens=1000):
    """
    Call the OpenAI API with the given prompt.
    
    Args:
        prompt (str): The prompt to send to the API
        model (str): The model to use for generation
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness (0-1)
        
    Returns:
        str: The generated response text
    """
    try:
        response = None
        if model != "o1-mini":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
        # Extract and return the response text
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    test_prompt = "What is the capital of France?"
    model_list = ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1", "o3-mini", "o3",
                  "google/gemini-2.0-flash-001", "google/gemini-2.5-flash-preview", "google/gemini-2.5-pro-preview",
                  "anthropic/claude-3.5-sonnet", "anthropic/claude-3.7-sonnet", "anthropic/claude-3.7-sonnet:thinking"]
    for model in model_list:
        response = call_openai_api(test_prompt, model)
        if response is not None:
            print(f"Response: {response}")
            print(f"Model: {model} is working")
