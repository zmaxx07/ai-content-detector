import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library is missing. Please run: pip install datasets")
    exit(1)

def main():
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), "../backend/.env")
    load_dotenv(env_path)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in backend/.env")
        print("Please add your key from https://aistudio.google.com/app/apikey")
        return
        
    genai.configure(api_key=api_key)
    
    print("Loading HC3 dataset (subset for tuning)...")
    try:
        # Load a small subset of the dataset to avoid massive upload times and limits
        dataset = load_dataset("Hello-SimpleAI/HC3", name="all", split="train[:50]")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Formatting data for Gemini tuning...")
    training_data = []
    
    # HC3 has 'human_answers' and 'chatgpt_answers' arrays.
    for item in dataset:
        # Add a human sample
        if item.get("human_answers") and len(item["human_answers"]) > 0:
            training_data.append({
                "text_input": item["human_answers"][0][:2000],  # Truncate to avoid exceeding input limits
                "output": "Human"
            })
        # Add an AI sample
        if item.get("chatgpt_answers") and len(item["chatgpt_answers"]) > 0:
            training_data.append({
                "text_input": item["chatgpt_answers"][0][:2000],
                "output": "AI"
            })

    # Gemini requires at least 20 examples for tuning
    print(f"Prepared {len(training_data)} training examples.")
    if len(training_data) < 20:
        print("Error: Need at least 20 examples for tuning.")
        return
    
    print("Starting Gemini tuning job...")
    try:
        # Using gemini-1.5-flash-001-tuning
        operation = genai.create_tuned_model(
            display_name="ai-content-detector-tuned",
            source_model="models/gemini-1.5-flash-001-tuning",
            epoch_count=3,
            batch_size=4,
            learning_rate=0.001,
            training_data=training_data
        )
    except Exception as e:
        print(f"Failed to start tuning job: {e}")
        return

    model_name = operation.name
    print(f"\nTuning job successfully created!")
    print(f"Model Resource Name: {model_name}")
    print("The model is now training remotely via Google AI Studio.")
    print("You can monitor the training progress at: https://aistudio.google.com/app/models\n")
    
    # Instead of blocking for hours, we gracefully exit.
    print("The script will now exit. The training job will continue on Google's servers.")

if __name__ == "__main__":
    main()
