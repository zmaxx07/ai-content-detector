import traceback

try:
    from transformers import pipeline
    print("Transformers imported.")
    p = pipeline('text-classification', model='Hello-SimpleAI/chatgpt-detector-roberta')
    print("Pipeline loaded.")
    print(p('Hello world'))
except Exception as e:
    print('Error loading pipeline:')
    traceback.print_exc()
