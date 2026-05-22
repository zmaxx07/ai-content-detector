import asyncio
from services.hf_inference import query_hf_model

MODELS = [
    {"id": "roberta-base-openai-detector", "weight": 0.40},
    {"id": "Hello-SimpleAI/chatgpt-qa-detector-roberta", "weight": 0.35},
    {"id": "unitary/toxic-bert", "weight": 0.25} # Placeholder for 3rd text model
]

async def run_ensemble(text: str) -> dict:
    tasks = [
        query_hf_model(m["id"], {"inputs": text}) for m in MODELS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_score = 0.0
    valid_weight = 0.0
    breakdown = {}
    
    for model, res in zip(MODELS, results):
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            # Parse typical HF classification output
            scores = {item['label']: item['score'] for item in res[0]}
            # Assuming 'Fake' or 'LABEL_1' implies AI
            ai_prob = scores.get('Fake', scores.get('LABEL_1', 0.5))
            total_score += ai_prob * model["weight"]
            valid_weight += model["weight"]
            breakdown[model["id"]] = ai_prob
            
    final_score = (total_score / valid_weight) if valid_weight > 0 else 0.5
    return {"final_score": final_score, "breakdown": breakdown}
