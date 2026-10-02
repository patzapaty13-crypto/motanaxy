import os
import asyncio
import json
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import torch
from motanaxy.model import load_checkpoint

# Load the MOTANAXY model
print("Loading MOTANAXY model...")
ROOT = Path(__file__).resolve().parent
default_checkpoint = ROOT / "runs/mtn_full/best_model.pt"
comparison_path = ROOT / "runs/knowledge_v1/comparison.json"
if comparison_path.exists():
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if (comparison["heldout_loss_improved"] and
            comparison["after"]["syntactically_valid_functions"] >=
            comparison["before"]["syntactically_valid_functions"]):
        default_checkpoint = comparison_path.parent / "best_model.pt"
CHECKPOINT_PATH = Path(os.environ.get("MOTANAXY_CHECKPOINT", str(default_checkpoint)))
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {CHECKPOINT_PATH}")

model = load_checkpoint(str(CHECKPOINT_PATH))
torch.set_num_threads(2)
print(f"Model loaded: {model.num_parameters:,} parameters")

app = FastAPI(title="MOTANAXY Chat API")

# Ensure static dir exists
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/model")
async def model_info():
    manifest_path = CHECKPOINT_PATH.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    evaluation_path = CHECKPOINT_PATH.parent / "comparison.json"
    result = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else None
    return {"parameters": model.num_parameters, "run": CHECKPOINT_PATH.parent.name,
            "context_bytes": model.config["context"],
            "training_documents": manifest.get("documents", {}).get("train"),
            "evaluation": {"before_loss": result["before"]["loss"],
                           "after_loss": result["after"]["loss"],
                           "syntax_passes": result["after"]["syntactically_valid_functions"],
                           "prompts": result["after"]["prompts"]} if result else None,
            "status": "Experimental: generated code may be invalid"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4096)
    temperature: float = Field(default=0.6, ge=0.1, le=1.5, allow_inf_nan=False)
    max_tokens: int = Field(default=250, ge=1, le=500)
    top_k: int = Field(default=40, ge=0, le=256)

    @field_validator("message")
    @classmethod
    def nonempty_message(cls, value):
        if not value.strip():
            raise ValueError("message must contain text")
        return value


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_file = STATIC_DIR / "chat.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "Frontend not found. Please create static/chat.html"


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Takes user input, formats it as a prompt, and generates a response
    using the MOTANAXY byte-level model.
    """
    # Create a contextual prompt based on user intent
    user_msg = req.message.strip()
    
    # Match the curriculum's comment format; preserve actual code-completion prefixes.
    if user_msg.startswith(("def ", "class ", "import ", "from ")):
        prompt_text = user_msg + ("\n    " if user_msg.endswith(":") else "")
    else:
        prompt_text = "\n".join(line if line.startswith("#") else f"# {line}"
                                for line in user_msg.splitlines()) + "\n"

    prompt_bytes = list(prompt_text.encode("utf-8"))
    
    # Run generation asynchronously to not block the server
    loop = asyncio.get_event_loop()
    
    def run_generation():
        with torch.no_grad():
            result_bytes = model.generate(
                prompt_bytes,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            )
        return bytes(result_bytes).decode("utf-8", errors="replace")

    started = time.perf_counter()
    full_output = await loop.run_in_executor(None, run_generation)
    
    # The output includes the prompt. We can separate it for a cleaner chat experience.
    if full_output.startswith(prompt_text):
        generated_code = full_output[len(prompt_text):]
    else:
        generated_code = full_output

    # Create a conversational wrapper since the user wants a great interactive AI
    response_text = (
        "Here is what I generated based on my training data:\n\n"
        f"```python\n{prompt_text}{generated_code}\n```\n\n"
        f"*(Experimental model: {model.num_parameters:,} parameters. "
        "Generated code may be invalid; review before use.)*"
    )

    return {"response": response_text, "raw_code": generated_code, "code": full_output,
            "elapsed_seconds": round(time.perf_counter() - started, 3)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_server:app", host="127.0.0.1", port=8000, reload=True)
