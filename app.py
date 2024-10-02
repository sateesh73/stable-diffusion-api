from fastapi import FastAPI
from model import getModel, genrate_image
from typing import List, Optional, Union
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io

app = FastAPI()
model_id = "emilianJR/epiCRealism"
device = "cuda" if torch.cuda.is_available() else "cpu"
access_token = "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"

class Item(BaseModel):
    prompt: str
    height: Optional[int] = 512
    width: Optional[int] = 768
    scale: Optional[float] = 7.5
    n_prompt: Optional[str] = "Worst quality, Normal quality, Low quality, Low res, Blurry, Jpeg artifacts, Grainy, Cropped, Out of frame, Out of focus, Bad anatomy, Bad proportions, Deformed, Disconnected limbs, Disfigured, Extra arms, Extra limbs, Extra hands, Fused fingers, Gross proportions, Long neck, Malformed limbs, Mutated, Mutated hands, Mutated limbs, Missing arms, Missing fingers, Poorly drawn hands, Poorly drawn face"
    num_images_per_prompt: Optional[int] = 1
    steps: Optional[int] = 50

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api")
async def getApi(item: Item):
    pipe = getModel(model_id, device, access_token) 
    print(item.prompt)
    image = genrate_image(item.prompt, item.height, item.width, item.n_prompt, item.scale, item.num_images_per_prompt, item.steps, pipe, device)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")