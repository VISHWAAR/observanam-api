# main.py
import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from model import SiameseUNet

app = FastAPI(title="Observanam NISAR API")

# Allow your React frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change "*" to your React app's URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model globally so it doesn't reload on every request
device = torch.device("cpu") # Render free tier has no GPU
model = SiameseUNet().to(device)
model.load_state_dict(torch.load("observanam_nisar_siamese_best.pth", map_location=device))
model.eval()

@app.get("/")
def health():
    return {"status": "ok", "message": "Observanam API is running"}

@app.post("/predict")
async def predict(
    t1: UploadFile = File(...),
    t2: UploadFile = File(...)
):
    """
    Receives two 256x256 SAR images, returns a 256x256 mask.
    The frontend sends PNGs, but the model expects dB floats. 
    """
    # 1. Read images
    t1_img = Image.open(io.BytesIO(await t1.read())).convert("L")
    t2_img = Image.open(io.BytesIO(await t2.read())).convert("L")
    
    # 2. Convert to numpy & normalize (reverse the [-1,1] scale back to [0,1] for display)
    e = np.array(t1_img).astype(np.float32) / 255.0
    l = np.array(t2_img).astype(np.float32) / 255.0
    
    # Note: In a real setup, you'd reverse the dB normalization here. 
    # For the demo, we assume the frontend sent normalized images.
    # 3-channel tensor: [early, late, diff]
    x = np.stack([e, l, l - e], axis=0)[None]
    xt = torch.from_numpy(x).to(device)

    # 3. Inference
    with torch.no_grad():
        logits = model(xt)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]

    # 4. Convert mask to PNG
    mask_img = Image.fromarray((pred * 255).astype(np.uint8))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")
