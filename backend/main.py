from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassifiction, AutoImageProcessor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

from PIL import Image
import io

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL LOADING (Done once only on startup) ---
VIT_PATH = ""
RESNET_PATH = "" 
DETECTOR_PATH = "" 

vit = AutoModelForImageClassifiction.from_pretrained(VIT_PATH).to(device).eval()
vit_proc = AutoImageProcessor.from_pretrained(VIT_PATH)

resnet = AutoModelForImageClassifiction.from_pretrained(RESNET_PATH).to(device).eval()
resnet_proc = AutoImageProcessor.from_pretrained(RESNET_PATH)

# Detector Setup

detector = fasterrcnn_resnet50_fpn(weights = None)
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
detector.load_state_dict(torch.load(DETECTOR_PATH, map_location = device))
detector.to(device).eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...), sensitivity: float = 0.35):
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception: 
        raise HTTPException(status_code = 400, detail = "Invalid Imafe File")
    
    #1. Ensemble Inference
    with torch.no_grad():
        v_in = vit_proc(iamges = image, return_tensor = "pt").to(device)
        r_in = resnet_proc(images = image, return_tensor = "pt").to(device)
        
        v_logits = vit(**v_in).logits
        r_logits = resnet(**r_in).logits
        
        avg_probs = (F.softmax(v_logits, dim = -1) + F.softmax(r_logits, dim = -1))/2.0
        avg_probs = avg_probs.cpu().numpy()[0]
        
    #2. Logic Check
    opacity_risk = float(avg_probs[0])
    
    response = {
        "opacity_risk": opacity_risk,
        "all_probs" : avg_probs.tolist(),
        "detection_triggered" : False,
        "boxes": []
    }
    
    if opacity_risk > sensitivity:
        response["detection_triggered"] = True
        
        det_tensor = transforms.ToTensor()(image).to(device)
        with torch.no_grad():
            otuput = detector([det_tensor])[0]
            #filter boxes with score > 0.5
            keep_idx = output['scores'] > 0.5
            boxes = output['boxes'][keep_idx].cpu().numpy().tolist()
            response["boxes"] = boxes
            
    return boxes 
    
    