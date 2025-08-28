import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DBNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(DBNet, self).__init__()
        
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            in_channels = 2048
        
        self.fpn = FeaturePyramidNetwork(in_channels)
        self.head = DBHead(256)
        
    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        output = self.head(fpn_features)
        return output

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for i in range(4):
            inner_block = nn.Conv2d(in_channels // (2**i), 256, 1)
            layer_block = nn.Conv2d(256, 256, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
    
    def forward(self, x):
        results = []
        last_inner = self.inner_blocks[0](x)
        results.append(self.layer_blocks[0](last_inner))
        
        for i in range(1, len(self.inner_blocks)):
            inner_lateral = self.inner_blocks[i](x)
            inner_top_down = nn.functional.interpolate(
                last_inner, scale_factor=2, mode='nearest'
            )
            last_inner = inner_lateral + inner_top_down
            results.append(self.layer_blocks[i](last_inner))
        
        return results[-1]

class DBHead(nn.Module):
    def __init__(self, in_channels):
        super(DBHead, self).__init__()
        self.probability_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels//4, in_channels//4, 2, stride=2),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels//4, 1, 2, stride=2),
            nn.Sigmoid()
        )
        
        self.threshold_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels//4, in_channels//4, 2, stride=2),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels//4, 1, 2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        prob_map = self.probability_head(x)
        thresh_map = self.threshold_head(x)
        return {'probability': prob_map, 'threshold': thresh_map}

class TextDetector:
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DBNet()
        
        if model_path:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        try:
            original_height, original_width = image.shape[:2]
            
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prob_map = output['probability'].cpu().numpy()[0, 0]
                
            detections = self._post_process(
                prob_map, 
                original_width, 
                original_height, 
                confidence_threshold
            )
            
            return detections
        
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _post_process(self, prob_map: np.ndarray, orig_width: int, orig_height: int, threshold: float) -> List[Dict[str, Any]]:
        binary_map = (prob_map > threshold).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue
                
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            
            x1, y1 = max(0, int(np.min(x_coords))), max(0, int(np.min(y_coords)))
            x2, y2 = min(640, int(np.max(x_coords))), min(640, int(np.max(y_coords)))
            
            x1 = int(x1 * orig_width / 640)
            y1 = int(y1 * orig_height / 640) 
            x2 = int(x2 * orig_width / 640)
            y2 = int(y2 * orig_height / 640)
            
            if x2 - x1 > 10 and y2 - y1 > 10:
                confidence = float(np.mean(prob_map[y1*640//orig_height:y2*640//orig_height, 
                                                    x1*640//orig_width:x2*640//orig_width]))
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'polygon': box.tolist()
                })
        
        return detections