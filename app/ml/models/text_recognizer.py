import torch
import torch.nn as nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 2):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        
        self.rnn = nn.LSTM(512, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x):
        conv_feat = self.cnn(x)
        b, c, h, w = conv_feat.size()
        conv_feat = conv_feat.view(b, c * h, w).permute(0, 2, 1)
        
        rnn_feat, _ = self.rnn(conv_feat)
        output = self.classifier(rnn_feat)
        
        return output

class TransformerRecognizer:
    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=50)
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                'text': generated_text,
                'confidence': 0.95  
            }
            
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            return {'text': '', 'confidence': 0.0}

class TextRecognizer:
    def __init__(self, model_path: str = None, use_transformer: bool = True):
        self.use_transformer = use_transformer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if use_transformer:
            self.model = TransformerRecognizer()
        else:
            self.vocab = self._build_vocab()
            self.model = CRNN(len(self.vocab))
            if model_path:
                self.load_model(model_path)
            self.model.to(self.device)
            self.model.eval()
    
    def _build_vocab(self) -> Dict[str, int]:
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        vocab = {char: i+1 for i, char in enumerate(chars)}
        vocab['<blank>'] = 0
        vocab['<unk>'] = len(vocab)
        return vocab
    
    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"CRNN model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load CRNN model: {e}")
            raise
    
    def recognize_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        if self.use_transformer:
            return [self.model.recognize(img) for img in images]
        else:
            return self._recognize_crnn_batch(images)
    
    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        if self.use_transformer:
            return self.model.recognize(image)
        else:
            return self._recognize_crnn_batch([image])[0]
    
    def _recognize_crnn_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        try:
            batch_tensors = []
            for img in images:
                img_resized = cv2.resize(img, (128, 32))
                img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                batch_tensors.append(img_tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                predictions = torch.softmax(outputs, dim=2)
            
            results = []
            for pred in predictions:
                text, confidence = self._decode_prediction(pred)
                results.append({
                    'text': text,
                    'confidence': confidence
                })
            
            return results
            
        except Exception as e:
            logger.error(f"CRNN batch recognition failed: {e}")
            return [{'text': '', 'confidence': 0.0}] * len(images)
    
    def _decode_prediction(self, prediction: torch.Tensor) -> Tuple[str, float]:
        pred_indices = torch.argmax(prediction, dim=1)
        
        text = ""
        confidences = []
        prev_char = None
        
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        for idx in pred_indices:
            char_idx = idx.item()
            if char_idx == 0:  # blank
                continue
            if char_idx == prev_char:  # CTC duplicate removal
                continue
                
            char = reverse_vocab.get(char_idx, '<unk>')
            if char != '<unk>':
                text += char
                confidences.append(torch.max(prediction[len(text)-1]).item())
            
            prev_char = char_idx
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return text, float(avg_confidence)