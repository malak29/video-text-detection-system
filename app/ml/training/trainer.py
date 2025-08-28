import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class TextDetectionDataset(Dataset):
    def __init__(self, images: List[np.ndarray], targets: List[Dict], transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

class TextDetectionLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        prob_loss = self.bce_loss(outputs['probability'], targets['probability_map'])
        thresh_loss = self.bce_loss(outputs['threshold'], targets['threshold_map'])
        dice_loss = self.dice_loss(outputs['probability'], targets['probability_map'])
        
        total_loss = prob_loss + thresh_loss + dice_loss
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_prob_loss', prob_loss, on_epoch=True)
        self.log('train_thresh_loss', thresh_loss, on_epoch=True) 
        self.log('train_dice_loss', dice_loss, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        prob_loss = self.bce_loss(outputs['probability'], targets['probability_map'])
        thresh_loss = self.bce_loss(outputs['threshold'], targets['threshold_map'])
        dice_loss = self.dice_loss(outputs['probability'], targets['probability_map'])
        
        total_loss = prob_loss + thresh_loss + dice_loss
        
        self.validation_outputs.append({
            'loss': total_loss,
            'predictions': outputs['probability'],
            'targets': targets['probability_map']
        })
        
        return total_loss
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
            
        avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
        
        all_preds = torch.cat([x['predictions'].flatten() for x in self.validation_outputs])
        all_targets = torch.cat([x['targets'].flatten() for x in self.validation_outputs])
        
        binary_preds = (all_preds > 0.5).float()
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets.cpu().numpy(), 
            binary_preds.cpu().numpy(), 
            average='binary', 
            zero_division=0
        )
        
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)
        
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.trainer = None
        
    def setup_trainer(self, model: nn.Module) -> pl.Trainer:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.config['checkpoint_dir'],
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            verbose=True
        )
        
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=True,
            mode='min'
        )
        
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            log_every_n_steps=50,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        return trainer
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        lightning_model = TextDetectionLightningModule(
            model=model,
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        trainer = self.setup_trainer(model)
        
        try:
            trainer.fit(
                model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )
            
            best_model_path = trainer.checkpoint_callback.best_model_path
            
            return {
                'status': 'success',
                'best_model_path': best_model_path,
                'best_val_loss': float(trainer.checkpoint_callback.best_model_score),
                'epochs_trained': trainer.current_epoch
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        lightning_model = TextDetectionLightningModule(model=model)
        trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1)
        
        test_results = trainer.test(lightning_model, test_loader)
        
        return test_results[0] if test_results else {}