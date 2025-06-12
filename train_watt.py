import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
import numpy as np
import time

# Import the complete model from our perfected watt_model.py
from watt_model import WorldAwareTrajectoryTransformer, NCPS_AVAILABLE

# --- UTILITY FUNCTIONS ---
def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Converts boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def move_to_device(data, device):
    """Recursively moves all tensors in a nested data structure to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    return data

# --- DATASET DEFINITION (WITH PAIRING FOR DYNAMICS) ---
class SimulatedPairedNuPlanDataset(Dataset):
    """A simulated Dataset that returns paired consecutive samples for dynamics model training."""
    def __init__(self, num_samples=1000, **kwargs):
        self.num_samples = num_samples
        self.kwargs = kwargs
        print(f"Generating {num_samples + 1} simulated data points in memory...")
        self.data_cache = [self._generate_sample(i) for i in range(num_samples + 1)]
        print("Dataset generation complete.")

    def __len__(self):
        return self.num_samples

    def _generate_sample(self, idx):
        """Generates a single, rich data sample."""
        h, w, t, c = self.kwargs['img_height'], self.kwargs['img_width'], self.kwargs['num_timesteps'], self.kwargs['num_cameras']
        num_points, bh, bw, nc = self.kwargs['num_trajectory_points'], self.kwargs['bev_height'], self.kwargs['bev_width'], self.kwargs['num_bev_classes']
        ma, nfs = self.kwargs['max_agents'], self.kwargs['num_future_steps']
        images = torch.rand(t, c, h, w, 3)
        
        # --- FIX: Separated definition of start_point and end_point to resolve UnboundLocalError ---
        start_point = np.random.rand(2) * 5
        end_point = start_point + np.random.rand(2) * 20
        trajectory = torch.from_numpy(np.linspace(start_point, end_point, num_points)).float()
        
        bev_map = torch.randint(0, nc, (bh, bw)).long()
        num_agents = np.random.randint(5, 20)
        gt_boxes, gt_labels = torch.rand(num_agents, 4), torch.randint(0, 3, (num_agents,)).long()
        padded_boxes, padded_labels = torch.zeros(ma, 4), torch.full((ma,), 3)
        padded_boxes[:num_agents], padded_labels[:num_agents] = gt_boxes, gt_labels
        future_trajectories = torch.zeros(ma, nfs, 2)
        for i in range(num_agents):
            agent_start = padded_boxes[i, :2]
            agent_end = agent_start + torch.rand(2) * 5
            future_trajectories[i] = torch.from_numpy(np.linspace(agent_start.numpy(), agent_end.numpy(), nfs)).float()
        return {"images": images, "ground_truth_trajectory": trajectory, "ground_truth_bev_map": bev_map,
                "ground_truth_agents": {"boxes": padded_boxes, "labels": padded_labels, "future_trajectories": future_trajectories}}

    def __getitem__(self, idx):
        sample_t, sample_t_plus_1 = self.data_cache[idx], self.data_cache[idx + 1]
        action_t = sample_t["ground_truth_trajectory"]
        return {"t": sample_t, "t+1": sample_t_plus_1, "action_t": action_t}

# --- LOSS FUNCTION DEFINITIONS ---
class AgentDetectionLoss(nn.Module):
    def __init__(self, num_classes=3, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.num_classes, self.cost_weights = num_classes, {'class': cost_class, 'bbox': cost_bbox, 'giou': cost_giou}
        empty_weight = torch.ones(num_classes + 1); empty_weight[-1] = 0.1
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    @torch.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, pred_outputs, gt_targets):
        pred_logits, pred_boxes = pred_outputs['pred_logits'], pred_outputs['pred_boxes']
        indices = []
        for i in range(pred_logits.shape[0]):
            log_probs_i, boxes_i_cxcywh = F.log_softmax(pred_logits[i], dim=-1), pred_boxes[i][..., :4]
            gt_labels_i_all, gt_boxes_i_cxcywh_all = gt_targets['labels'][i], gt_targets['boxes'][i]
            valid_gt_mask = gt_labels_i_all < self.num_classes
            gt_labels_i, gt_boxes_i_cxcywh = gt_labels_i_all[valid_gt_mask], gt_boxes_i_cxcywh_all[valid_gt_mask]
            if gt_labels_i.numel() == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device), torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue
            cost_class = -log_probs_i[:, gt_labels_i]
            cost_bbox = torch.cdist(boxes_i_cxcywh, gt_boxes_i_cxcywh, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes_i_cxcywh), box_cxcywh_to_xyxy(gt_boxes_i_cxcywh))
            cost_matrix = (self.cost_weights['bbox'] * cost_bbox + self.cost_weights['class'] * cost_class + self.cost_weights['giou'] * cost_giou)
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)))
        
        src_idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        if src_idx[0].numel() > 0:
            b_idx, t_idx = self._get_tgt_permutation_idx(indices)
            matched_gt_labels = gt_targets['labels'][b_idx, t_idx]
            target_classes[src_idx] = matched_gt_labels
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        if src_idx[0].numel() > 0:
            b_idx, t_idx = self._get_tgt_permutation_idx(indices)
            matched_pred_boxes = pred_boxes[src_idx]
            matched_gt_boxes = gt_targets['boxes'][b_idx, t_idx]
            loss_bbox = F.l1_loss(matched_pred_boxes[..., :4], matched_gt_boxes)
            loss_giou = (1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(matched_pred_boxes[..., :4]), box_cxcywh_to_xyxy(matched_gt_boxes)))).mean()
        else:
            loss_bbox = loss_giou = torch.tensor(0.0, device=pred_logits.device)
        return {'loss_ce': loss_ce, 'loss_bbox': loss_bbox, 'loss_giou': loss_giou}, indices

class WATT_Loss(nn.Module):
    def __init__(self, loss_weights: dict):
        super().__init__()
        self.weights = loss_weights
        self.planning_loss_fn, self.bev_loss_fn = nn.HuberLoss(), nn.CrossEntropyLoss()
        self.agent_det_loss_fn = AgentDetectionLoss()
        self.agent_pred_loss_fn, self.dynamics_loss_fn = nn.HuberLoss(), nn.MSELoss()

    def forward(self, model, batch):
        gt_t, outputs_t = batch['t'], model(batch['t']['images'])
        loss_planning, loss_bev = self.planning_loss_fn(outputs_t['trajectory'], gt_t['ground_truth_trajectory']), self.bev_loss_fn(outputs_t['bev_map_logits'], gt_t['ground_truth_bev_map'])
        det_losses, matched_indices = self.agent_det_loss_fn(outputs_t['agent_detection'], gt_t['ground_truth_agents'])
        loss_agents_det = (self.weights['agent_ce'] * det_losses['loss_ce'] + self.weights['agent_bbox'] * det_losses['loss_bbox'] + self.weights['agent_giou'] * det_losses['loss_giou'])
        loss_agents_pred = torch.tensor(0.0, device=loss_planning.device)
        if NCPS_AVAILABLE and self.weights.get('agent_pred', 0) > 0:
            src_idx = self.agent_det_loss_fn._get_src_permutation_idx(matched_indices)
            if src_idx[0].numel() > 0:
                agent_features = outputs_t['agent_detection']['decoder_features'][src_idx]
                predicted_futures = model.agent_prediction_head(agent_features)
                b_idx, t_idx = self.agent_det_loss_fn._get_tgt_permutation_idx(matched_indices)
                matched_gt_futures = gt_t['ground_truth_agents']['future_trajectories'][b_idx, t_idx]
                loss_agents_pred = self.agent_pred_loss_fn(predicted_futures, matched_gt_futures)
        with torch.no_grad(): latent_t_plus_1_real = model.encode_world(batch['t+1']['images'])
        action_embedding = model.action_encoder(batch['action_t'])
        latent_t_plus_1_pred = model.dynamics_model(outputs_t['latent_world_model'], action_embedding)
        loss_dynamics = self.dynamics_loss_fn(latent_t_plus_1_pred, latent_t_plus_1_real)
        total_loss = (self.weights['planning'] * loss_planning + self.weights['bev'] * loss_bev + self.weights['agent_det'] * loss_agents_det +
                      self.weights.get('agent_pred', 0) * loss_agents_pred + self.weights.get('dynamics', 0) * loss_dynamics)
        return {"total_loss": total_loss, "planning": loss_planning.detach(), "bev": loss_bev.detach(), 
                "agent_det": loss_agents_det.detach(), "agent_pred": loss_agents_pred.detach(), "dynamics": loss_dynamics.detach()}

# --- TRAINING FUNCTION ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    loss_tracker = {k: 0.0 for k in ["total_loss", "planning", "bev", "agent_det", "agent_pred", "dynamics"]}
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = criterion(model, batch)
        loss = loss_dict['total_loss']
        if torch.isnan(loss) or torch.isinf(loss): print(f"Invalid loss detected (NaN or Inf) at batch {i}. Skipping."); continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        for k, v in loss_dict.items(): loss_tracker[k] += v.item()
        if (i + 1) % 25 == 0 and i > 0:
            log_str = f"  Batch {i+1}/{len(dataloader)} | "
            for k, v in loss_tracker.items(): log_str += f"{k.capitalize()}: {v/(i+1):.3f} | "
            print(log_str[:-3])
    epoch_time = time.time() - start_time
    avg_loss = {k: v / len(dataloader) for k,v in loss_tracker.items()}
    print(f"Epoch finished in {epoch_time:.2f}s. Avg Losses -> Total: {avg_loss['total_loss']:.4f}, Plan: {avg_loss['planning']:.4f}, BEV: {avg_loss['bev']:.4f}, AgentDet: {avg_loss['agent_det']:.4f}, AgentPred: {avg_loss['agent_pred']:.4f}, Dynamics: {avg_loss['dynamics']:.4f}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    CONFIG = {
        "dataset_params": {
            "num_samples": 200, "img_height": 224, "img_width": 224,
        },
        "model_arch_params": {
            "d_model": 256, "nhead": 8, "num_encoder_layers": 4, "num_decoder_layers": 4,
            "dim_feedforward": 1024, "dropout": 0.1,
            "num_cameras": 6, "num_timesteps": 4, "num_trajectory_points": 10,
            "bev_height": 50, "bev_width": 50, "num_bev_classes": 3,
            "max_agents": 30, "num_agent_classes": 4,
            "num_future_steps": 10, "prediction_hidden_size": 128,
            "action_embedding_dim": 64
        },
        "training_params": {
            "batch_size": 4, "epochs": 5, "learning_rate": 5e-5, "num_workers": 2
        },
        "loss_weights": {
            "planning": 1.0, "bev": 0.8, "agent_det": 1.0, "agent_pred": 1.5, "dynamics": 2.5,
            "agent_ce": 1.0, "agent_bbox": 5.0, "agent_giou": 2.0
        }
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_config = {**CONFIG["dataset_params"], **CONFIG["model_arch_params"]}
    model_config = CONFIG["model_arch_params"]
    
    train_dataset = SimulatedPairedNuPlanDataset(**dataset_config)
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["training_params"]["batch_size"], shuffle=True, 
                                  num_workers=CONFIG["training_params"]["num_workers"], pin_memory=True if device.type == 'cuda' else False)
    
    model = WorldAwareTrajectoryTransformer(**model_config).to(device)
    
    criterion = WATT_Loss(loss_weights=CONFIG["loss_weights"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["training_params"]["learning_rate"], weight_decay=1e-4)

    print("\n--- Starting Full WATT Model Training ---")
    print(f"Using device: {device}")
    for epoch in range(CONFIG["training_params"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['training_params']['epochs']}")
        train_one_epoch(model, train_dataloader, optimizer, criterion, device)
    print("\n--- Training Finished ---")