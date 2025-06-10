import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
import numpy as np
import time

# Import the model from our previous file
from watt_model import WorldAwareTrajectoryTransformer

# --- UTILITY FUNCTIONS ---

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Converts boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
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

# --- DATASET DEFINITION ---

class SimulatedNuPlanDataset(Dataset):
    """A simulated PyTorch Dataset that mimics the output of the nuplan-sdk."""
    def __init__(self, num_samples=1000, num_timesteps=4, num_cameras=6,
                 img_height=224, img_width=224, num_trajectory_points=10,
                 bev_height=100, bev_width=100, num_bev_classes=3,
                 max_agents=50, num_future_steps=10):
        self.num_samples, self.num_timesteps, self.num_cameras = num_samples, num_timesteps, num_cameras
        self.img_height, self.img_width, self.num_trajectory_points = img_height, img_width, num_trajectory_points
        self.bev_height, self.bev_width, self.num_bev_classes = bev_height, bev_width, num_bev_classes
        self.max_agents, self.num_future_steps = max_agents, num_future_steps
        print(f"Creating a simulated dataset with {num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images = torch.rand(self.num_timesteps, self.num_cameras, self.img_height, self.img_width, 3)
        start_point = np.random.rand(2) * 5
        end_point = start_point + np.random.rand(2) * 20
        trajectory = torch.from_numpy(np.linspace(start_point, end_point, self.num_trajectory_points)).float()
        bev_map = torch.randint(0, self.num_bev_classes, (self.bev_height, self.bev_width)).long()
        num_agents = np.random.randint(5, 20)
        gt_boxes = torch.rand(num_agents, 4)
        gt_labels = torch.randint(0, 3, (num_agents,)).long()
        padded_boxes = torch.zeros(self.max_agents, 4)
        padded_labels = torch.full((self.max_agents,), 3)
        padded_boxes[:num_agents] = gt_boxes
        padded_labels[:num_agents] = gt_labels
        future_trajectories = torch.zeros(self.max_agents, self.num_future_steps, 2)
        start_points = padded_boxes[:num_agents, :2]
        for i in range(num_agents):
            end_point = start_points[i] + torch.rand(2) * 5
            future_trajectories[i] = torch.from_numpy(np.linspace(start_points[i].numpy(), end_point.numpy(), self.num_future_steps)).float()
        return {"images": images, "ground_truth_trajectory": trajectory, "ground_truth_bev_map": bev_map,
                "ground_truth_agents": {"boxes": padded_boxes, "labels": padded_labels, "future_trajectories": future_trajectories}}

# --- LOSS FUNCTION DEFINITIONS ---

class AgentDetectionLoss(nn.Module):
    """Computes the loss for the agent detection head using bipartite matching (DETR-style)."""
    def __init__(self, num_classes, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class_w, self.cost_bbox_w, self.cost_giou_w = cost_class, cost_bbox, cost_giou
        empty_weight = torch.tensor([0.1] * num_classes + [1.0])
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, pred_outputs, gt_targets):
        pred_logits, pred_boxes = pred_outputs['pred_logits'], pred_outputs['pred_boxes']
        indices = []
        for i in range(pred_logits.shape[0]):
            log_probs_i = F.log_softmax(pred_logits[i], dim=-1)
            boxes_i_cxcywh = pred_boxes[i][..., :4]
            gt_labels_i_all, gt_boxes_i_cxcywh_all = gt_targets['labels'][i], gt_targets['boxes'][i]
            valid_gt_mask = gt_labels_i_all < self.num_classes
            gt_labels_i, gt_boxes_i_cxcywh = gt_labels_i_all[valid_gt_mask], gt_boxes_i_cxcywh_all[valid_gt_mask]
            if gt_labels_i.numel() == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=pred_logits.device), torch.tensor([], dtype=torch.long, device=pred_logits.device)))
                continue
            cost_class = -log_probs_i[:, gt_labels_i]
            cost_bbox = torch.cdist(boxes_i_cxcywh, gt_boxes_i_cxcywh, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes_i_cxcywh), box_cxcywh_to_xyxy(gt_boxes_i_cxcywh))
            cost_matrix = self.cost_bbox_w * cost_bbox + self.cost_class_w * cost_class + self.cost_giou_w * cost_giou
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach())
            indices.append((torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)))

        src_idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        if src_idx[0].numel() > 0:
            matched_gt_labels = torch.cat([gt_targets['labels'][i][j] for i, (_, j) in enumerate(indices)])
            target_classes[src_idx] = matched_gt_labels
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)

        if src_idx[0].numel() > 0:
            matched_pred_boxes = pred_boxes[src_idx]
            matched_gt_boxes = torch.cat([gt_targets['boxes'][i][j] for i, (_, j) in enumerate(indices)])
            loss_bbox = F.l1_loss(matched_pred_boxes[..., :4], matched_gt_boxes)
            loss_giou = (1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(matched_pred_boxes[..., :4]), box_cxcywh_to_xyxy(matched_gt_boxes)))).mean()
        else:
            loss_bbox = loss_giou = torch.tensor(0.0, device=pred_logits.device)
        return {'loss_ce': loss_ce, 'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

class WATT_Loss(nn.Module):
    """Calculates the combined loss for the WATT model."""
    def __init__(self, planning_weight=1.0, bev_weight=0.5, agents_weight=0.5, bev_label_smoothing=0.0,
                 agent_ce_weight=1.0, agent_bbox_weight=5.0, agent_giou_weight=2.0):
        super().__init__()
        self.planning_weight, self.bev_weight, self.agents_weight = planning_weight, bev_weight, agents_weight
        self.agent_ce_w, self.agent_bbox_w, self.agent_giou_w = agent_ce_weight, agent_bbox_weight, agent_giou_weight
        self.planning_loss_fn = nn.HuberLoss()
        self.bev_loss_fn = nn.CrossEntropyLoss(label_smoothing=bev_label_smoothing)
        self.agent_loss_fn = AgentDetectionLoss(num_classes=3)

    def forward(self, model_outputs, ground_truth):
        loss_planning = self.planning_loss_fn(model_outputs['trajectory'], ground_truth['ground_truth_trajectory'])
        loss_bev = self.bev_loss_fn(model_outputs['bev_map_logits'], ground_truth['ground_truth_bev_map'])
        agent_losses = self.agent_loss_fn(model_outputs['agent_predictions'], ground_truth['ground_truth_agents'])
        loss_agents = (self.agent_ce_w * agent_losses['loss_ce'] + self.agent_bbox_w * agent_losses['loss_bbox'] + self.agent_giou_w * agent_losses['loss_giou'])
        total_loss = self.planning_weight * loss_planning + self.bev_weight * loss_bev + self.agents_weight * loss_agents
        return {"total_loss": total_loss, "planning_loss": loss_planning.detach(), "bev_loss": loss_bev.detach(), "agents_loss": loss_agents.detach()}

# --- TRAINING FUNCTION ---

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss_sum, planning_loss_sum, bev_loss_sum, agents_loss_sum = 0, 0, 0, 0
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        batch = move_to_device(batch, device)
        images, ground_truth = batch['images'], {k: v for k, v in batch.items() if k != 'images'}
        optimizer.zero_grad()
        model_outputs = model(images)
        loss_dict = criterion(model_outputs, ground_truth)
        loss = loss_dict['total_loss']
        if torch.isnan(loss):
            print("NaN loss detected. Skipping batch.")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss_sum += loss.item()
        planning_loss_sum += loss_dict.get('planning_loss', torch.tensor(0.0)).item()
        bev_loss_sum += loss_dict.get('bev_loss', torch.tensor(0.0)).item()
        agents_loss_sum += loss_dict.get('agents_loss', torch.tensor(0.0)).item()
        if i > 0 and (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Avg Total Loss: {total_loss_sum/(i+1):.4f}, Plan: {planning_loss_sum/(i+1):.4f}, BEV: {bev_loss_sum/(i+1):.4f}, Agents: {agents_loss_sum/(i+1):.4f}")
    epoch_time = time.time() - start_time
    print(f"Epoch finished in {epoch_time:.2f}s. Average Total Loss: {total_loss_sum / len(dataloader):.4f}")

# --- MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    CONFIG = {"num_samples": 500, "batch_size": 1, "epochs": 5, "learning_rate": 1e-4, "d_model": 256, "nhead": 8,
              "num_encoder_layers": 4, "num_decoder_layers": 4, "loss_planning_weight": 1.0, "loss_bev_weight": 0.5,
              "loss_agents_weight": 1.0, "bev_height": 100, "bev_width": 100, "num_bev_classes": 3, "max_agents": 50}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SimulatedNuPlanDataset(num_samples=CONFIG["num_samples"], bev_height=CONFIG["bev_height"],
                                           bev_width=CONFIG["bev_width"], num_bev_classes=CONFIG["num_bev_classes"],
                                           max_agents=CONFIG["max_agents"])
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    model = WorldAwareTrajectoryTransformer(d_model=CONFIG["d_model"], nhead=CONFIG["nhead"],
                                            num_encoder_layers=CONFIG["num_encoder_layers"], num_decoder_layers=CONFIG["num_decoder_layers"],
                                            bev_height=CONFIG["bev_height"], bev_width=CONFIG["bev_width"], num_bev_classes=CONFIG["num_bev_classes"],
                                            max_agents=CONFIG["max_agents"]).to(device)
    criterion = WATT_Loss(planning_weight=CONFIG["loss_planning_weight"], bev_weight=CONFIG["loss_bev_weight"],
                          agents_weight=CONFIG["loss_agents_weight"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    print("\n--- Starting Training ---")
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        train_one_epoch(model, train_dataloader, optimizer, criterion, device)
    print("\n--- Training Finished ---")