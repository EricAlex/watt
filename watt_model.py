import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from einops import rearrange, repeat

# Attempt to import ncps, but make it optional so the model can be
# instantiated without it if only some parts are used.
try:
    from ncps.torch import CfC
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False
    print("Warning: ncps library not found. AgentPredictionHead will not be available.")
    # Define a dummy class if ncps is not available
    class CfC(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("Please install ncps: pip install ncps")
        def forward(self, *args, **kwargs):
            raise NotImplementedError

# --- Perception Backbone ---
class ImageEncoder(nn.Module):
    """
    Encodes a batch of multi-camera, multi-timestamp images into feature patches.
    This module acts as the perception backbone.
    """
    def __init__(self, d_model, in_channels=3):
        super().__init__()
        self.d_model = d_model
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        backbone = convnext_tiny(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.features.children())[:-2])
        
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            dummy_output = self.feature_extractor(dummy_input)
            backbone_out_dim = dummy_output.shape[1]

        self.projection = nn.Linear(backbone_out_dim, d_model)

    def forward(self, images):
        b, t, c, h, w, _ = images.shape
        x = rearrange(images, 'b t c h w d -> (b t c) d h w')
        features = self.feature_extractor(x)
        patches = rearrange(features, 'btc d h w -> btc (h w) d')
        tokens = self.projection(patches)
        final_tokens = rearrange(tokens, '(b t c) p d -> b (t c p) d', b=b, t=t, c=c)
        return final_tokens

# --- World Model Auxiliary Heads ---
class BevProjectionHead(nn.Module):
    """Projects latent tokens onto a BEV grid for static segmentation."""
    def __init__(self, d_model, bev_height, bev_width, num_bev_classes, 
                 nhead=8, num_decoder_layers=2, dim_feedforward=1024):
        super().__init__()
        self.d_model, self.bev_height, self.bev_width = d_model, bev_height, bev_width
        num_bev_queries = bev_height * bev_width
        self.bev_queries = nn.Parameter(torch.randn(1, num_bev_queries, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, num_bev_classes))

    def forward(self, latent_world_model):
        b = latent_world_model.shape[0]
        bev_queries = repeat(self.bev_queries, '1 n d -> b n d', b=b)
        bev_features = self.decoder(tgt=bev_queries, memory=latent_world_model)
        bev_logits = self.classifier(bev_features)
        return rearrange(bev_logits, 'b (h w) c -> b c h w', h=self.bev_height, w=self.bev_width)

class AgentDetectionHead(nn.Module):
    """Detects dynamic agents using a DETR-like approach."""
    def __init__(self, d_model, max_agents=50, num_agent_classes=4, 
                 nhead=8, num_decoder_layers=2, dim_feedforward=1024):
        super().__init__()
        self.agent_queries = nn.Parameter(torch.randn(1, max_agents, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.class_head = nn.Linear(d_model, num_agent_classes)
        self.bbox_head = nn.Linear(d_model, 5)

    def forward(self, latent_world_model):
        b = latent_world_model.shape[0]
        agent_queries = repeat(self.agent_queries, '1 n d -> b n d', b=b)
        agent_features = self.decoder(tgt=agent_queries, memory=latent_world_model)
        agent_features_norm = self.output_norm(agent_features)
        class_logits = self.class_head(agent_features_norm)
        pred_boxes = self.bbox_head(agent_features_norm).sigmoid()
        # Return features for the prediction head to use
        return {'pred_logits': class_logits, 'pred_boxes': pred_boxes, 'decoder_features': agent_features}

class AgentPredictionHead(nn.Module):
    """Predicts future trajectories for detected agents using a CfC RNN."""
    def __init__(self, d_model, num_future_steps=10, prediction_hidden_size=128):
        super().__init__()
        if not NCPS_AVAILABLE:
            raise ImportError("AgentPredictionHead requires the 'ncps' library.")
        self.num_future_steps = num_future_steps
        self.state_init_fc = nn.Linear(d_model, prediction_hidden_size * 2) # For h_0 and c_0
        self.control_input = nn.Parameter(torch.randn(1, 1, d_model))
        self.rnn = CfC(input_size=d_model, units=prediction_hidden_size, proj_size=2, 
                       return_sequences=True, batch_first=True, mixed_memory=True)

    def forward(self, agent_features):
        if agent_features.shape[0] == 0:
            return torch.zeros(0, self.num_future_steps, 2, device=agent_features.device)
        num_agents = agent_features.shape[0]
        h_c_init = self.state_init_fc(agent_features)
        h_0, c_0 = torch.chunk(h_c_init, 2, dim=-1)
        rnn_input = repeat(self.control_input, '1 1 d -> n t d', n=num_agents, t=self.num_future_steps)
        predicted_trajectories, _ = self.rnn(rnn_input, (h_0, c_0))
        return predicted_trajectories

# --- Simulation Engine Components ---
class ActionEncoder(nn.Module):
    """Encodes a planned trajectory (action) into a compact latent vector."""
    def __init__(self, d_model, num_action_points=10, action_embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_action_points * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_embedding_dim)
        )
    def forward(self, actions):
        b = actions.shape[0]
        return self.encoder(actions.view(b, -1))

class LatentDynamicsModel(nn.Module):
    """Learns the transition dynamics of the world in latent space: z_{t+1} = f(z_t, a_t)."""
    def __init__(self, d_model, action_embedding_dim=64, hidden_dim=1024):
        super().__init__()
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(d_model + action_embedding_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, d_model) # Predicts the change (delta)
        )
    def forward(self, latent_state_tokens, action_embedding):
        z_t_mean = latent_state_tokens.mean(dim=1)
        z_a_concat = torch.cat([z_t_mean, action_embedding], dim=1)
        delta_z_pred = self.dynamics_mlp(z_a_concat)
        return latent_state_tokens + delta_z_pred.unsqueeze(1)


# --- The Main Unified Model ---
class WorldAwareTrajectoryTransformer(nn.Module):
    """
    The complete, perfected WATT model with perception, prediction, and simulation capabilities.
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1,
                 # Parameters for sub-modules
                 num_cameras=6, num_timesteps=4, num_trajectory_points=10,
                 bev_height=50, bev_width=50, num_bev_classes=3,
                 max_agents=30, num_agent_classes=4,
                 num_future_steps=10, prediction_hidden_size=128,
                 action_embedding_dim=64):
        super().__init__()
        self.d_model = d_model
        self.num_trajectory_points = num_trajectory_points
        
        # --- 1. Perception Encoder (The "Eyes") ---
        self.image_encoder = ImageEncoder(d_model=d_model)
        self.camera_embedding = nn.Embedding(num_cameras, d_model)
        self.time_embedding = nn.Embedding(num_timesteps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # --- 2. Ego-Planner Head (The "Hands") ---
        self.planning_queries = nn.Parameter(torch.randn(1, num_trajectory_points, d_model))
        planner_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True)
        self.planner_decoder = nn.TransformerDecoder(planner_decoder_layer, num_decoder_layers)
        self.trajectory_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 2))
        
        # --- 3. World Model Heads (The "Inner World") ---
        self.bev_segmentation_head = BevProjectionHead(d_model, bev_height, bev_width, num_bev_classes, nhead)
        self.agent_detection_head = AgentDetectionHead(d_model, max_agents, num_agent_classes, nhead)
        if NCPS_AVAILABLE:
            self.agent_prediction_head = AgentPredictionHead(d_model, num_future_steps, prediction_hidden_size)
        
        # --- 4. Simulation Engine (The "Imagination") ---
        self.action_encoder = ActionEncoder(d_model, num_trajectory_points, action_embedding_dim)
        self.dynamics_model = LatentDynamicsModel(d_model, action_embedding_dim)

    def encode_world(self, images):
        """Helper function to run the main perception encoder."""
        b, t, c, _, _, _ = images.shape
        image_tokens = self.image_encoder(images)
        num_patches_per_image = image_tokens.shape[1] // (t * c)
        
        time_indices = torch.arange(t, device=images.device).repeat_interleave(c * num_patches_per_image)
        cam_indices = torch.arange(c, device=images.device).repeat_interleave(num_patches_per_image).repeat(t)
        time_indices = repeat(time_indices, '... -> b ...', b=b)
        cam_indices = repeat(cam_indices, '... -> b ...', b=b)
        
        contextualized_tokens = image_tokens + self.time_embedding(time_indices) + self.camera_embedding(cam_indices)
        return self.transformer_encoder(contextualized_tokens)

    def forward(self, images):
        """
        Standard forward pass for multi-task training.
        Does not include prediction or simulation, which are handled in the loss/planner.
        """
        # --- 1. Encode sensor data into the latent world model ---
        latent_world_model = self.encode_world(images)
        b = latent_world_model.shape[0]
        
        # --- 2. Decode the Ego-Plan ---
        planning_queries = repeat(self.planning_queries, '1 n d -> b n d', b=b)
        plan_features = self.planner_decoder(tgt=planning_queries, memory=latent_world_model)
        predicted_trajectory = self.trajectory_head(plan_features)
        
        # --- 3. Decode the World Model representations ---
        bev_map_logits = self.bev_segmentation_head(latent_world_model)
        agent_detection_output = self.agent_detection_head(latent_world_model)
        
        return {
            "trajectory": predicted_trajectory,
            "bev_map_logits": bev_map_logits,
            "agent_detection": agent_detection_output,
            "latent_world_model": latent_world_model, # Pass this out for dynamics loss
        }


# --- Example Usage & Sanity Check ---
if __name__ == '__main__':
    # Define model parameters
    config = {
        "num_cameras": 6, "num_timesteps": 4, "num_trajectory_points": 10,
        "d_model": 256, "nhead": 8, "num_encoder_layers": 4, "num_decoder_layers": 4,
        "bev_height": 50, "bev_width": 50, "num_bev_classes": 3,
        "max_agents": 30, "num_agent_classes": 4,
        "num_future_steps": 10, "prediction_hidden_size": 128,
        "action_embedding_dim": 64
    }
    BATCH_SIZE = 2
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    
    # Instantiate the full model
    model = WorldAwareTrajectoryTransformer(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Full WATT Model created and moved to {device}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")
    
    # Create dummy input
    dummy_images = torch.randn(BATCH_SIZE, config["num_timesteps"], config["num_cameras"], 
                               IMG_HEIGHT, IMG_WIDTH, 3).to(device)
    
    # Perform a forward pass
    print("\nPerforming a forward pass for training...")
    output = model(dummy_images)
    print("Forward pass successful!")
    
    # Verify outputs
    print("\n--- Verifying Training Output Shapes ---")
    assert output['trajectory'].shape == (BATCH_SIZE, config['num_trajectory_points'], 2)
    assert output['bev_map_logits'].shape == (BATCH_SIZE, config['num_bev_classes'], config['bev_height'], config['bev_width'])
    assert output['agent_detection']['pred_logits'].shape == (BATCH_SIZE, config['max_agents'], config['num_agent_classes'])
    assert 'latent_world_model' in output
    print("All training output shapes are correct.")
    
    # --- Sanity Check the Simulation Engine ---
    print("\n--- Verifying Simulation Engine ---")
    latent_z_t = output['latent_world_model']
    dummy_action = torch.randn(BATCH_SIZE, config['num_trajectory_points'], 2).to(device)
    
    action_embedding = model.action_encoder(dummy_action)
    latent_z_t_plus_1 = model.dynamics_model(latent_z_t, action_embedding)
    
    assert action_embedding.shape == (BATCH_SIZE, config['action_embedding_dim'])
    assert latent_z_t_plus_1.shape == latent_z_t.shape
    print("Simulation engine forward pass is successful and shapes are correct.")

    # --- Sanity Check the Prediction Head (if available) ---
    if NCPS_AVAILABLE:
        print("\n--- Verifying Prediction Head ---")
        agent_features = output['agent_detection']['decoder_features'] # (B, max_agents, d_model)
        # In a real scenario, we only predict for detected agents, so we'd select them.
        # Here we just use all queries for a shape check.
        predicted_agent_futures = model.agent_prediction_head(agent_features.view(-1, config['d_model']))
        expected_shape = (BATCH_SIZE * config['max_agents'], config['num_future_steps'], 2)
        assert predicted_agent_futures.shape == expected_shape
        print("Agent prediction head forward pass is successful and shapes are correct.")