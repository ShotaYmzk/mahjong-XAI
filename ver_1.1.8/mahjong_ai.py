# mahjong_ai.py - Transformer-based mahjong AI inference
import torch
import numpy as np
import os
import sys
import logging
import time
from collections import defaultdict

# --- Ensure other modules can be imported ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
# ---------------------------------------------

try:
    # Import GameState class
    from game_state import GameState, NUM_TILE_TYPES, NUM_PLAYERS
    # Import tile utilities
    from tile_utils import tile_id_to_index, tile_index_to_id, tile_id_to_string
    # Import the model architecture from train.py
    from train import MahjongTransformerV2, D_MODEL, NHEAD, D_HID, NLAYERS, DROPOUT, ACTIVATION
except ImportError as e:
    print(f"[FATAL ERROR in mahjong_ai.py] Failed to import modules: {e}")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = "./trained_model/mahjong_transformer_v2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
LOG_FILE = "./logs/ai_inference.log"
DEBUG_MODE = True  # Set to True for detailed decision explanations
# ---------------------------------

# --- Logging Setup ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
# -----------------------

class MahjongAI:
    """Advanced Transformer-based Mahjong AI for optimal tile discard decisions"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.game_state = GameState()  # Initialize game state
        self.model = None
        self.event_dim = None  # Will be set when loading model
        self.static_dim = None  # Will be set when loading model
        self.load_model(model_path)
        self.last_decision_explanation = ""
        self.decision_history = []
        
    def load_model(self, model_path):
        """Load the trained transformer model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # For now, we'll set reasonable defaults for dimension sizes
        # These would typically be determined from the training data
        self.event_dim = 32  # Must match feature_extractor_imitation.py
        self.static_dim = 192  # Must match feature_extractor_imitation.py
        
        # Initialize the model with proper dimensions
        self.model = MahjongTransformerV2(
            event_feature_dim=self.event_dim,
            static_feature_dim=self.static_dim,
            d_model=D_MODEL,
            nhead=NHEAD,
            d_hid=D_HID,
            nlayers=NLAYERS,
            dropout=DROPOUT,
            activation=ACTIVATION,
            output_dim=NUM_TILE_TYPES
        )
        
        # Load the trained weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()  # Set to evaluation mode
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
            
    def initialize_round(self, round_data):
        """Initialize the game state for a new round"""
        self.game_state.init_round(round_data)
        self.decision_history = []
        logging.info(f"Initialized round: {round_data.get('round_index', 'Unknown')}")
        
    def process_game_event(self, event):
        """Process a game event to update the internal state"""
        tag = event["tag"]
        attrib = event["attrib"]
        
        try:
            # Handle different event types
            if tag.startswith("T") or tag.startswith("U") or tag.startswith("V") or tag.startswith("W"):
                # Tsumo event
                for t_tag, p_id in GameState.TSUMO_TAGS.items():
                    if tag.startswith(t_tag) and tag[1:].isdigit():
                        tile_id = int(tag[1:])
                        self.game_state.process_tsumo(p_id, tile_id)
                        logging.debug(f"Processed Tsumo: Player {p_id}, Tile {tile_id_to_string(tile_id)}")
                        break
                        
            elif tag.startswith("D") or tag.startswith("E") or tag.startswith("F") or tag.startswith("G"):
                # Discard event
                for d_tag, p_id in GameState.DISCARD_TAGS.items():
                    if tag.startswith(d_tag) and tag[1:].isdigit():
                        tile_id = int(tag[1:])
                        tsumogiri = tag[0].islower()
                        self.game_state.process_discard(p_id, tile_id, tsumogiri)
                        logging.debug(f"Processed Discard: Player {p_id}, Tile {tile_id_to_string(tile_id)}")
                        break
                        
            elif tag == "N":  # Naki (call) event
                naki_player_id = int(attrib.get("who", -1))
                meld_code = int(attrib.get("m", "0"))
                if naki_player_id != -1:
                    self.game_state.process_naki(naki_player_id, meld_code)
                    logging.debug(f"Processed Naki: Player {naki_player_id}, Code {meld_code}")
                    
            elif tag == "REACH":  # Riichi declaration
                reach_player_id = int(attrib.get("who", -1))
                step = int(attrib.get("step", -1))
                if reach_player_id != -1 and step == 1:
                    self.game_state.process_reach(reach_player_id, step)
                    logging.debug(f"Processed Reach: Player {reach_player_id}, Step {step}")
                    
            elif tag == "DORA":  # New dora indicator revealed
                hai_attr = attrib.get("hai")
                if hai_attr is not None and hai_attr.isdigit():
                    dora_indicator_id = int(hai_attr)
                    self.game_state.process_dora(dora_indicator_id)
                    logging.debug(f"Processed Dora: Indicator {tile_id_to_string(dora_indicator_id)}")
                    
            elif tag == "AGARI":  # Win
                self.game_state.process_agari(attrib)
                logging.debug(f"Processed Agari: {attrib}")
                
            elif tag == "RYUUKYOKU":  # Draw
                self.game_state.process_ryuukyoku(attrib)
                logging.debug(f"Processed Ryuukyoku: {attrib}")
                
        except Exception as e:
            logging.error(f"Error processing event {tag}: {e}")
            
    def decide_discard(self, player_id, with_explanation=True):
        """
        Decide which tile to discard based on the current game state.
        
        Args:
            player_id: The player ID (0-3) that needs to make a discard decision
            with_explanation: Whether to generate an explanation
            
        Returns:
            tile_id: The ID of the tile to discard
            explanation: A string explaining the decision if with_explanation=True
        """
        try:
            # Get features for the current state
            seq_features = self.game_state.get_event_sequence_features()
            static_features = self.game_state.get_static_features(player_id)
            
            # Get current hand for validation
            hand_indices = self.game_state.get_hand_indices(player_id)
            valid_discard_options = self.game_state.get_valid_discard_options(player_id)
            
            if not valid_discard_options:
                logging.warning("No valid discard options available")
                return -1
                
            # Create attention mask for padding tokens
            # We need to identify which positions in the sequence are padding
            padding_code = float(GameState.EVENT_TYPES["PADDING"])
            padding_mask = (seq_features[:, 0] == padding_code)
            
            # Convert to PyTorch tensors
            seq_tensor = torch.FloatTensor(seq_features).unsqueeze(0).to(DEVICE)
            static_tensor = torch.FloatTensor(static_features).unsqueeze(0).to(DEVICE)
            padding_tensor = torch.BoolTensor(padding_mask).unsqueeze(0).to(DEVICE)
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(seq_tensor, static_tensor, padding_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                
            # Filter probabilities to only valid discard options
            valid_probs = np.zeros(NUM_TILE_TYPES)
            for idx in valid_discard_options:
                if 0 <= idx < NUM_TILE_TYPES:
                    valid_probs[idx] = probs[idx]
                    
            # Normalize probabilities for valid options only
            if np.sum(valid_probs) > 0:
                valid_probs = valid_probs / np.sum(valid_probs)
                
            # Get tile index with highest probability
            best_tile_index = np.argmax(valid_probs)
            
            # Convert back to tile ID
            best_tile_id = tile_index_to_id(best_tile_index)
            
            # Generate explanation if requested
            explanation = ""
            if with_explanation:
                # Get top 3 candidates
                top3_indices = np.argsort(valid_probs)[-3:][::-1]
                top3_probs = valid_probs[top3_indices]
                
                explanation = f"Decision: Discard {tile_id_to_string(best_tile_id)} (idx:{best_tile_index}, prob:{valid_probs[best_tile_index]:.3f})\n"
                explanation += f"Top 3 candidates:\n"
                
                for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
                    tile_id = tile_index_to_id(idx)
                    explanation += f"  {i+1}. {tile_id_to_string(tile_id)} (idx:{idx}, prob:{prob:.3f})\n"
                    
                # Add context about the hand
                explanation += f"\nCurrent hand: "
                for idx in sorted(hand_indices):
                    tile_id = tile_index_to_id(idx)
                    explanation += f"{tile_id_to_string(tile_id)} "
                    
                # Add shanten information if available
                shanten = -1
                if hasattr(self.game_state, 'calculate_shanten'):
                    hand_ids = [tile_index_to_id(idx) for idx in hand_indices]
                    melds = self.game_state.get_melds_indices(player_id)
                    shanten, ukeire = self.game_state.calculate_shanten(hand_ids, melds)
                    explanation += f"\nShanten: {shanten}"
                    
                self.last_decision_explanation = explanation
                
            # Record the decision
            decision_record = {
                'tile_id': best_tile_id,
                'tile_index': best_tile_index,
                'probabilities': valid_probs.tolist(),
                'hand_indices': hand_indices,
                'junme': self.game_state.junme
            }
            self.decision_history.append(decision_record)
            
            logging.debug(f"AI decided to discard {tile_id_to_string(best_tile_id)}")
            return best_tile_id, explanation
            
        except Exception as e:
            logging.error(f"Error in decide_discard: {e}")
            return -1, f"Error: {e}"
            
    def analyze_game_statistics(self):
        """Analyze the AI's performance statistics for the current game"""
        if not self.decision_history:
            return "No decisions recorded yet."
            
        stats = {
            'total_decisions': len(self.decision_history),
            'avg_prob': np.mean([d['probabilities'][d['tile_index']] for d in self.decision_history]),
            'decisions_by_junme': defaultdict(int)
        }
        
        for decision in self.decision_history:
            junme = int(decision['junme'])
            stats['decisions_by_junme'][junme] += 1
            
        report = f"Game Statistics:\n"
        report += f"Total decisions: {stats['total_decisions']}\n"
        report += f"Average confidence: {stats['avg_prob']:.3f}\n"
        report += f"Decisions by turn number:\n"
        
        for junme in sorted(stats['decisions_by_junme'].keys()):
            report += f"  Turn {junme}: {stats['decisions_by_junme'][junme]} decisions\n"
            
        return report
            
    def reset(self):
        """Reset the game state"""
        self.game_state.reset_state()
        self.decision_history = []
        self.last_decision_explanation = ""
        logging.info("AI state reset")

# --- Example usage ---
def demo():
    """Demo function to show how to use the AI"""
    ai = MahjongAI()
    
    # This would typically be loaded from a game log or live connection
    sample_round_data = {
        "round_index": 0,
        "init": {
            "seed": "0,0,0,0,0,135",  # Game seed includes dora indicator
            "oya": 0,  # Dealer is Player 0
            "ten": "25000,25000,25000,25000",  # Starting scores
            "hai0": "1,2,3,4,13,14,15,16,22,24,33,41,43,45",  # Player 0's hand
            "hai1": "5,6,7,8,17,18,26,28,30,35,47,49,51,53",  # Player 1's hand
            "hai2": "9,10,11,12,19,20,27,29,31,37,55,57,59,61",  # Player 2's hand
            "hai3": "21,23,25,32,34,36,38,39,40,42,63,65,67,69"   # Player 3's hand
        },
        "events": []  # Events would be populated during a game
    }
    
    # Initialize the round
    ai.initialize_round(sample_round_data)
    
    # Example tsumo event
    tsumo_event = {
        "tag": "T71",  # Player 0 draws tile 71
        "attrib": {}
    }
    ai.process_game_event(tsumo_event)
    
    # Decide which tile to discard
    tile_to_discard, explanation = ai.decide_discard(player_id=0, with_explanation=True)
    
    print(f"AI demo completed. Decided to discard: {tile_id_to_string(tile_to_discard)}")
    print("\nExplanation:")
    print(explanation)

if __name__ == "__main__":
    logging.info("=== Starting Mahjong AI ===")
    demo()
    logging.info("=== Mahjong AI finished ===") 