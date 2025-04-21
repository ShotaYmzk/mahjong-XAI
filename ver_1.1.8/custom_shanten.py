# custom_shanten.py - Basic shanten (distance to ready hand) calculation for mahjong
import numpy as np
from collections import Counter

# Constants for tile types
MANS = list(range(0, 9))   # Man tiles (characters) - indices 0-8
PINS = list(range(9, 18))  # Pin tiles (circles) - indices 9-17
SOUS = list(range(18, 27)) # Sou tiles (bamboo) - indices 18-26
HONORS = list(range(27, 34)) # Honor tiles - indices 27-33

def convert_136_to_34(tile_id):
    """Convert a 136-format tile ID to a 34-format index"""
    return tile_id // 4

def convert_tiles_136_to_34_array(tiles):
    """Convert a list of 136-format tile IDs to a 34-array of counts"""
    counts = [0] * 34
    for tile in tiles:
        index = convert_136_to_34(tile)
        if 0 <= index < 34:
            counts[index] += 1
    return counts

def is_valid_chow(a, b, c):
    """Check if three tiles form a valid chow (sequence)"""
    # Must be same suit (man, pin, or sou) and consecutive
    if a // 9 != b // 9 or b // 9 != c // 9 or a // 9 == 3:  # Honor tiles can't form sequences
        return False
    # Check if the tiles form a sequence (they must be consecutive numbers in the same suit)
    sorted_values = sorted([a % 9, b % 9, c % 9])
    for i in range(7):
        if sorted_values == [i, i+1, i+2]:
            return True
    return False

def is_valid_pung(a, b, c):
    """Check if three tiles form a valid pung (triplet)"""
    return a == b == c

def find_all_sets(tiles_34):
    """Find all possible sets (pungs and chows) in the hand"""
    complete_sets = []
    
    # Find pungs (triplets)
    for i in range(34):
        if tiles_34[i] >= 3:
            complete_sets.append(('pung', [i, i, i]))
    
    # Find chows (sequences)
    for suit in [0, 1, 2]:  # Man, Pin, Sou suits
        base = suit * 9
        for i in range(7):  # Can only start a sequence from 0-6 within a suit
            idx1, idx2, idx3 = base + i, base + i + 1, base + i + 2
            if tiles_34[idx1] > 0 and tiles_34[idx2] > 0 and tiles_34[idx3] > 0:
                complete_sets.append(('chow', [idx1, idx2, idx3]))
    
    return complete_sets

def find_all_pairs(tiles_34):
    """Find all pairs in the hand"""
    return [i for i in range(34) if tiles_34[i] >= 2]

def basic_shanten_calculation(tiles_34):
    """Calculate the basic shanten number (distance to tenpai)"""
    # A complete hand needs 4 sets and 1 pair
    # Start with worst case: 8 (completely random tiles)
    min_shanten = 8
    
    # Find all potential sets and pairs
    sets = find_all_sets(tiles_34)
    pairs = find_all_pairs(tiles_34)
    
    # Try using different combinations of sets
    for num_sets in range(min(5, len(sets) + 1)):
        # Make a copy of tiles that we can modify
        remaining_tiles = tiles_34.copy()
        
        # Use the first num_sets sets
        # In a real implementation, you would try all combinations
        for i in range(min(num_sets, len(sets))):
            _, tile_indices = sets[i]
            for idx in tile_indices:
                remaining_tiles[idx] -= 1
        
        # Find pairs in remaining tiles
        remaining_pairs = [i for i in range(34) if remaining_tiles[i] >= 2]
        
        # Calculate shanten based on this arrangement
        # Formula: 8 - 2*(number of sets) - (pair exists ? 1 : 0)
        shanten = 8 - 2 * num_sets
        if remaining_pairs:
            shanten -= 1
            
        min_shanten = min(min_shanten, shanten)
    
    # Can't go below -1 (complete hand)
    return max(-1, min_shanten)

def calculate_shanten(hand_tiles, melds=None):
    """
    Calculate the shanten number for a mahjong hand.
    
    Args:
        hand_tiles: List of tile IDs (136 format)
        melds: List of meld dictionaries
        
    Returns:
        tuple: (shanten_number, ukeire)
            - shanten_number: Distance to ready hand (-1=complete, 0=tenpai, etc.)
            - ukeire: List of tile indices that would improve the hand
    """
    # Convert hand to 34-format array
    tiles_34 = convert_tiles_136_to_34_array(hand_tiles)
    
    # Account for melds
    num_melds = 0
    if melds:
        for meld in melds:
            num_melds += 1
            meld_tiles = meld.get('tiles', [])
            for tile in meld_tiles:
                idx = convert_136_to_34(tile)
                tiles_34[idx] += 1
    
    # Calculate basic shanten
    shanten = basic_shanten_calculation(tiles_34)
    
    # Adjust for melds (each meld counts as one complete set)
    shanten -= num_melds
    
    # Simple ukeire calculation - tiles that would reduce shanten if drawn
    ukeire = []
    if shanten >= 0:  # Only calculate ukeire if not already a complete hand
        for i in range(34):
            if tiles_34[i] < 4:  # Can't have more than 4 of any tile
                # Try adding this tile
                tiles_34[i] += 1
                new_shanten = basic_shanten_calculation(tiles_34) - num_melds
                if new_shanten < shanten:
                    ukeire.append(i)
                # Remove the tile
                tiles_34[i] -= 1
    
    return max(-1, shanten), ukeire

# Simplified function to match the interface in game_state.py
def calculate_shanten_and_ukeire(hand_tile_ids, melds_data=None):
    """
    Public interface for shanten calculation.
    
    Args:
        hand_tile_ids: List of tile IDs (136 format)
        melds_data: List of meld dictionaries
        
    Returns:
        tuple: (shanten_number, ukeire)
    """
    return calculate_shanten(hand_tile_ids, melds_data) 