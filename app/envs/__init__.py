from gym.envs.registration import register

from app.envs.inventoryv1.inventory_env import InventoryEnv

register(
    id="Blackjack-v1",
    entry_point="gym.envs.toy_text:BlackjackEnv",
)

register(
    id='Inventory-v1',
    entry_point='gym_inventory.envs:InventoryEnv',
    max_episode_steps=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)
