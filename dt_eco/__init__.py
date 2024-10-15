from gymnasium.envs.registration import register
from mo_gymnasium.envs.dt_eco import dt_eco

register(
    id="dt-eco-v0",
    entry_point="mo_gymnasium.envs.dt_eco.dt_eco:DT_ECO",
    kwargs={'current_design': 'aes_cipher_top'}
)