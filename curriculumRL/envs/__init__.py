from curriculumRL.envs.AI_CollabEnv import AI_CollabEnv

from gymnasium.envs.registration import register

register(
     id="AI_CollabEnv-v0",
     entry_point="curriculumRL.envs:AI_CollabEnv",
     max_episode_steps=00,
)
