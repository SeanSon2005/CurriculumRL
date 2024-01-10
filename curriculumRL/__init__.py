from gymnasium.envs.registration import register

register(
     id="AI_Collab-v0",
     entry_point="curriculumRL.envs:AI_CollabEnv",
     max_episode_steps=00,
)
