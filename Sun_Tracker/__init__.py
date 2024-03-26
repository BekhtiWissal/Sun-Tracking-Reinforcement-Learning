from gym.envs.registration import register

register(
    id='Sun_Tracker/SolarTracking-v0',
    entry_point='Sun_Tracker.envs:SolarTrackingEnv',
    max_episode_steps=5000,
)