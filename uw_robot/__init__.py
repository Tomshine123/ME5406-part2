from gym.envs.registration import register

register(
    id='CustomEnv-v0',
    entry_point='uw_robot.env:UnderwaterRobot',
    max_episode_steps=1000,
    reward_threshold=200,
)

