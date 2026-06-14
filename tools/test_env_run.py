from agv_ga_robot.utils.helpers import load_config
from agv_ga_robot.maps.map_loader import MapLoader
from agv_ga_robot.env.agv_env import AGVEnvironment

if __name__ == '__main__':
    config = load_config('agv_ga_robot/config/config.yaml')
    map_path = config.get('MAP', 'agv_ga_robot/maps/learning_maze_simple.json')
    map_data = MapLoader.load_map(map_path)
    env = AGVEnvironment(config, map_data)
    obs, _ = env.reset()
    print('Reset done')
    for i in range(120):
        action = [0.5, 0.5]  # try driving forward
        obs, reward, terminated, truncated, info = env.step(action)
        avg_speed = env.robot.get_average_speed()
        print(f'step={i:03d} reward={reward:.2f} collision={info.get("collision")} steps_without_progress={env.steps_without_progress} avg_speed={avg_speed:.3f}')
        if terminated:
            print('Terminated at step', i)
            break
    print('Summary:', env.episode_data.get_summary())
