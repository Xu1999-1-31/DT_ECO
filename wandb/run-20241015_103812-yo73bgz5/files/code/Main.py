import argparse
import mo_gymnasium as mo_gym
import DT_ECO
from tee import StdoutTee, StderrTee, close_all

parser = argparse.ArgumentParser(description='This is the main python program for DT_ECO project.')

parser.add_argument('--train', action='store_false', help='Train the model')
parser.add_argument('--stdout', type=str, help='File path to redirect stdout')
parser.add_argument('--stderr', type=str, help='File path to redirect stderr')
parser.add_argument('--disable-log', action='store_true', help='Disable logging to file')

args = parser.parse_args()

loggers = [] # logger to store iostream

if not args.disable_log:
    # set stdout
    if args.stdout:
        stdout_file = args.stdout
    else:
        stdout_file = 'stdout.log'  # 默认文件名
    
    # redirect stdout
    stdout_logger = StdoutTee(stdout_file, mode="w")
    loggers.append(stdout_logger)

    # set stderr
    if args.stderr:
        stderr_file = args.stderr
    else:
        stderr_file = 'stderr.log'  # 默认文件名

    # redirect stderr
    stderr_logger = StderrTee(stderr_file, mode="w")
    loggers.append(stderr_logger)

# main program
if __name__ == '__main__':
    if args.train:
        env = mo_gym.make('dt-eco-v0')
        # reset env and agent
        obs, _ = env.reset()
        agent = DT_ECO.create_agent(env)
        agent.train(
            total_timesteps=1000,
            eval_env=env,
        )

# 关闭所有日志文件
close_all()
