import gym
import pandas as pd
import gym_maze
import gym_multiplexer
import gym_corridor
import gc
import inspect
import os
from joblib import Parallel, delayed
from lcs.agents.acs2 import ACS2, Configuration as ACS2Config
from lcs.agents.acs2eder import ACS2EDER, Configuration as ACS2EDERConfig
from lcs.agents.acs2er import ACS2ER, Configuration as ACS2ERConfig
from lcs.agents.acs2her import ACS2HER, Configuration as ACS2HERConfig
from lcs.agents.acs2vcp import ACS2VCP, Configuration as ACS2VCPConfig
from lcs.metrics import population_metrics
from multiprocessing import Pool


class MxObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return [str(x) for x in observation]


def _maze_knowledge(agent, environment) -> float:
    transitions = environment.env.get_transitions()

    if hasattr(agent, 'ensemble_heads'):
        population = agent.ensemble_heads[0].population
    else:
        population = agent.population
    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    if not reliable_classifiers:
        return 0.0

    target_length = len(reliable_classifiers[0].condition)
    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for start, action, end in transitions:

        p0 = environment.env.maze.perception(start)
        p1 = environment.env.maze.perception(end)

        if len(p0) < target_length:
            hashes = ["#"] * (target_length - len(p0))
            p0 += hashes
            p1 += hashes

        if any([True for cl in reliable_classifiers
                if cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return nr_correct / len(transitions) * 100.0


def _maze_metrics(agent, env):
    pop = agent.ensemble_heads[0].population if hasattr(agent, 'ensemble_heads') else agent.population
    metrics = {
        'knowledge': _maze_knowledge(agent, env)
    }

    # Add basic population metrics
    metrics.update(population_metrics(pop, env))

    return metrics


def _mx_metrics(agent, env):
    pop = agent.ensemble_heads[0].population if hasattr(agent, 'ensemble_heads') else agent.population
    metrics = {}

    # Add basic population metrics
    metrics.update(population_metrics(pop, env))

    return metrics


def parse_metrics_to_df(explore_metrics, exploit_metrics, model_name, env_name, run_id, env_type='maze'):

    explore_df = pd.DataFrame(explore_metrics)
    exploit_df = pd.DataFrame(exploit_metrics)

    explore_df['phase'] = 'explore'
    exploit_df['phase'] = 'exploit'

    exploit_df['trial'] = exploit_df['trial'] + len(explore_df)

    df = pd.concat([explore_df, exploit_df], ignore_index=True)

    df['model'] = model_name
    df['env'] = env_name
    df['run_id'] = run_id

    if env_type == 'maze':
        df['steps'] = df['steps_in_trial']
        columns_to_keep = ['trial', 'steps_in_trial', 'reward', 'perf_time', 'knowledge', 'population', 'numerosity', 'reliable', 'phase', 'steps', 'model', 'env', 'run_id']
    elif env_type == 'mx':
        columns_to_keep = ['trial', 'reward', 'perf_time', 'population', 'numerosity', 'reliable', 'phase', 'model', 'env', 'run_id']
    else:
        raise ValueError(f"Nieobsługiwany typ środowiska: {env_type}")

    df = df[columns_to_keep]

    df.set_index('trial', inplace=True)

    return df


model_definitions = {
    'ACS2': (ACS2, ACS2Config, {}),
    'ACS2ER (s=8)': (ACS2ER, ACS2ERConfig, {'er_samples_number': 8}),
    'ACS2HER (s=8,k=2)': (ACS2HER, ACS2HERConfig,
                          {'er_samples_number': 8,
                           'her_goals_number': 2}),
    'ACS2VCP (s=8,k=2,h=4)': (ACS2VCP, ACS2VCPConfig,
                              {'er_samples_number': 8,
                               'her_goals_number': 2}),
    'ACS2EDER (s=3,b=2)': (ACS2EDER, ACS2EDERConfig,
                           {'eder_samples_number': 3,
                            'eder_subtrajectory_length': 2}),
    'ACS2EDER (s=5,b=2)': (ACS2EDER, ACS2EDERConfig,
                           {'eder_samples_number': 5,
                            'eder_subtrajectory_length': 2}),
    'ACS2EDER (s=3,b=4)': (ACS2EDER, ACS2EDERConfig,
                           {'eder_samples_number': 3,
                            'eder_subtrajectory_length': 4}),
    'ACS2EDER (s=5,b=4)': (ACS2EDER, ACS2EDERConfig,
                           {'eder_samples_number': 5,
                            'eder_subtrajectory_length': 4}),
}

def run_single_experiment(model_name, env_name, run_id):
    try:
        ModelClass, ConfigClass, base_extra_params = model_definitions[model_name]

        extra_params = base_extra_params.copy()

        explore_trials = 5000
        exploit_trials = 1000

        if 'multiplexer' in env_name:
            env = MxObservationWrapper(gym.make(env_name))
            c_length = env.observation_space.n
            possible_actions = 2
            metrics_fcn = _mx_metrics
            do_ga = True
            env_type_for_df = 'mx'
            min_samples = 100
        else:
            env = gym.make(env_name)
            c_length = 8
            possible_actions = 8
            metrics_fcn = _maze_metrics
            do_ga = False
            env_type_for_df = 'maze'
            min_samples = 1000

        if "HER" in model_name or "VCP" in model_name:
            c_length *= 2
            if 'multiplexer' and 'FrozenLake' not in env_name:
                explore_trials //= 10
                exploit_trials //= 10
                exploit_trials *= 2

        if model_name.startswith('ACS2ER'):
            extra_params['er_min_samples'] = min_samples
        elif model_name.startswith('ACS2EDER'):
            extra_params['eder_min_samples'] = min_samples

        cfg = ConfigClass(
            classifier_length=c_length,
            number_of_possible_actions=possible_actions,
            user_metrics_collector_fcn=metrics_fcn,
            do_ga=do_ga,
            **extra_params
        )

        # Explore
        agent = ModelClass(cfg)
        metrics_explore = agent.explore(env, explore_trials)
        # Exploit
        if hasattr(agent, 'ensemble_heads'):
            learned_population = agent.ensemble_heads[0].population
        else:
            learned_population = agent.population
        agent_exploit = ModelClass(cfg, population=learned_population)
        metrics_exploit = agent_exploit.exploit(env, exploit_trials)
        df = parse_metrics_to_df(metrics_explore, metrics_exploit, model_name,
                                 env_name, run_id, env_type=env_type_for_df)

        os.makedirs('results',
                    exist_ok=True)

        file_path = f"results/res_{model_name}_{env_name}_{run_id}.csv"
        df.to_csv(file_path, index=True)

        env.close()
        del agent, agent_exploit
        gc.collect()

        print(f"DONE: {model_name} | {env_name} | Run {run_id} "
              f"-> Saved to {file_path}")
        return True
    except Exception as e:
        import traceback
        import sys

        err_type, err_value, err_traceback = sys.exc_info()
        full_traceback = traceback.format_exc()

        print("\n" + "=" * 50)
        print(f"BŁĄD KRYTYCZNY: {model_name} | Run: {run_id}")
        print(f"Typ błędu: {err_type.__name__}")
        print(f"Komunikat: {err_value}")
        print("-" * 20)
        print("Ścieżka błędu (Traceback):")
        print(full_traceback)
        print("=" * 50 + "\n")
        print(f"ERROR in {model_name} | {run_id}: {e}")
        return None

if __name__ == '__main__':
    ENVIRONMENTS = [
        'Maze4-v0',
        'Maze5-v0',
        'Maze6-v0',
        'Maze7-v0',
        'Maze10-v0',
        'MazeA-v0',
        'MazeA1-v0',
        'MazeB-v0',
        'MazeD-v0',
        'MazeE1-v0',
        'MazeE2-v0',
        'MazeE3-v0',
        'MazeF1-v0',
        'MazeF2-v0',
        'MazeF3-v0',
        'MazeF4-v0',
        'MazeF8-v0',
        'MazeF9-v0',
        'MazeH1-v0',
        'MazeMA-v0',
        'MazeT2-v0',
        'MazeT3-v0',
        'MazeT4-v0',
        'Littman57-v0',
        'Littman89-v0',
        'Cassandra4x4-v0',
        'MiyazakiA-v0',
        'MiyazakiB-v0',
        'Woods1-v0',
        'Woods14-v0',
        'Woods100-v0',
        'Woods101-v0',
        'Woods101demi-v0',
        'Woods102-v0',
        'MazeH1-v0',
        'MazeMA-v0',
        # MPX
        'boolean-multiplexer-6bit-v0',
        'boolean-multiplexer-11bit-v0',
        # Corridor
        'corridor-40-v0',
        'corridor-100-v0',
    ]
    for environment in ENVIRONMENTS:
        env = gym.make(environment)
        env.save_grid(f"{environment}.pdf")

    RUNS_PER_MODEL = 10

    for environment in ENVIRONMENTS:
        tasks = [
            (m_name, environment, r_id)
            # for e_name in environment
            for m_name in model_definitions.keys()
            for r_id in range(RUNS_PER_MODEL)
        ]

        results = Parallel(n_jobs=4, backend="multiprocessing", verbose=51)(
            delayed(run_single_experiment)(m, e, r) for m, e, r in tasks
        )

        success_count = sum(1 for r in results if r is True)
        print(f"\nFinished! {success_count}/{len(tasks)} experiments saved "
              f"in 'results/' folder.")

