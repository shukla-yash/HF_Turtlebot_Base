import os

import gym
import numpy as np
from SimpleDQN import SimpleDQN

from dqnlambda import dqn as LambdaDQN
from dqnlambda.utils import minimize_with_grad_clipping
from dqnlambda.wrappers import HistoryWrapper
from dqnlambda.replay_memory import make_replay_memory

import argparse
import tensorflow as tf
from tensorflow.python.layers.layers import conv2d, dense, flatten

# not technically needed here but it'll fail later if it's not available, so keeping it
import TurtleBot_v0

from abc import abstractmethod


def CheckTrainingDoneCallback(reward_array, done_array, env):

    done_cond = False
    reward_cond = False
    if len(done_array) > 40:
        if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
            if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
                done_cond = True

        if done_cond == True:
            if env < 3:
                if np.mean(reward_array[-40:]) > 730:
                    reward_cond = True
            # else:
            # 	if np.mean(reward_array[-10:]) > 950:
            # 		reward_cond = True

        if done_cond == True and reward_cond == True:
            return 1
        else:
            return 0
    else:
        return 0


class CurriculumAgent(object):
    def __init__(self, seed: int):
        self.MAX_EPSILON = 0.1

        self.seed = seed
        print("Current seed: " + str(self.seed))
        self.init()

    @abstractmethod
    def init(self):
        raise NotImplementedError

    @abstractmethod
    def agent_init(self):
        raise NotImplementedError

    @abstractmethod
    def agent_load(self, curriculum_no, beam_no, env_no):
        raise NotImplementedError

    @abstractmethod
    def process_step(self, obs):
        raise NotImplementedError

    @abstractmethod
    def give_reward(self, reward, action=None):
        raise NotImplementedError

    @abstractmethod
    def finish_episode(self):
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, curriculum_no, beam_no, env_no):
        raise NotImplementedError

    def set_env(self, env):
        self.env = env

class SimpleDQN_CurriculumAgent(CurriculumAgent):
    def init(self):
        self.actionCnt = 5
        self.D = 83  # 90 beams x 4 items lidar + 3 inventory items
        self.NUM_HIDDEN = 16
        self.GAMMA = 0.995
        self.LEARNING_RATE = 1e-3
        self.DECAY_RATE = 0.99

        self.agent = None

    def agent_init(self):
        self.agent = SimpleDQN(
            self.actionCnt,
            self.D,
            self.NUM_HIDDEN,
            self.LEARNING_RATE,
            self.GAMMA,
            self.DECAY_RATE,
            self.MAX_EPSILON,
            self.seed,
        )
        self.agent.set_explore_epsilon(self.MAX_EPSILON)

    def agent_load(self, curriculum_no, beam_no, env_no):
        if self.agent is not None:
            self.agent.load_model(curriculum_no, beam_no, env_no)
            self.agent.reset()

    def process_step(self, obs):
        if self.agent is not None:
            return self.agent.process_step(obs, True)

    def give_reward(self, reward, action=None):
        del action
        if self.agent is not None:
            self.agent.give_reward(reward)

    def finish_episode(self):
        if self.agent is not None:
            self.agent.finish_episode()

    def update_parameters(self):
        if self.agent is not None:
            self.agent.update_parameters()

    def save_model(self, curriculum_no, beam_no, env_no):
        if self.agent is not None:
            self.agent.save_model(curriculum_no, beam_no, env_no)


class DQNLambda_CurriculumAgent(CurriculumAgent):
    def init(self):
        self.session = tf.Session()
        self.done = False

        self.return_est='nstep-1'
        self.mem_size=1e6
        self.history_len=4
        self.discount=0.99
        self.cache_size=80e3
        self.block_size=100
        self.priority=0.0

    def save_model(self, curriculum_no, beam_no, env_no):
        log_dir = "results"
        env_id = "NovelGridworld-v0"
        saver = tf.train.Saver()

        experiment_file_name = (
            "_c" + str(curriculum_no) + "_b" + str(beam_no) + "_e" + str(env_no)
        )
        path_to_save = log_dir + os.sep + env_id + experiment_file_name

        saver.save(self.session, path_to_save)

    def agent_load(self, curriculum_no, beam_no, env_no):
        log_dir = "results"
        env_id = "NovelGridworld-v0"
        experiment_file_name = (
            "_c" + str(curriculum_no) + "_b" + str(beam_no) + "_e" + str(env_no)
        )
        filename = log_dir + os.sep + env_id + experiment_file_name
        saver = tf.train.import_meta_graph(filename)
        if saver is not None:
            saver.restore(self.session, log_dir + os.sep)

    def agent_init(self):
        grad_clip = None # defaults to None in the impl, but this feels wrong...
        self.n_actions = self.env.action_space.n
        self.replay_memory = make_replay_memory(self.return_est, self.mem_size, self.history_len, self.discount,
                                           self.cache_size, self.block_size, self.priority)
        input_shape = (self.replay_memory.history_len, *self.env.observation_space.shape)
        n_actions = self.env.action_space.n
        self.benchmark_env = HistoryWrapper(self.env, self.replay_memory.history_len)

        # Build TensorFlow model
        self.state_ph  = tf.placeholder(self.env.observation_space.dtype, [None] + list(input_shape))
        self.action_ph = tf.placeholder(tf.int32, [None])
        self.return_ph = tf.placeholder(tf.float32, [None])

        qvalues = self.q_function(self.state_ph, n_actions, scope='main')

        self.greedy_actions = tf.argmax(qvalues, axis=1)
        self.greedy_qvalues = tf.reduce_max(qvalues, axis=1)

        action_indices = tf.stack([tf.range(tf.size(self.action_ph)), self.action_ph], axis=-1)
        self.onpolicy_qvalues = tf.gather_nd(qvalues, action_indices)

        td_error = self.return_ph - self.onpolicy_qvalues #type: ignore
        loss = tf.reduce_mean(tf.square(td_error))

        main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)
        self.train_op = minimize_with_grad_clipping(optimizer, loss, main_vars, grad_clip)

        self.replay_memory.register_refresh_func(self.refresh)

        self.session.run(tf.global_variables_initializer())

    def refresh(self, states, actions):
        assert len(states) == len(actions) + 1  # We should have an extra bootstrap state
        assert (self.session is not None)
        greedy_qvals, greedy_acts, onpolicy_qvals = self.session.run([self.greedy_qvalues, self.greedy_actions, self.onpolicy_qvalues], feed_dict={
            self.state_ph: states,
            self.action_ph: actions,
            }) #type: ignore
        mask = (actions == greedy_acts[:-1])
        return greedy_qvals, mask, onpolicy_qvals

    def process_step(self, obs):
        self.replay_memory.store_obs(obs)
        state = self.replay_memory.encode_recent_observation()

        return self.epsilon_greedy(state, self.MAX_EPSILON)

    def give_reward(self, reward, action):
        self.replay_memory.store_effect(action, reward, self.done)

    def finish_episode(self):
        # my understanding of dqn-lambda implementation is that there's no need
        # to make use of this clean-up step, since the same accumuation is
        # happening at the end of each step. But I might be totally wrong here.
        self.done = True

    def update_parameters(self):
        pass

    def q_function(self, state, n_actions, scope):
        # TODO- stand-in from atari env, probably needs to be customized a bit
        hidden = tf.cast(state, tf.float32) / 255.0  # type: ignore
        hidden = tf.unstack(hidden, axis=1)
        hidden = tf.concat(hidden, axis=-1)

        # not sure what's going on here tbh --cst

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            hidden = conv2d(
                hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu
            )
            hidden = conv2d(
                hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu
            )
            hidden = conv2d(
                hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu
            )

            hidden = flatten(hidden)

            hidden = dense(hidden, units=512, activation=tf.nn.relu)
            qvalues = dense(hidden, units=n_actions, activation=None)

        return qvalues

    def epsilon_greedy(self, state, epsilon):
        if np.random.random() < epsilon:  # type: ignore
            action = np.random.choice(range(0, self.n_actions)) # type: ignore
        else:
            qvalues = self.q_function(self.state_ph, self.n_actions, scope="main")
            greedy_actions = tf.argmax(qvalues, axis=1)
            assert self.session is not None
            action = self.session.run(greedy_actions, feed_dict={self.state_ph: state[None]})[0] # type: ignore
        return action


class CurriculumRunner(object):
    def __init__(self, curriculum_agent: CurriculumAgent, random_seed=1):
        self.no_of_environments = 4

        self.width_array = [1.5, 2.5, 3, 3]
        self.height_array = [1.5, 2.5, 3, 3]
        self.no_trees_array = [1, 1, 3, 4]
        self.no_rocks_array = [0, 1, 2, 2]
        self.crafting_table_array = [0, 0, 1, 1]
        self.starting_trees_array = [0, 0, 0, 0]
        self.starting_rocks_array = [0, 0, 0, 0]
        self.type_of_env_array = [0, 1, 2, 2]

        self.total_timesteps_array = []
        self.total_reward_array = []
        self.avg_reward_array = []
        self.task_completion_array = []
        self.random_seed = random_seed

        self.total_episodes_arr = []

        self.curriculum_agent = curriculum_agent

    def run(self):
        for i in range(self.no_of_environments):
            print("Environment: ", i)

            width = self.width_array[i]
            height = self.height_array[i]
            no_trees = self.no_trees_array[i]
            no_rocks = self.no_rocks_array[i]
            crafting_table = self.crafting_table_array[i]
            starting_trees = self.starting_trees_array[i]
            starting_rocks = self.starting_rocks_array[i]
            type_of_env = self.type_of_env_array[i]

            final_status = False

            if i == self.no_of_environments - 1:
                final_status = True

            env_id = "TurtleBot-v0"
            env = gym.make(
                env_id,
                map_width=width,
                map_height=height,
                items_quantity={
                    "tree": no_trees,
                    "rock": no_rocks,
                    "crafting_table": crafting_table,
                    "stone_axe": 0,
                },
                initial_inventory={
                    "wall": 0,
                    "tree": starting_trees,
                    "rock": starting_rocks,
                    "crafting_table": 0,
                    "stone_axe": 0,
                },
                goal_env=type_of_env,
                is_final=final_status,
            )

            self.curriculum_agent.set_env(env)
            self.curriculum_agent.agent_init()
            if i != 0:
                self.curriculum_agent.agent_load(0, 0, i - 1)
                print("loaded model")


            t_step = 0
            episode = 0
            t_limit = 600
            reward_sum = 0
            reward_arr = []
            avg_reward = []
            done_arr = []
            env_flag = 0

            env.reset()

            while True:

                # get obseration from sensor
                obs = env.get_observation()

                # act
                a = self.curriculum_agent.process_step(obs)

                _, reward, done, _ = env.step(a)

                # give reward
                self.curriculum_agent.give_reward(reward, a)
                reward_sum += reward

                t_step += 1

                if t_step > t_limit or done == True:

                    # finish agent
                    if done == True:
                        done_arr.append(1)
                        self.task_completion_array.append(1)
                    elif t_step > t_limit:
                        done_arr.append(0)
                        self.task_completion_array.append(0)

                    print(
                        "\n\nfinished episode = "
                        + str(episode)
                        + " with "
                        + str(reward_sum)
                        + "\n"
                    )

                    reward_arr.append(reward_sum)
                    avg_reward.append(np.mean(reward_arr[-40:]))

                    self.total_reward_array.append(reward_sum)
                    self.avg_reward_array.append(np.mean(reward_arr[-40:]))
                    self.total_timesteps_array.append(t_step)

                    done = True
                    t_step = 0
                    self.curriculum_agent.finish_episode()

                    # update after every episode
                    if episode % 10 == 0:
                        self.curriculum_agent.update_parameters()

                    # reset environment
                    episode += 1

                    env.reset()
                    reward_sum = 0

                    env_flag = 0
                    if i < 3:
                        env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)

                    # quit after some number of episodes
                    if episode > 70000 or env_flag == 1:
                        self.curriculum_agent.save_model(0, 0, i)
                        self.total_episodes_arr.append(episode)

                        break

        print("Total epsiode array is: ", self.total_episodes_arr)

        log_dir = "logs_" + str(self.random_seed)
        os.makedirs(log_dir, exist_ok=True)

        total_timesteps_array = np.asarray(total_timesteps_array)  # type: ignore (linter being picky about object membership)
        print("size total_timesteps_array: ", total_timesteps_array.shape)

        total_reward_array = np.asarray(total_reward_array)  # type: ignore (linter being picky about object membership)
        print("size total_reward_array: ", total_reward_array.shape)

        avg_reward_array = np.asarray(avg_reward_array)  # type: ignore (linter being picky about object membership)
        print("size avg_reward_array: ", avg_reward_array.shape)

        self.total_episodes_arr = np.asarray(self.total_episodes_arr)  # type: ignore (linter being picky about object membership)
        print("size total_episodes_arr: ", self.total_episodes_arr.shape)

        task_completion_arr = np.asarray(task_completion_array)  # type: ignore (linter being picky about object membership)

        # final_timesteps_array = np.asarray(final_timesteps_array)
        # print("size final_timesteps_array: ", final_timesteps_array.shape)

        # final_reward_array = np.asarray(final_reward_array)
        # print("size final_reward_array: ", final_reward_array.shape)

        # final_avg_reward_array = np.asarray(final_avg_reward_array)
        # print("size final_avg_reward_array: ", final_avg_reward_array.shape)

        experiment_file_name_total_timesteps = (
            "randomseed_" + str(self.random_seed) + "_total_timesteps"
        )
        path_to_save_total_timesteps = (
            log_dir + os.sep + experiment_file_name_total_timesteps + ".npz"
        )

        experiment_file_name_total_reward = (
            "randomseed_" + str(self.random_seed) + "_total_reward"
        )
        path_to_save_total_reward = (
            log_dir + os.sep + experiment_file_name_total_reward + ".npz"
        )

        experiment_file_name_avg_reward = (
            "randomseed_" + str(self.random_seed) + "_avg_reward"
        )
        path_to_save_avg_reward = (
            log_dir + os.sep + experiment_file_name_avg_reward + ".npz"
        )

        experiment_file_name_total_episodes = (
            "randomseed_" + str(self.random_seed) + "_total_episodes"
        )
        path_to_save_total_episodes = (
            log_dir + os.sep + experiment_file_name_total_episodes + ".npz"
        )

        experiment_file_name_task_completion = (
            "randomseed_" + str(self.random_seed) + "_task_completion_curr"
        )
        path_to_save_task_completion = (
            log_dir + os.sep + experiment_file_name_task_completion + ".npz"
        )

        # experiment_file_name_final_timesteps = 'randomseed_' + str(random_seed) + '_final_timesteps'
        # path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'

        # experiment_file_name_final_reward = 'randomseed_' + str(random_seed) + '_final_reward'
        # path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'

        # experiment_file_name_final_avg_reward = 'randomseed_' + str(random_seed) + '_final_avg_reward'
        # path_to_save_final_avg_reward = log_dir + os.sep + experiment_file_name_final_avg_reward + '.npz'

        np.savez_compressed(
            path_to_save_total_timesteps, curriculum_timesteps=total_timesteps_array
        )
        # np.delete(total_timesteps_array)

        np.savez_compressed(
            path_to_save_total_reward, curriculum_reward=total_reward_array
        )
        # np.delete(total_reward_array)

        np.savez_compressed(
            path_to_save_avg_reward, curriculum_avg_reward=avg_reward_array
        )
        # np.delete(avg_reward_array)

        np.savez_compressed(
            path_to_save_total_episodes, curriculum_episodes=self.total_episodes_arr
        )
        # np.delete(total_episodes_arr)

        np.savez_compressed(
            path_to_save_task_completion, task_completion_curr=task_completion_arr
        )

        # np.savez_compressed(path_to_save_final_timesteps, final_timesteps = final_timesteps_array)
        # # np.delete(final_timesteps_array)

        # np.savez_compressed(path_to_save_final_reward, final_reward = final_reward_array)
        # # final_reward_array.cler()

        # np.savez_compressed(path_to_save_final_avg_reward, final_avg_reward = final_avg_reward_array)
        # # np.delete(final_avg_reward_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices={"simpledqn", "dqnlambda"}, default="simpledqn"
    )
    args = parser.parse_args()
    cr = None
    if args.mode == "simpledqn":
        cr = CurriculumRunner(SimpleDQN_CurriculumAgent(1))
    if args.mode == "dqnlambda":
        cr = CurriculumRunner(DQNLambda_CurriculumAgent(1))

    if cr is not None and issubclass(CurriculumRunner, cr.__class__):
        cr.run()


if __name__ == "__main__":
    main()
