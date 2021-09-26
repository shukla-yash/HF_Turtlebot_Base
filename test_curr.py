import os

import gym
import numpy as np

from SimpleDQN import SimpleDQN

# not technically needed here but it'll fail later if it's not available, so keeping it
import TurtleBot_v0


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
    # TODO
    def __init__(self):
        pass


class CurriculumRunner(object):
    def __init__(self, agent: CurriculumAgent, random_seed=1):
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

        self.actionCnt = 5
        self.D = 83  # 90 beams x 4 items lidar + 3 inventory items
        self.NUM_HIDDEN = 16
        self.GAMMA = 0.995
        self.LEARNING_RATE = 1e-3
        self.DECAY_RATE = 0.99
        self.MAX_EPSILON = 0.1
        self.random_seed = random_seed

        # agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
        # agent.set_explore_epsilon(MAX_EPSILON)
        self.total_episodes_arr = []

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

            if i == 0:
                agent = SimpleDQN(
                    self.actionCnt,
                    self.D,
                    self.NUM_HIDDEN,
                    self.LEARNING_RATE,
                    self.GAMMA,
                    self.DECAY_RATE,
                    self.MAX_EPSILON,
                    self.random_seed,
                )
                agent.set_explore_epsilon(self.MAX_EPSILON)
            else:
                agent = SimpleDQN(
                    self.actionCnt,
                    self.D,
                    self.NUM_HIDDEN,
                    self.LEARNING_RATE,
                    self.GAMMA,
                    self.DECAY_RATE,
                    self.MAX_EPSILON,
                    self.random_seed,
                )
                agent.set_explore_epsilon(self.MAX_EPSILON)
                agent.load_model(0, 0, i - 1)
                agent.reset()
                print("loaded model")

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
                a = agent.process_step(obs, True)

                _, reward, done, _ = env.step(a)

                # give reward
                agent.give_reward(reward)
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
                    agent.finish_episode()

                    # update after every episode
                    if episode % 10 == 0:
                        agent.update_parameters()

                    # reset environment
                    episode += 1

                    env.reset()
                    reward_sum = 0

                    env_flag = 0
                    if i < 3:
                        env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)

                    # quit after some number of episodes
                    if episode > 70000 or env_flag == 1:

                        agent.save_model(0, 0, i)
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
    cr = CurriculumRunner(CurriculumAgent())
    cr.run()


if __name__ == "__main__":
    main()
