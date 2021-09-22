import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from actor import ActorNetwork
from actor_regression import ActorNetworkRegression, StateScaler
from critic import CriticNetwork
import os
import time
import operator

start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILES_PATH = Path('./saved_models_tl/')
if not os.path.exists(str(FILES_PATH)):
    os.makedirs(str(FILES_PATH))

np.random.seed(1)

kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)

game_list = []  # 'MountainCarContinuous-v0','Acrobot-v1','CartPole-v1'
games_to_load_from = []

for game in game_list:
    if game == 'MountainCarContinuous-v0':
        game_action_size = 1
    elif game == 'CartPole-v1':
        game_action_size = 2
    elif game == 'Acrobot-v1':
        game_action_size = 3

    env = gym.make(game)
    state_size = 6
    action_size = 3
    max_steps = 500
    max_episodes = 5000
    discount_factor = 0.99
    lr_actor = 0.0002
    lr_critic = 0.001
    render = False

    save_file = FILES_PATH / str(game)
    req_file = FILES_PATH / str(games_to_load_from[0] + f"_critic_h1_bias_section2.npy")

    if not req_file.exists():
        print("Not found saved model. Exiting!")
        exit()

    # Initialize the policy network
    tf.compat.v1.reset_default_graph()
    tf.get_logger().setLevel('INFO')

    sess = tf.Session()

    if game == 'MountainCarContinuous-v0':
        scaler = StateScaler(env)
        actor = ActorNetworkRegression(state_size, action_size, lr_actor, scaler, sess, game, games_to_load_from)
    else:
        actor = ActorNetwork(state_size, action_size, lr_actor, sess, game, games_to_load_from)

    critic = CriticNetwork(state_size, lr_critic, sess, game, games_to_load_from)
    sess.run(tf.global_variables_initializer())

    rewards = []
    rewards_avg = []
    actor_loss = []
    critic_loss = []

    with sess:
        solved = False
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()
            state = np.pad(state, (0, state_size - len(state)), mode='constant')  # pad with 0
            state = state.reshape([1, state_size])

            for step in range(max_steps):

                if game == 'MountainCarContinuous-v0':
                    state = actor.scaler.scale(state)
                    action = sess.run(actor.action_clapped, {actor.state: state})
                else:
                    actions_distribution = sess.run(actor.actions_distribution, {actor.state: state})
                    action = np.inf
                    while action > game_action_size - 1:
                        dict_actions_freq = {}
                        for choice in actions_distribution:
                            chosen_action = np.random.choice(np.arange(len(choice)), p=choice)
                            action_label = str(chosen_action)
                            if action_label in dict_actions_freq.keys():
                                dict_actions_freq[action_label] += 1
                            else:
                                dict_actions_freq[action_label] = 1
                        chosen_action = max(dict_actions_freq.items(), key=operator.itemgetter(1))[0]  # returns str
                        action = int(chosen_action)

                next_state, reward, done, _ = env.step(action)

                next_state = next_state.reshape([len(next_state)])
                next_state = np.pad(next_state, (0, state_size - len(next_state)), mode='constant')  # pad with 0

                next_state = next_state.reshape([1, state_size])

                if game == 'MountainCarContinuous-v0':
                    next_state = actor.scaler.scale(next_state)

                if render:
                    env.render()

                episode_rewards[episode] += reward

                td_value = reward

                if not done:
                    td_value += discount_factor * sess.run(critic.output, {critic.state: next_state})

                td_error = td_value - sess.run(critic.output, {critic.state: state})

                feed_dict = {critic.state: state, critic.R_t: td_value}
                _, loss = sess.run([critic.optimizer, critic.loss], feed_dict)
                critic_loss.append(loss)

                if game == 'MountainCarContinuous-v0':
                    feed_dict = {actor.action: action, actor.state: state, actor.R_t: td_error}
                    _, loss = sess.run([actor.optimizer, actor.loss], feed_dict)
                    actor_loss.append(loss.reshape(-1, ))
                else:
                    action_one_hot = np.zeros(action_size)
                    action_one_hot[action] = 1
                    feed_dict = {actor.state: state, actor.R_t: td_error, actor.action: action_one_hot}
                    _, loss = sess.run([actor.optimizer, actor.loss], feed_dict)
                    actor_loss.append(loss)

                if done:
                    rewards.append(episode_rewards[episode])

                    if episode <= 98:
                        print("Episode {} Reward: {}".format(episode, episode_rewards[episode]))

                    else:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                        rewards_avg.append(average_rewards)
                        print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode,
                                                                                           episode_rewards[episode],
                                                                                           round(average_rewards, 2)))
                        if game == 'CartPole-v1' and average_rewards >= 475:
                            print(' Solved at episode: ' + str(episode))
                            solved = True

                        elif game == 'Acrobot-v1' and average_rewards >= (-100):
                            print(' Solved at episode: ' + str(episode))
                            solved = True

                        elif game == 'MountainCarContinuous-v0' and average_rewards >= 90:
                            print(' Solved at episode: ' + str(episode))
                            solved = True

                    break
                state = next_state

            if solved:
                critic.save_critic_model()
                actor.save_actor_model()
                break

    print("--- %s seconds ---" % (time.time() - start_time))

    indexes_of_right_losses = []

    # fix critic shape by deleting
    critic_loss_fixed = []

    for idx, loss in enumerate(critic_loss):
        shape = loss.shape
        if len(shape) == 3 and shape[1] == 1 and shape[2] == 2:  # shape (None,1,2)
            continue
        else:  # the right loss to save
            loss_lst = list(loss)
            indexes_of_right_losses.append(idx)
            critic_loss_fixed.append(
                loss_lst[len(loss_lst) - 1])  # shape is (None,), cuz of the architecture we need the last val.

    # fix actor shape by deleting
    actor_loss_fixed = []
    for idx in indexes_of_right_losses:
        loss = actor_loss[idx]
        loss_lst = list(loss)
        actor_loss_fixed.append(
            loss_lst[len(loss_lst) - 1])  # shape is (None,), cuz of the architecture we need the last val.

    critic_loss = critic_loss_fixed
    actor_loss = actor_loss_fixed

    plt.rcParams["figure.figsize"] = (16, 5)
    plt.plot(rewards, zorder=1, label='Rewards')
    plt.plot(list(range(99, len(rewards_avg) + 99)), rewards_avg, zorder=2, label='Mean over 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    plt.legend()
    plt.savefig(f"{game}_rewards_ac_{len(rewards)}.png")
    plt.show()

    plt.plot(critic_loss, zorder=1, label='Critic loss')
    plt.plot(actor_loss, zorder=2, label='Actor loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Actor and Critic Losses')
    plt.legend()
    plt.savefig(f"{game}_loss_ac_{len(rewards)}.png")
    plt.show()
