"""Three LOLA agents for Escape Room game."""

import os
import numpy as np
import tensorflow as tf

from . import logger

from .corrections import *
from .networks import *
from .utils import *


def update(mainQN, lr, tuple_final_delta_v):
    for idx, Q in enumerate(mainQN):
        _ = Q.setparams(
            Q.getparams() + lr * np.squeeze(tuple_final_delta_v[idx]))


def train(env, *, num_episodes, trace_length, batch_size, gamma,
          set_zero, lr, corrections, simple_net, hidden1, hidden2,
          mem_efficient=True, logdir=''):
    observation_space = env.NUM_STATES
    y = gamma
    load_model = False #Whether to load a saved model.
    n_agents = env.NUM_AGENTS
    total_n_agents = n_agents
    max_epLength = trace_length + 1 #The max allowed length of our episode.
    summaryLength = 100 #Number of episodes to periodically save for analysis

    tf.reset_default_graph()
    mainQN = []

    agent_list = np.arange(total_n_agents)
    for agent in range(total_n_agents):
        mainQN.append(Qnetwork_er('main' + str(agent), agent, env, lr=lr, gamma=gamma,
                                  batch_size=batch_size, trace_length=trace_length,
                                  hidden1=hidden1, hidden2=hidden2, simple_net=simple_net,
                                  num_states=env.NUM_STATES, num_actions=env.NUM_ACTIONS))
        
    if not mem_efficient:
        cube, cube_ops = make_cube(trace_length)
    else:
        cube, cube_ops = None, None

    corrections_func_3player(mainQN,
                             batch_size=batch_size,
                             trace_length=trace_length)

    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()

    buffers = []
    for i in range(total_n_agents):
        buffers.append(ExperienceBuffer(batch_size))

    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    aList = []

    total_steps = 0

    episodes_run = np.zeros(total_n_agents)
    episodes_run_counter =  np.zeros(total_n_agents)
    episodes_reward = np.zeros(total_n_agents)
    episodes_actions = np.zeros((total_n_agents, env.NUM_ACTIONS))
                # need to multiple with
    pow_series = np.arange(trace_length)
    discount = np.array([pow(gamma, item) for item in pow_series])
    discount_array = gamma**trace_length / discount
    # print('discount_array',discount_array.shape)
    discount = np.expand_dims(discount, 0)
    discount_array = np.reshape(discount_array,[1,-1])


    array = np.eye(env.NUM_STATES)
    feed_dict_log_pi = {mainQN[0].scalarInput: array,
                        mainQN[1].scalarInput: array,}

    if logdir != '':
        with open(os.path.join(logdir, 'log.csv'), 'w') as f:
            s = 'episode,'
            for agent_id in range(1, n_agents+1):
                s += 'A%d_reward_total,' % agent_id
            s += 'sum_reward\n'
            f.write(s)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Session(config=config_proto) as sess:
        sess.run(init)
        if cube_ops is not None:
            sess.run(cube_ops)

        # if set_zero == 1:
        #     for i in range(2):
        #         mainQN[i].setparams(np.zeros((5)))
        #         theta_2_vals =  mainQN[i].getparams()

        sP = env.reset()
        updated =True
        for i in range(num_episodes):
            episodeBuffer = []
            for ii in range(n_agents):
                episodeBuffer.append([])
            np.random.shuffle(agent_list)
            if n_agents  == total_n_agents:
                these_agents = range(n_agents)
            else:
                these_agents = sorted(agent_list[0:n_agents])

            # Reset environment and get first new observation
            sP = env.reset()
            s = sP
            state = []

            d = False
            rAll = np.zeros((n_agents))
            aAll = np.zeros((n_agents*env.NUM_ACTIONS))
            j = 0

            for agent in these_agents:
                episodes_run[agent] += 1
                episodes_run_counter[agent] += 1
            a_all_old = [0,0]

            # The Q-Network
            while j < max_epLength:
                j += 1
                a_all = []
                for agent_role, agent in enumerate(these_agents):
                    a = sess.run(
                        [mainQN[agent].predict],
                        feed_dict={
                            mainQN[agent].scalarInput: [s[agent_role]]
                        }
                    )
                    a_all.append(a[0])

                a_all_old = a_all
                # if a_all[0] > 1 or a_all[1] > 1:
                #     print('warning!!!', a_all, 's', s)
                s1P, r, d = env.step(a_all)
                s1 = s1P

                total_steps += 1
                for agent_role, agent in enumerate(these_agents):
                    episodeBuffer[agent_role].append([
                        s[agent_role], a_all[agent_role], r[agent_role], s1[agent_role],
                        d, these_agents[agent_role]
                    ])
                    episodes_reward[agent] += r[agent_role]
                rAll += [r[ii]*gamma**(j-1) for ii in range(n_agents)]

                for agent_id in range(n_agents):
                    aAll[a_all[agent_id] + agent_id*env.NUM_ACTIONS] += 1
                s_old = s
                s = s1
                sP = s1P
                if d == True:
                    break

            # Add the episode to the experience buffer
            for agent_role, agent in enumerate(these_agents):
                buffers[agent].add(np.array(episodeBuffer[agent_role]))

            jList.append(j)
            rList.append(rAll)
            aList.append(aAll)

            if (episodes_run[agent] % batch_size == 0 and
                episodes_run[agent] > 0):

                trainBatch = []
                sample_return = []
                sample_reward = []
                last_state = []
                
                for agent_id in range(n_agents):
                    trainBatch.append(buffers[agent_id].sample(batch_size, trace_length))
                    sample_return.append(
                        np.reshape(get_monte_carlo(trainBatch[agent_id][:,2], y, trace_length, batch_size), [batch_size, -1]))
                    sample_reward.append(
                        np.reshape(trainBatch[agent_id][:,2]- np.mean(trainBatch[agent_id][:,2]), [-1, trace_length]) * discount)
                    last_state.append(
                        np.reshape(np.vstack(trainBatch[agent_id][:,3]), [-1, trace_length, env.NUM_STATES])[:,-1,:])

                fetches = [mainQN[idx].value for idx in range(n_agents)]
                feed = {}
                for idx in range(n_agents):
                    feed[mainQN[idx].scalarInput] = last_state[idx]
                value_next = sess.run(fetches, feed_dict=feed)

                fetches = [mainQN[idx].updateModel for idx in range(n_agents)]
                fetches += [mainQN[idx].delta for idx in range(n_agents)]
                feed_dict = {}
                for idx in range(n_agents):
                    feed_dict[mainQN[idx].scalarInput] = np.vstack(trainBatch[idx][:,0])
                    feed_dict[mainQN[idx].sample_return] = sample_return[idx]
                    feed_dict[mainQN[idx].actions] = trainBatch[idx][:,1]
                    feed_dict[mainQN[idx].sample_reward] = sample_reward[idx]
                    feed_dict[mainQN[idx].next_value] = value_next[idx]
                    feed_dict[mainQN[idx].gamma_array] = discount
                    feed_dict[mainQN[idx].gamma_array_inverse] = discount_array

                if episodes_run[agent] % batch_size == 0 and episodes_run[agent] > 0:
                    # only need the last <n_agents> objects in the tuple
                    tuple_blank_and_updates = sess.run(fetches, feed_dict=feed_dict)
                    # _, _, update1, update2 = sess.run(fetches, feed_dict=feed_dict)

                if episodes_run[agent] % batch_size == 0  and episodes_run[agent] > 0:
                    # update(mainQN, lr, update1, update2)
                    update(mainQN, lr, tuple_blank_and_updates[n_agents:])
                    updated =True
                    # print('update params')
                episodes_run_counter[agent] = episodes_run_counter[agent] *0
                episodes_actions[agent] = episodes_actions[agent]*0
                episodes_reward[agent] =episodes_reward[agent] *0

            if len(rList) % summaryLength == 0 and len(rList) != 0 and updated == True:
                # print('summarizing at episode', i)
                updated = False
                gamma_discount = 1 / (1-gamma)
                print(total_steps,'reward', np.mean(rList[-summaryLength:], 0)/gamma_discount, 'action', (np.mean(aList[-summaryLength:], 0)*2.0/ np.sum(np.mean(aList[-summaryLength:], 0)))*100//1)

                action_prob = np.mean(aList[-summaryLength:], 0)*2.0/ np.sum(np.mean(aList[-summaryLength:], 0))
                log_items = {}
                for idx_agent in range(n_agents):
                    log_items['reward_agent%d'%idx_agent] = np.mean(rList[-summaryLength:], 0)[idx_agent]

                for key in sorted(log_items.keys()):
                    logger.record_tabular(key, log_items[key])
                logger.dump_tabular()
                logger.info('')

                if logdir != '':
                    with open(os.path.join(logdir, 'log.csv'), 'a') as f:
                        s = '%d,' % (i+1)
                        sum_reward = 0
                        for idx_agent in range(n_agents):
                            temp = log_items['reward_agent%d'%idx_agent]
                            s += '%.3e,' % temp
                            sum_reward += temp
                        s += '%.3e\n' % sum_reward
                        f.write(s)
