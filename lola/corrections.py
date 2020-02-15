"""
The magic corrections of LOLA.
"""
import tensorflow as tf

from .utils import flatgrad


def corrections_func(mainPN, batch_size, trace_length,
                     corrections=False, cube=None):
    """Computes corrections for policy gradients.

    Args:
    -----
        mainPN: list of policy/Q-networks
        batch_size: int
        trace_length: int
        corrections: bool (default: False)
            Whether policy networks should use corrections.
        cube: tf.Varialbe or None (default: None)
            If provided, should be constructed via `lola.utils.make_cube`.
            Used for variance reduction of the value estimation.
            When provided, the computation graph for corrections is faster to
            compile but is quite memory inefficient.
            When None, variance reduction graph is contructed dynamically,
            is a little longer to compile, but has lower memory footprint.
    """
    if cube is not None:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, 1, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length, 1])
        mat_1 = tf.reshape(tf.squeeze(tf.matmul(ac_logp1, ac_logp0)),
                           [batch_size, 1, trace_length * trace_length])

        v_0 = tf.matmul(tf.reshape(mainPN[0].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_0 = tf.reshape(v_0, [batch_size, trace_length, trace_length, trace_length])

        v_1 = tf.matmul(tf.reshape(mainPN[1].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_1 = tf.reshape(v_1, [batch_size, trace_length, trace_length, trace_length])

        v_0 = 2 * tf.reduce_sum(v_0 * cube) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1 * cube) / batch_size
    else:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length])

        # Static exclusive cumsum
        ac_logp0_cumsum = [tf.constant(0.)]
        ac_logp1_cumsum = [tf.constant(0.)]
        for i in range(trace_length - 1):
            ac_logp0_cumsum.append(tf.add(ac_logp0_cumsum[-1], ac_logp0[:, i]))
            ac_logp1_cumsum.append(tf.add(ac_logp1_cumsum[-1], ac_logp1[:, i]))

        # Compute v_0 and v_1
        mat_cumsum = ac_logp0[:, 0] * ac_logp1[:, 0]
        v_0 = mat_cumsum * mainPN[0].sample_reward[:, 0]
        v_1 = mat_cumsum * mainPN[1].sample_reward[:, 0]
        for i in range(1, trace_length):
            mat_cumsum = tf.add(mat_cumsum, ac_logp0[:, i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp0_cumsum[i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp1_cumsum[i] * ac_logp0[:, i])
            v_0 = tf.add(v_0, mat_cumsum * mainPN[0].sample_reward[:, i])
            v_1 = tf.add(v_1, mat_cumsum * mainPN[1].sample_reward[:, i])
        v_0 = 2 * tf.reduce_sum(v_0) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1) / batch_size

    v_0_pi_0 = 2*tf.reduce_sum(((mainPN[0].target-tf.stop_gradient(mainPN[0].value)) * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_0_pi_1 = 2*tf.reduce_sum(((mainPN[0].target-tf.stop_gradient(mainPN[0].value)) * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    v_1_pi_0 = 2*tf.reduce_sum(((mainPN[1].target-tf.stop_gradient(mainPN[1].value)) * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_1_pi_1 = 2*tf.reduce_sum(((mainPN[1].target-tf.stop_gradient(mainPN[1].value)) * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    v_0_grad_theta_0 = flatgrad(v_0_pi_0, mainPN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0_pi_1, mainPN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1_pi_0, mainPN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1_pi_1, mainPN[1].parameters)

    mainPN[0].grad = v_0_grad_theta_0
    mainPN[1].grad = v_1_grad_theta_1

    mainPN[0].grad_v_1 = v_1_grad_theta_0
    mainPN[1].grad_v_0 = v_0_grad_theta_1

    if corrections:
        v_0_grad_theta_0_wrong = flatgrad(v_0, mainPN[0].parameters)
        v_1_grad_theta_1_wrong = flatgrad(v_1, mainPN[1].parameters)

        param_len = v_0_grad_theta_0_wrong.get_shape()[0].value

        multiply0 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
            tf.reshape(v_1_grad_theta_1_wrong, [param_len, 1])
        )
        multiply1 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_1_grad_theta_0), [1, param_len]),
            tf.reshape(v_0_grad_theta_0_wrong, [param_len, 1])
        )

        second_order0 = flatgrad(multiply0, mainPN[0].parameters)
        second_order1 = flatgrad(multiply1, mainPN[1].parameters)

        mainPN[0].v_0_grad_01 = second_order0
        mainPN[1].v_1_grad_10 = second_order1

        mainPN[0].delta = v_0_grad_theta_0 + second_order0
        mainPN[1].delta = v_1_grad_theta_1 + second_order1
    else:
        mainPN[0].delta = v_0_grad_theta_0
        mainPN[1].delta = v_1_grad_theta_1


def corrections_func_lola_pg(mainPN, batch_size, trace_length,
                             cube=None):
    """Computes corrections for policy gradients.
    Agent 0 is LOLA, Agent 1 is policy gradient.

    Args:
    -----
        mainPN: list of policy/Q-networks
        batch_size: int
        trace_length: int
        cube: tf.Varialbe or None (default: None)
            If provided, should be constructed via `lola.utils.make_cube`.
            Used for variance reduction of the value estimation.
            When provided, the computation graph for corrections is faster to
            compile but is quite memory inefficient.
            When None, variance reduction graph is contructed dynamically,
            is a little longer to compile, but has lower memory footprint.
    """
    if cube is not None:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, 1, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length, 1])
        mat_1 = tf.reshape(tf.squeeze(tf.matmul(ac_logp1, ac_logp0)),
                           [batch_size, 1, trace_length * trace_length])

        v_0 = tf.matmul(tf.reshape(mainPN[0].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_0 = tf.reshape(v_0, [batch_size, trace_length, trace_length, trace_length])

        v_1 = tf.matmul(tf.reshape(mainPN[1].sample_reward, [batch_size, trace_length, 1]), mat_1)
        v_1 = tf.reshape(v_1, [batch_size, trace_length, trace_length, trace_length])

        v_0 = 2 * tf.reduce_sum(v_0 * cube) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1 * cube) / batch_size
    else:
        ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                              [batch_size, trace_length])
        ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                              [batch_size, trace_length])

        # Static exclusive cumsum
        ac_logp0_cumsum = [tf.constant(0.)]
        ac_logp1_cumsum = [tf.constant(0.)]
        for i in range(trace_length - 1):
            ac_logp0_cumsum.append(tf.add(ac_logp0_cumsum[-1], ac_logp0[:, i]))
            ac_logp1_cumsum.append(tf.add(ac_logp1_cumsum[-1], ac_logp1[:, i]))

        # Compute v_0 and v_1
        mat_cumsum = ac_logp0[:, 0] * ac_logp1[:, 0]
        v_0 = mat_cumsum * mainPN[0].sample_reward[:, 0]
        v_1 = mat_cumsum * mainPN[1].sample_reward[:, 0]
        for i in range(1, trace_length):
            mat_cumsum = tf.add(mat_cumsum, ac_logp0[:, i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp0_cumsum[i] * ac_logp1[:, i])
            mat_cumsum = tf.add(mat_cumsum, ac_logp1_cumsum[i] * ac_logp0[:, i])
            v_0 = tf.add(v_0, mat_cumsum * mainPN[0].sample_reward[:, i])
            v_1 = tf.add(v_1, mat_cumsum * mainPN[1].sample_reward[:, i])
        v_0 = 2 * tf.reduce_sum(v_0) / batch_size
        v_1 = 2 * tf.reduce_sum(v_1) / batch_size

    v_0_pi_0 = 2*tf.reduce_sum(((mainPN[0].target-tf.stop_gradient(mainPN[0].value)) * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_0_pi_1 = 2*tf.reduce_sum(((mainPN[0].target-tf.stop_gradient(mainPN[0].value)) * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    v_1_pi_0 = 2*tf.reduce_sum(((mainPN[1].target-tf.stop_gradient(mainPN[1].value)) * mainPN[0].gamma_array) * mainPN[0].log_pi_action_bs_t) / batch_size
    v_1_pi_1 = 2*tf.reduce_sum(((mainPN[1].target-tf.stop_gradient(mainPN[1].value)) * mainPN[1].gamma_array) * mainPN[1].log_pi_action_bs_t) / batch_size

    v_0_grad_theta_0 = flatgrad(v_0_pi_0, mainPN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0_pi_1, mainPN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1_pi_0, mainPN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1_pi_1, mainPN[1].parameters)

    mainPN[0].grad = v_0_grad_theta_0
    mainPN[1].grad = v_1_grad_theta_1

    mainPN[0].grad_v_1 = v_1_grad_theta_0
    mainPN[1].grad_v_0 = v_0_grad_theta_1

    # Corrections enabled for V0
    v_0_grad_theta_0_wrong = flatgrad(v_0, mainPN[0].parameters)
    v_1_grad_theta_1_wrong = flatgrad(v_1, mainPN[1].parameters)

    # param_len = v_0_grad_theta_0_wrong.get_shape()[0].value
    param_len = v_1_grad_theta_1_wrong.get_shape()[0].value

    multiply0 = tf.matmul(
        tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
        tf.reshape(v_1_grad_theta_1_wrong, [param_len, 1])
    )

    second_order0 = flatgrad(multiply0, mainPN[0].parameters)

    mainPN[0].v_0_grad_01 = second_order0

    mainPN[0].delta = v_0_grad_theta_0 + second_order0

    # Correction disabled for V1
    mainPN[1].delta = v_1_grad_theta_1        


def corrections_func_3player(mainPN, batch_size, trace_length):
    """Computes corrections for policy gradients.
    Corresponds to the case of Corrections=True, Cube=None in
    the original corrections_func

    Args:
    -----
        mainPN: list of policy/Q-networks
        batch_size: int
        trace_length: int
    """
    n_agents = len(mainPN)
    # ------ else case of original cube condition ---------- #
    ac_logp0 = tf.reshape(mainPN[0].log_pi_action_bs_t,
                          [batch_size, trace_length])
    ac_logp1 = tf.reshape(mainPN[1].log_pi_action_bs_t,
                          [batch_size, trace_length])
    ac_logp2 = tf.reshape(mainPN[2].log_pi_action_bs_t,
                          [batch_size, trace_length])    
    ac_logp = [ac_logp0, ac_logp1, ac_logp2]

    # Static exclusive cumsum
    ac_logp0_cumsum = [tf.constant(0.)]
    ac_logp1_cumsum = [tf.constant(0.)]
    ac_logp2_cumsum = [tf.constant(0.)]    
    for i in range(trace_length - 1):
        ac_logp0_cumsum.append(tf.add(ac_logp0_cumsum[-1], ac_logp0[:, i]))
        ac_logp1_cumsum.append(tf.add(ac_logp1_cumsum[-1], ac_logp1[:, i]))
        ac_logp2_cumsum.append(tf.add(ac_logp2_cumsum[-1], ac_logp2[:, i]))
    ac_logp_cumsum = [ac_logp0_cumsum, ac_logp1_cumsum, ac_logp2_cumsum]

    v_i_pi_i = [None] * n_agents
    v_i_grad_theta_i = [None] * n_agents
    for i in range(n_agents):
        v_i_pi_i[i] = 2*tf.reduce_sum(((mainPN[i].target-tf.stop_gradient(mainPN[i].value)) * mainPN[i].gamma_array) * mainPN[i].log_pi_action_bs_t) / batch_size
        v_i_grad_theta_i[i] = flatgrad(v_i_pi_i[i], mainPN[i].parameters)
        mainPN[i].delta = v_i_grad_theta_i[i]

    for ai in range(n_agents):
        for aj in range(ai, n_agents):
            # Compute v_i and v_j
            mat_cumsum = ac_logp[ai][:, 0] * ac_logp[aj][:, 0]
            v_ij = mat_cumsum * mainPN[ai].sample_reward[:, 0]
            v_ji = mat_cumsum * mainPN[aj].sample_reward[:, 0]
            for i in range(1, trace_length):
                mat_cumsum = tf.add(mat_cumsum, ac_logp[ai][:, i] * ac_logp[aj][:, i])
                mat_cumsum = tf.add(mat_cumsum, ac_logp_cumsum[ai][i] * ac_logp[aj][:, i])
                mat_cumsum = tf.add(mat_cumsum, ac_logp_cumsum[aj][i] * ac_logp[ai][:, i])
                v_ij = tf.add(v_ij, mat_cumsum * mainPN[ai].sample_reward[:, i])
                v_ji = tf.add(v_ji, mat_cumsum * mainPN[aj].sample_reward[:, i])
            v_ij = 2 * tf.reduce_sum(v_ij) / batch_size
            v_ji = 2 * tf.reduce_sum(v_ji) / batch_size            

            v_i_pi_j = 2*tf.reduce_sum(((mainPN[ai].target-tf.stop_gradient(mainPN[ai].value)) * mainPN[aj].gamma_array) * mainPN[aj].log_pi_action_bs_t) / batch_size

            v_j_pi_i = 2*tf.reduce_sum(((mainPN[aj].target-tf.stop_gradient(mainPN[aj].value)) * mainPN[ai].gamma_array) * mainPN[ai].log_pi_action_bs_t) / batch_size

            v_i_grad_theta_j = flatgrad(v_i_pi_j, mainPN[aj].parameters)

            v_j_grad_theta_i = flatgrad(v_j_pi_i, mainPN[ai].parameters)

            mainPN[ai].grad = v_i_grad_theta_i[ai]
            mainPN[aj].grad = v_i_grad_theta_i[aj]

            v_i_grad_theta_i_wrong = flatgrad(v_ij, mainPN[ai].parameters)
            v_j_grad_theta_j_wrong = flatgrad(v_ji, mainPN[aj].parameters)

            param_len = v_i_grad_theta_i_wrong.get_shape()[0].value

            multiplyi = tf.matmul(
                tf.reshape(tf.stop_gradient(v_i_grad_theta_j), [1, param_len]),
                tf.reshape(v_j_grad_theta_j_wrong, [param_len, 1])
            )
            multiplyj = tf.matmul(
                tf.reshape(tf.stop_gradient(v_j_grad_theta_i), [1, param_len]),
                tf.reshape(v_i_grad_theta_i_wrong, [param_len, 1])
            )

            second_orderi = flatgrad(multiplyi, mainPN[ai].parameters)
            second_orderj = flatgrad(multiplyj, mainPN[aj].parameters)

            mainPN[ai].delta = mainPN[ai].delta + second_orderi
            mainPN[aj].delta = mainPN[aj].delta + second_orderj




