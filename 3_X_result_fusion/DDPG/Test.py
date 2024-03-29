import time
from EnvUAV.env import YawControlEnv
from Agent import DDPG
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    path += '/DDPG_0.98/'
    agent = DDPG(path, s_dim=6, gamma=0.98)
    agent.load_net('1', '197')
    # 3 192; 6 171; 7 92; 9 154
    agent.var = 0
    env = YawControlEnv()

    x = []
    x_target = []
    roll = []
    roll_target = []

    y = []
    y_target = []
    pitch = []
    pitch_target = []

    z = []
    z_target = []
    yaw = []
    yaw_target = []

    action = []

    targets = np.array([[5, 5, 5],
                        [-5, -5, -5]])
    for episode in range(2):
        target = targets[episode, :]  # [0, 0, 0]
        s = env.reset(target=target)
        for ep_step in range(500):
            a = agent.get_action(s)
            s_, r, done, info = env.step(a[0])
            s = s_
            print(ep_step, env.current_ang[2], np.sin(env.current_ang[2]), s, r)

            action.append(a[0])

            x.append(env.current_pos[0])
            x_target.append(0)
            roll.append(env.current_ang[0])
            roll_target.append(target[0])

            y.append(env.current_pos[1])
            y_target.append(0)
            pitch.append(env.current_ang[1])
            pitch_target.append(target[1])

            z.append(env.current_pos[2])
            z_target.append(0)
            yaw.append(env.current_ang[2])
            yaw_target.append(target[2])

    index = np.array(range(len(x))) * 0.01
    zeros = np.zeros_like(index)
    roll = np.array(roll) / np.pi * 180
    pitch = np.array(pitch) / np.pi * 180
    yaw = np.array(yaw) / np.pi * 180
    roll_target = np.array(roll_target) / np.pi * 180
    pitch_target = np.array(pitch_target) / np.pi * 180
    yaw_target = np.array(yaw_target) / np.pi * 180
    plt.subplot(3, 2, 1)
    plt.plot(index, x, label='x')
    plt.plot(index, x_target, label='x_target')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(index, pitch, label='pitch')
    # plt.plot(index, pitch_target)
    plt.plot(index, zeros)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(index, y, label='y')
    plt.plot(index, y_target, label='y_target')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(index, roll, label='roll')
    # plt.plot(index, roll_target)
    plt.plot(index, zeros)
    plt.plot(index, action, label='action')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(index, z, label='z')
    plt.plot(index, z_target, label='z_target')
    # plt.plot(index, action)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(index, yaw, label='yaw')
    # plt.plot(index, yaw_target)
    # plt.plot(index, action)
    plt.plot(index, zeros)
    plt.legend()

    plt.show()

    x = np.array(x)
    print(x.shape)
    x_record = np.empty(shape=[2, 500])
    x_record[0, :] = x[0:500]
    x_record[1, :] = x[500:1000]
    np.save(path + 'test.npy', x_record)
    plt.plot(np.array(range(500)), x_record[0, :])
    plt.plot(np.array(range(500)), x_record[1, :])
    plt.show()


if __name__ == '__main__':
    main()
