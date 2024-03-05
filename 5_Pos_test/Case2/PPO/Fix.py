import time
from EnvUAV.env import YawControlEnv
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))

    env = YawControlEnv()

    pos = []
    ang = []

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

    name = 'Test'
    env.reset(base_pos=np.array([0.001, 0, -5]), base_ang=np.array([0, 0, 0]))
    targets = np.array([[0, 0,  0,  0],
                        [0, 0, 0, 0]])
    for episode in range(1):
        target = targets[episode, :]
        for ep_step in range(500):
            env.step(target)

            pos.append(env.current_pos.tolist())
            ang.append(env.current_ang.tolist())

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

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    ax.plot(x, y, z)
    ax.view_init(azim=45., elev=30)
    # plt.xticks([-0.05, -0.025, 0, 0.025, 0.05])
    # plt.yticks([-0.05, -0.025, 0, 0.025, 0.05])
    # plt.show()
    # plt.subplot(1, 2, 1)
    # plt.title('x')
    # plt.plot(x, z, label='x')
    # plt.subplot(1, 2, 2)
    # plt.title('y')
    # plt.plot(y, z, label='y')
    # plt.show()
    pos = np.array(pos)
    ang = np.array(ang)
    np.save(path + '/PPO_' + 'Case2_' + 'bias'+ '_pos.npy', pos)
    # np.save(path + '/PPO_' + 'Case1_' + 'no_bias'+ '_pos.npy', pos)


if __name__ == '__main__':
    main()

