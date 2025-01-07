import simulation as sim
import numpy as np
import matplotlib.pyplot as plt
import h5py

def try_gather_data_and_plot(state,beta,k,n,measurements):
    try_gather_data(state,beta,k,n,measurements)
    try_plot_data(len(state),beta,measurements)

def try_gather_data(state,beta,k,n,measurements):
    with h5py.File('lattice.hdf5','a') as f:
        width = len(state)
        if not f"actions_w{width}_b{np.round(beta,decimals=2)}" in f:
            actions = sim.run_simulation(state,beta,k,n,measurements)
            f.create_dataset(f"actions_w{width}_b{np.round(beta,decimals=2)}",data=actions)

def try_plot_data(width,beta,measurements):
    with h5py.File('lattice.hdf5','r') as f:
        average_plaquette_actions = f[f"actions_w{width}_b{np.round(beta,decimals=2)}"][()]

        plt.scatter(np.linspace(1,measurements,measurements),average_plaquette_actions)
        plt.xlabel("iterations")
        plt.ylabel("average action per plaquette")

        plt.plot(np.linspace(1,measurements,measurements),[np.mean(average_plaquette_actions) for _ in range(measurements)],color="orange")

        print(f"Mean: {np.mean(average_plaquette_actions)}")

def time_until_convergence(width,beta):
    """Sample two Markov chains simultaneously, one started in an aligned state
    and one in a uniform one. Return the first time (in steps) at which the
    absolute action of the second drops below the first."""
    num_links = width*width*width*width*4

    uniform_state = sim.disordered_state(width)
    aligned_state = sim.ordered_state(width)

    uniform_action = sim.average_plaquette_action(uniform_state)
    aligned_action = sim.average_plaquette_action(aligned_state)

    step = 0
    while abs(aligned_action) < abs(uniform_action):
        step += 1

        for _ in range(num_links):
            sim.lattice_heatbath_update(uniform_state,beta)
            sim.lattice_heatbath_update(aligned_state,beta)

        uniform_action = sim.average_plaquette_action(uniform_state)
        aligned_action = sim.average_plaquette_action(aligned_state)

    return step
