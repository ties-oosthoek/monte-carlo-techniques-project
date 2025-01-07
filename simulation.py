import numpy as np
rng = np.random.default_rng()

def ordered_state(width):
    """Return the ordered state in which every link variable is aligned and equal to unity."""
    return np.exp(2j * np.pi * np.zeros((width,width,width,width,4)))

def disordered_state(width):
    """Return a disordered state in which every link variable is initiated with a random angle."""
    return np.exp(2j * np.pi * np.random.rand(width,width,width,width,4))

def run_simulation(state,beta,k,n,measurements):
    """Run the simulation with k equilibration sweeps and n measurement sweeps.
    Returns the action measurement results."""
    width = len(state)
    num_links = width*width*width*width*4

    # equilibrate
    run_lattice_heatbath(state,beta,k*num_links)

    # measure
    results = np.empty(measurements)
    for i in range(measurements):
        run_lattice_heatbath(state,beta,n*num_links)
        results[i] = average_plaquette_action(state)
        print(f"Running... ({i+1} / {measurements})")

    print("Completed run")
    return results

def run_lattice_heatbath(state,beta,n):
    """Perform n heatbath updates on the lattice state."""
    for _ in range(n):
        lattice_heatbath_update(state,beta)

def lattice_heatbath_update(state,beta):
    """Perform a heatbath update on a random link in the lattice state."""
    width = len(state)
    link_index = random_link_index(width)
    state[link_index] = sample_link_variable(state,beta,link_index)

def random_link_index(width):
    """Return a random link index (n_x,n_y,n_z,kappa)."""
    return tuple(rng.integers(0,[width,width,width,width,4]))

def sample_link_variable(state,beta,link_index):
    """Sample link variable U = exp(i*theta)."""
    link_variable_sum = relevant_link_variable_sum(state,link_index)
    alpha = beta*np.real(link_variable_sum)
    phi = np.angle(link_variable_sum)
    while True:
        Z = rng.uniform(0,1)
        x = -1 + np.log(1 + (np.exp(2*alpha) - 1)*Z)/alpha

        Q = np.exp(alpha*(np.cos(np.pi/2*(1-x))-x))
        Q_max = np.exp(0.2105137*alpha)

        Z_prime = rng.uniform(0,1)
        if Q/Q_max > Z_prime:
            theta = np.pi*(1-x)/2 - phi
            return np.exp(1j*theta)

def relevant_link_variable_sum(state,link_index):
    """Return the sum of the link variables present in the plaquettes containing the relevant link,
    without the contribution of the link itself."""
    width = len(state)
    n = np.array(link_index[:4])
    kappa = link_index[-1]
    kappa_hat = np.array(get_unit_vector(kappa))

    link_variable_sum = 0
    for nu in range(4):
        if nu != kappa:
            nu_hat = get_unit_vector(nu)
            contribution = state[get_lattice_vector(n+kappa_hat,width)][nu]
            contribution *= state[get_lattice_vector(n+nu_hat,width)][kappa]
            contribution *= state[get_lattice_vector(n,width)][nu]
            link_variable_sum += contribution

    return link_variable_sum

def get_unit_vector(index):
    """Return the unit vector from the dimension index."""
    vector = np.zeros(4).astype(int)
    vector[index] = 1
    return vector

def get_lattice_vector(vector,width):
    """Get the lattice vector periodic with the lattice size."""
    return tuple(map(lambda i: i % width, vector))

def average_plaquette_action(state):
    """Compute action of the lattice state."""
    width = len(state)

    action = 0
    for n in lattice_vertices(width):
        for mu in range(4):
            for nu in range(4):
                if mu < nu:
                    mu_hat, nu_hat = get_unit_vector(mu), get_unit_vector(nu)
                    contribution = state[get_lattice_vector(n,width)][mu]
                    contribution *= state[get_lattice_vector(n+mu_hat,width)][nu]
                    contribution *= state[get_lattice_vector(n+nu_hat,width)][mu]
                    contribution *= state[get_lattice_vector(n,width)][nu]
                    action += (1 - np.real(contribution))

    num_plaquettes = width*width*width*width*6
    return action / num_plaquettes

def lattice_vertices(width):
    """Compute the vertices of the lattice with given width."""
    vertices = []
    for n_x in range(width):
        for n_y in range(width):
            for n_z in range(width):
                for n_t in range(width):
                    vertex = (n_x,n_y,n_z,n_t)
                    vertices.append(vertex)
    return vertices
