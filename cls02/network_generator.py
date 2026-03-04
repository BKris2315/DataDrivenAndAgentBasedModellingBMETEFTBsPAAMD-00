import argparse
import random
import igraph as ig


"""
Example usage:

Random-field Ising model on a 2D lattice:
python generate_graph.py \
--type square \
--L 32 \
--pbc \
--field_mode gaussian \
--field 0.0 \
--field_strength 0.5 \
--random_spins \
--output rfim.dat

Spin-glass example:
python generate_graph.py \
--type ba \
--N 1000 \
--m 3 \
--weight_mode bimodal \
--J 1.0 \
--random_spins \
--output spin_glass.dat

Gaussian bond disorder on ER graph:
python generate_graph.py \
--type er \
--N 500 \
--p 0.02 \
--weight_mode gaussian \
--J 0.0 \
--J_strength 1.0 \
--random_spins \
--output gaussian_bonds.dat
"""

# -------------------------------------------------
# Random helpers
# -------------------------------------------------

def random_spin():
    return random.choice([-1, 1])


def sample_field(mode, base_value, strength):
    if mode == "uniform":
        return base_value
    elif mode == "random_uniform":
        return random.uniform(-strength, strength)
    elif mode == "gaussian":
        return random.gauss(base_value, strength)
    else:
        raise ValueError("Unknown field mode")


def sample_weight(mode, base_value, strength):
    if mode == "uniform":
        return base_value
    elif mode == "random_uniform":
        return random.uniform(-strength, strength)
    elif mode == "gaussian":
        return random.gauss(base_value, strength)
    elif mode == "bimodal":
        return base_value if random.random() < 0.5 else -base_value
    else:
        raise ValueError("Unknown weight mode")


# -------------------------------------------------
# Spin & Field Assignment
# -------------------------------------------------

def assign_spins_fields(N, random_spins, field_mode,
                        field_value, field_strength):

    spins = [random_spin() if random_spins else 1 for _ in range(N)]

    fields = [
        sample_field(field_mode, field_value, field_strength)
        for _ in range(N)
    ]

    return spins, fields


# -------------------------------------------------
# Graph Generators
# -------------------------------------------------

def generate_square_lattice(L, pbc=False):
    return ig.Graph.Lattice([L, L], circular=pbc)


def generate_erdos_renyi(N, p):
    return ig.Graph.Erdos_Renyi(n=N, p=p)


def generate_barabasi(N, m):
    return ig.Graph.Barabasi(n=N, m=m, directed=False)


def generate_watts_strogatz(N, k, p):
    return ig.Graph.Watts_Strogatz(dim=1, size=N, nei=k//2, p=p)


# -------------------------------------------------
# Writer
# -------------------------------------------------

def write_graph(filename, g, spins, fields,
                weight_mode, J_value, J_strength):

    with open(filename, "w") as f:
        f.write("# node_i node_j weight spin_i spin_j field_i field_j\n")

        for e in g.es:
            i, j = e.tuple

            Jij = sample_weight(weight_mode, J_value, J_strength)

            f.write(f"{i} {j} {Jij:.6f} "
                    f"{spins[i]} {spins[j]} "
                    f"{fields[i]:.6f} {fields[j]:.6f}\n")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type",
                        choices=["square", "er", "ba", "ws"],
                        required=True)

    parser.add_argument("--N", type=int)
    parser.add_argument("--L", type=int)

    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=4)

    # Weight disorder
    parser.add_argument("--weight_mode",
                        choices=["uniform", "random_uniform",
                                 "gaussian", "bimodal"],
                        default="uniform")
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--J_strength", type=float, default=1.0)

    # Field disorder
    parser.add_argument("--field_mode",
                        choices=["uniform", "random_uniform", "gaussian"],
                        default="uniform")
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--field_strength", type=float, default=1.0)

    parser.add_argument("--pbc", action="store_true")
    parser.add_argument("--random_spins", action="store_true")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        ig.set_random_number_generator(random)

    # ----- Generate graph -----

    if args.type == "square":
        if args.L is None:
            raise ValueError("Provide --L for square lattice")
        g = generate_square_lattice(args.L, args.pbc)

    elif args.type == "er":
        if args.N is None:
            raise ValueError("Provide --N for ER graph")
        g = generate_erdos_renyi(args.N, args.p)

    elif args.type == "ba":
        if args.N is None:
            raise ValueError("Provide --N for BA graph")
        g = generate_barabasi(args.N, args.m)

    elif args.type == "ws":
        if args.N is None:
            raise ValueError("Provide --N for WS graph")
        g = generate_watts_strogatz(args.N, args.k, args.p)

    N = g.vcount()

    spins, fields = assign_spins_fields(
        N,
        args.random_spins,
        args.field_mode,
        args.field,
        args.field_strength
    )

    write_graph(
        args.output,
        g,
        spins,
        fields,
        args.weight_mode,
        args.J,
        args.J_strength
    )

    print(f"Graph written to {args.output}")
    print(f"Nodes: {g.vcount()}, Edges: {g.ecount()}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Field mode: {args.field_mode}")


if __name__ == "__main__":
    main()
