/**
 * @file ising.c
 * @author Kristof Benedek
 * @date 2025-02-26
 *
 * @brief Monte Carlo simulation of the Ising model on arbitrary graphs.
 *
 * This program implements an Ising spin model on a weighted graph defined
 * by an edge list. Each node carries a spin value (+1 or -1) and interacts
 * with its neighbors through edge weights representing coupling strengths.
 *
 * The code supports two Monte Carlo update algorithms:
 *
 *  - Wolff cluster updates (fast equilibration near criticality)
 *  - Metropolis single-spin updates (useful for hysteresis studies)
 *
 * Two types of parameter sweeps are supported:
 *
 *  - Temperature sweep (annealing): T0 → Tf
 *  - External field sweep: h0 → hf
 *
 * Observables measured during the simulation include:
 *
 *  - Energy per node
 *  - Magnetization
 *  - Absolute magnetization
 *  - Specific heat
 *  - Magnetic susceptibility
 *  - Binder cumulant
 *
 * The network is read from a file containing an edge list with optional
 * spin and external field values. Results are written to an output folder
 * together with optional final spin configurations.
 */

#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/* ---- Structural variables ----*/

/**
 * @struct Edge
 * @brief Represents a weighted interaction between two nodes.
 *
 * Each edge connects two nodes and stores the coupling strength
 * used in the Ising Hamiltonian.
 */
typedef struct Edge {
    int node_i; // node u
    int node_j; // node v
    double weight; // 
} Edge;

Edge* graph;

/**
 * @struct Node
 * @brief Represents a node in the graph with spin and adjacency data.
 *
 * Each node stores:
 *  - its current spin value
 *  - neighbor indices
 *  - edge indices corresponding to neighbors
 *  - node degree
 *  - local external field contribution
 */
typedef struct {
    int spin; // spin value of the node (-1 or 1)
    int* neighbors; // array of neighbor node indices
    int* edge_ids; // array of edge indices corresponding to neighbors
    int degree; // number of neighbors
    double field; // external field contribution (if needed)
} Node;

Node* nodes;

/* ---- Initialization ---- */
int init_random = 0;
int seed = 42;
// Default standard 10x10 square lattice, no PBC
int num_nodes; // number of nodes
int num_edges; // number of edges

/* ---- Simulation Params ---- */
double T0 = 1.0; // init temperature
double Tf = 0.1; // fnite temperature
double T = 2.0;
double ext_field = 0.0; // external magnetic field (not used in current implementation)
double ext_field0 = 0.0; // external magnetic field (not used in current implementation)
double ext_fieldf = 0.0; // external magnetic field (not used in current implementation)
double ext_field_step = 0.0; // external magnetic field (not used in current implementation)
double temp_step = 0.01; // temperature step for annealing
char* input_file = NULL;
int thrm_steps = 1000;
int meas_steps = 1000;
int num_temps;
int store_spin_history = 0;     // store all measured configs
int store_final_config = 0;     // store final config per temperature
char network_name[256]; 
int sweep_mode = 0; // 0 = temp sweep, 1 = field sweep
int algorithm = 0; // 0 = Wolff, 1 = Metropolis

/* ---- Storage arrays ---- */
double *temperatures;

double *E_avg;
double *E2_avg;

double *M_avg;
double *M2_avg;
double *M_abs_values;

double *C_values;
double *chi_values;
double *binder_values;

double *E_values;   // size = num_temps
double *M_values;   // size = num_temps

int ***spin_history;   // [temp][meas][node]
int **final_config;    // [temp][node]

int *cluster_stack;
int *cluster_tag;
int current_tag = 1;
int cluster_size;
int *cluster_nodes;

char output_path[1024] = "output";

int verbose = 0;

/**
 * @brief Generate a random spin value.
 *
 * Returns either +1 or -1 with equal probability.
 *
 * @return int Random spin (+1 or -1).
 */
int random_spin() {
    return (rand() % 2) ? 1 : -1;
}

void print_help() {
    printf("Usage: ./ising [options]\n");
    printf("Options:\n");
    printf("  -seed=value          Set random seed (default: 42)\n");
    printf("  -input_file=value    Path to input file with spin values\n");
    printf("  -T0=value            Set initial temperature (default: 1.0)\n");
    printf("  -Tf=value            Set final temperature (default: 0.1)\n");
    printf("  -T_step=value        Set temperature step for annealing (default: 0.01)\n");
    printf("  -T=value             Set fixed temperature for field sweep (default: 2.0)\n");
    printf("  -sweep_mode=value    Set sweep mode (0 for temperature, 1 for field, default: 0)\n");
    printf("  -directed=value      Set graph as directed (0 or 1, default: 0)\n");
    printf("  -init_random=value   Initialize spins randomly (0 or 1, default: 0)\n");
    printf("  -thrm_steps=value    Number of thermalization steps (default: 1000)\n");
    printf("  -meas_steps=value    Number of measurement steps (default: 1000)\n"); 
    printf("  -ext_field=value      External magnetic field (default: 0.0)\n");
    printf("  -ext_field0=value     Initial external magnetic field (default: 0.0)\n");
    printf("  -ext_fieldf=value     Final external magnetic field (default: 0.0)\n");
    printf("  -ext_field_step=value Step size for external magnetic field sweep (default: 0.0)\n");


    printf("  -store_spin_history=value    Store all measured spin configurations (0 or 1, default: 0)\n");
    printf("  -store_final_config=value    Store final spin configuration per temperature (0 or 1, default: 1)\n");
    printf("  -output=value          Output path (default: output)\n");
    printf("  -verbose=value          Verbose output (0 or 1, default: 0)\n");
    printf("  -algorithm=value       Algorithm to use (0 for Wolff, 1 for Metropolis, default: 0)\n");
    printf("\n");
    printf("  --h                  Show this help message\n");
}

void read_cli_args(int argc, char *argv[]){
    // Read command line arguments for number of nodes, edges, and other parameters
    // Example: ./ising n m [other parameters]
     /* ---- Read Command-line Arguments ---- */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--h") == 0) {
            print_help();
            exit(0);
        }
        
        char *value = strchr(argv[i], '=');
        if (value != NULL) {
            *value = '\0';
            value++; 
        } else {
            printf("Error: Invalid argument format. Use -option=value or --option=value\n");
            exit(1);
        }

        if (strcmp(argv[i], "-seed") == 0) seed = atoi(value);
        else if (strcmp(argv[i], "-input_file") == 0) input_file = value;

        else if (strcmp(argv[i], "-T0") == 0) T0 = atof(value);
        else if (strcmp(argv[i], "-Tf") == 0) Tf = atof(value);
        else if (strcmp(argv[i], "-T_step") == 0) temp_step = atof(value);
        else if (strcmp(argv[i], "-T") == 0) T = atof(value);
        else if (strcmp(argv[i], "-sweep_mode") == 0) sweep_mode = atoi(value);
        else if (strcmp(argv[i], "-ext_field0") == 0) ext_field0 = atof(value);
        else if (strcmp(argv[i], "-ext_fieldf") == 0) ext_fieldf = atof(value);
        else if (strcmp(argv[i], "-ext_field_step") == 0) ext_field_step = atof(value);
        else if (strcmp(argv[i], "-init_random") == 0) init_random = atoi(value);
        else if (strcmp(argv[i], "-thrm_steps") == 0) thrm_steps = atoi(value);
        else if (strcmp(argv[i], "-meas_steps") == 0) meas_steps = atoi(value);
        else if (strcmp(argv[i], "-store_spin_history") == 0) store_spin_history = atoi(value);
        else if (strcmp(argv[i], "-store_final_config") == 0) store_final_config = atoi(value);
        else if (strcmp(argv[i], "-output") == 0) {
            strncpy(output_path, value, sizeof(output_path) - 1);
            output_path[sizeof(output_path) - 1] = '\0';
        }
        else if (strcmp(argv[i], "-verbose") == 0) verbose = atoi(value);
        else if (strcmp(argv[i], "-algorithm") == 0) algorithm = atoi(value);
        else {
            printf("Error: Unknown option '%s'\n", argv[i]);
            exit(1);
        }
    }

}

/**
 * @brief Generate a unique identifier for output files.
 *
 * Uses the current wall-clock time (seconds + nanoseconds) to
 * construct a unique run identifier.
 *
 * @param buffer Output buffer receiving the ID string.
 * @param size   Size of the buffer.
 */
void generate_unique_id(char *buffer, size_t size)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    snprintf(buffer, size, "%ld_%ld",
             ts.tv_sec,
             ts.tv_nsec);
            //  getpid());
}

/**
 * @brief Extract the network name from an input file path.
 *
 * Removes directory prefixes and file extensions so the name
 * can be used when constructing output directories.
 *
 * Example:
 *   "network/lattice32.dat" → "lattice32"
 *
 * @param input_path Input file path.
 * @param out        Output buffer.
 * @param out_size   Size of output buffer.
 */
void extract_network_name(const char *input_path, char *out, size_t out_size)
{
    const char *last_slash = strrchr(input_path, '/');
    const char *filename = last_slash ? last_slash + 1 : input_path;

    const char *last_dot = strrchr(filename, '.');
    size_t len = last_dot ? (size_t)(last_dot - filename)
                          : strlen(filename);

    if (len >= out_size)
        len = out_size - 1;

    strncpy(out, filename, len);
    out[len] = '\0';
}

/**
 * @brief Initialize the Ising system from an input graph file.
 *
 * Reads the network structure from an edge list file and constructs
 * the adjacency lists used during simulation.
 *
 * The input format supports three variants:
 *
 * 1) node_i node_j weight
 * 2) node_i node_j weight spin_i spin_j
 * 3) node_i node_j weight spin_i spin_j field_i field_j
 *
 * If spins are not provided they are initialized randomly.
 *
 * Steps performed:
 *  - Count edges and determine node range
 *  - Allocate graph and node arrays
 *  - Load edges and optional spin/field values
 *  - Construct adjacency lists
 */
void init_system() {

    if (input_file == NULL) {
        fprintf(stderr, "Error: No input file specified.\n");
        exit(EXIT_FAILURE);
    }

    FILE *file = fopen(input_file, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open %s\n", input_file);
        exit(EXIT_FAILURE);
    }

    extract_network_name(input_file, network_name, sizeof(network_name));

    char line[8192];
    int node_i, node_j;
    double weight;
    int spin_i, spin_j;
    double field_i, field_j;

    int min_node_id = INT_MAX;
    int max_node_id = -1;

    num_edges = 0;
    num_nodes = 0;

    int format_with_spins = 0;
    int format_without_spins = 0;
    int format_with_fields = 0;

    /* ========================= */
    /* PASS 1: Count + Detect    */
    /* ========================= */

    while (fgets(line, sizeof(line), file)) {

        char *p = line;
        while (isspace((unsigned char)*p)) p++;

        if (*p == '#' || *p == '\0')
            continue;

        int nread = sscanf(p, "%d %d %lf %d %d %lf %lf",
                           &node_i, &node_j,
                           &weight,
                           &spin_i, &spin_j,
                           &field_i, &field_j);

        if (nread == 5)
            format_with_spins = 1;
        else if (nread == 7)
            format_with_fields = 1;
        else if (nread == 3)
            format_without_spins = 1;
        else
            continue;

        if (format_with_spins && format_without_spins && format_with_fields) {
            fprintf(stderr, "Error: Mixed spin and non-spin lines detected.\n");
            exit(EXIT_FAILURE);
        }

        num_edges++;

        if (node_i < min_node_id) min_node_id = node_i;
        if (node_j < min_node_id) min_node_id = node_j;

        if (node_i > max_node_id) max_node_id = node_i;
        if (node_j > max_node_id) max_node_id = node_j;
    }

    fclose(file);

    num_nodes = max_node_id - min_node_id + 1;

    printf("# Nodes: %d, Edges: %d\n", num_nodes, num_edges);

    /* ========================= */
    /* Allocate graph + nodes    */
    /* ========================= */

    nodes = malloc(num_nodes * sizeof(Node));
    graph = malloc(num_edges * sizeof(Edge));

    if (!nodes || !graph) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_nodes; i++) {
        nodes[i].spin = 1; // initialize spins randomly (will be overwritten if provided in file)
        nodes[i].degree = 0;
        nodes[i].neighbors = NULL;
        nodes[i].edge_ids = NULL;
        nodes[i].field = 1.0; // initialize external field contribution
    }

    /* ========================= */
    /* PASS 2: Fill edges        */
    /* ========================= */

    file = fopen(input_file, "r");
    if (!file) {
        fprintf(stderr, "Error reopening file\n");
        exit(EXIT_FAILURE);
    }

    int edge_index = 0;

    while (fgets(line, sizeof(line), file)) {

        char *p = line;
        while (isspace((unsigned char)*p)) p++;

        if (*p == '#' || *p == '\0')
            continue;

        int nread = sscanf(p, "%d %d %lf %d %d %lf %lf",
                           &node_i, &node_j,
                           &weight,
                           &spin_i, &spin_j,
                           &field_i, &field_j);

        if (nread != 3 && nread != 5 && nread != 7)
            continue;

        int u = node_i - min_node_id;
        int v = node_j - min_node_id;

        graph[edge_index].node_i = u;
        graph[edge_index].node_j = v;
        graph[edge_index].weight = weight;

        if (init_random) {
             spin_i = random_spin();
             spin_j = random_spin();
        } else if (format_with_spins) {
            nodes[u].spin = spin_i;
            nodes[v].spin = spin_j;
        }

        if (format_with_fields) {
            nodes[u].field = field_i;
            nodes[v].field = field_j;
        }

        nodes[u].degree++;
        nodes[v].degree++;

        edge_index++;
    }

    fclose(file);

    /* ========================= */
    /* Random spin initialization */
    /* ========================= */

    if (format_without_spins) {
        printf("# INFO: No spins provided. Initializing uniformly at random.\n");

        srand(seed);
        for (int i = 0; i < num_nodes; i++)
            nodes[i].spin = random_spin();
    }

    /* ========================= */
    /* Allocate adjacency arrays */
    /* ========================= */

    for (int i = 0; i < num_nodes; i++) {
        nodes[i].neighbors = malloc(nodes[i].degree * sizeof(int));
        nodes[i].edge_ids  = malloc(nodes[i].degree * sizeof(int));

        if (!nodes[i].neighbors || !nodes[i].edge_ids) {
            fprintf(stderr, "Memory allocation failed (adjacency)\n");
            exit(EXIT_FAILURE);
        }

        nodes[i].degree = 0;  // reuse as insertion counter
    }

    /* ========================= */
    /* PASS 3: Build adjacency   */
    /* ========================= */

    for (int i = 0; i < num_edges; i++) {

        int u = graph[i].node_i;
        int v = graph[i].node_j;

        nodes[u].neighbors[nodes[u].degree] = v;
        nodes[u].edge_ids[nodes[u].degree] = i;
        nodes[u].degree++;

        nodes[v].neighbors[nodes[v].degree] = u;
        nodes[v].edge_ids[nodes[v].degree] = i;
        nodes[v].degree++;
    }

    printf("# Network successfully initialized\n");
}

/**
 * @brief Compute the total energy of the current spin configuration.
 *
 * The Hamiltonian is
 *
 *   H = - Σ_ij J_ij s_i s_j - Σ_i h_i s_i
 *
 * where J_ij are edge weights and h_i are node fields.
 *
 * @return double Total system energy.
 */
double compute_energy() {
    double E = 0.0;

    // Edge contribution
    for (int e = 0; e < num_edges; e++) {
        int i = graph[e].node_i;
        int j = graph[e].node_j;
        double Jij = graph[e].weight;

        E -= Jij * nodes[i].spin * nodes[j].spin;
    }

    // External field contribution
    for (int i = 0; i < num_nodes; i++) {
        E -= nodes[i].field * nodes[i].spin;
    }

    return E;
}

/**
 * @brief Compute the normalized magnetization of the system.
 *
 * Magnetization is defined as
 *
 *   M = (1/N) Σ_i s_i
 *
 * where N is the number of nodes.
 *
 * @return double Magnetization in the range [-1,1].
 */
double compute_magnetization() {
    double M = 0.0;

    for (int i = 0; i < num_nodes; i++)
        M += nodes[i].spin;

    return M / num_nodes;
}

/**
 * @brief Perform a single Wolff cluster update.
 *
 * A random seed spin is chosen and a cluster of aligned spins is
 * grown using the Wolff bond probability
 *
 *   p = 1 - exp(-2 β J)
 *
 * The entire cluster is then flipped simultaneously.
 *
 * This algorithm significantly reduces critical slowing down
 * near the phase transition.
 *
 * @param beta   Inverse temperature (1/T).
 * @param energy Pointer to the system energy (updated after flip).
 */
void wolff_step(double beta, double *energy)
{
    int stack_size = 0;
    cluster_size = 0;

    int seed_node = rand() % num_nodes;
    int seed_spin = nodes[seed_node].spin;

    current_tag++;

    cluster_stack[stack_size++] = seed_node;
    cluster_tag[seed_node] = current_tag;
    cluster_nodes[cluster_size++] = seed_node;

    while (stack_size > 0)
    {
        int i = cluster_stack[--stack_size];

        for (int k = 0; k < nodes[i].degree; k++)
        {
            int j = nodes[i].neighbors[k];
            int edge_id = nodes[i].edge_ids[k];

            if (cluster_tag[j] == current_tag)
                continue;

            if (nodes[j].spin != seed_spin)
                continue;

            double Jij = graph[edge_id].weight;
            double p = 1.0 - exp(-2.0 * beta * Jij);

            if (((double)rand() / RAND_MAX) < p)
            {
                cluster_tag[j] = current_tag;
                cluster_stack[stack_size++] = j;
                cluster_nodes[cluster_size++] = j;
            }
        }
    }

    /* ===== COMPUTE ΔE ===== */

    double deltaE = 0.0;

    for (int c = 0; c < cluster_size; c++)
    {
        int i = cluster_nodes[c];

        /* field contribution */
        deltaE += 2.0 * nodes[i].field * nodes[i].spin;

        for (int k = 0; k < nodes[i].degree; k++)
        {
            int j = nodes[i].neighbors[k];
            int edge_id = nodes[i].edge_ids[k];
            double Jij = graph[edge_id].weight;

            /* Only count boundary edges */
            if (cluster_tag[j] != current_tag)
            {
                deltaE += 2.0 * Jij * nodes[i].spin * nodes[j].spin;
            }
        }
    }

    /* flip cluster */
    for (int c = 0; c < cluster_size; c++)
        nodes[cluster_nodes[c]].spin *= -1;

    *energy += deltaE;
}

/**
 * @brief Perform one Metropolis Monte Carlo sweep.
 *
 * A sweep consists of N attempted single-spin flips, where N is
 * the number of nodes in the system.
 *
 * Each flip is accepted according to the Metropolis criterion:
 *
 *   P = min(1, exp(-β ΔE))
 *
 * This algorithm preserves metastability and is therefore suitable
 * for hysteresis simulations.
 *
 * @param beta   Inverse temperature.
 * @param energy Pointer to the current system energy.
 */
void metropolis_sweep(double beta, double *energy)
{
    for (int k = 0; k < num_nodes; k++) {

        int i = rand() % num_nodes;

        int s = nodes[i].spin;

        int neighbor_sum = 0;

        for (int n = 0; n < nodes[i].degree; n++) {
            int j = nodes[i].neighbors[n];
            neighbor_sum += nodes[j].spin;
        }

        double dE = 2.0 * s * (neighbor_sum + ext_field);

        if (dE <= 0.0 || ((double)rand()/RAND_MAX) < exp(-beta * dE)) {

            nodes[i].spin = -s;

            *energy += dE;
        }
    }
}

/**
 * @brief Allocate memory for simulation observables and cluster data.
 *
 * Determines the number of sweep points depending on the selected
 * sweep mode (temperature or field) and allocates arrays for:
 *
 *  - thermodynamic observables
 *  - cluster data structures
 *  - optional final configurations
 *
 * Terminates the program if memory allocation fails.
 */
void allocate_memory()
{
    if (sweep_mode == 0) {
        /* ---- Compute number of temperatures ---- */
        if (temp_step <= 0) {
            temp_step *= -1;
        }
        if (T0 < 0 || Tf < 0) {
            fprintf(stderr, "Error: Temperatures must be non-negative.\n");
            exit(EXIT_FAILURE);
        }
        if (T0 < Tf) {
            double temp_range = Tf - T0;
            num_temps = (int)(temp_range / temp_step) + 1;
        } else {
            num_temps = (int)((T0 - Tf) / temp_step) + 1;
        }
    } else if (sweep_mode == 1) {

        if (ext_field_step == 0) {
            fprintf(stderr, "Error: ext_field_step must be non-zero for field sweep\n");
            exit(EXIT_FAILURE);
        }

        if (ext_field_step < 0)
            ext_field_step *= -1;

        if (ext_field0 < ext_fieldf) {
            double field_range = ext_fieldf - ext_field0;
            num_temps = (int)(field_range / ext_field_step) + 1;
        } else {
            num_temps = (int)((ext_fieldf - ext_field0) / ext_field_step * -1) + 1;
        }
    } else {
        fprintf(stderr, "Error: Invalid sweep mode %d\n", sweep_mode);
        exit(EXIT_FAILURE);
    }

    printf("# Allocating memory for %d temperatures\n", num_temps);

    /* ---- Allocate Wolff arrays ---- */
    cluster_stack = malloc(num_nodes * sizeof(int));
    cluster_tag   = malloc(num_nodes * sizeof(int));
    cluster_nodes = malloc(num_nodes * sizeof(int));
    binder_values = malloc(num_temps * sizeof(double));

    if (!cluster_stack || !cluster_tag) {
        fprintf(stderr, "Memory allocation failed (cluster arrays)\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_nodes; i++)
        cluster_tag[i] = 0;

    /* ---- Allocate observable arrays ---- */
    temperatures = malloc(num_temps * sizeof(double));
    E_values     = malloc(num_temps * sizeof(double));
    M_values     = malloc(num_temps * sizeof(double));
    M_abs_values = malloc(num_temps * sizeof(double));
    C_values     = malloc(num_temps * sizeof(double));
    chi_values   = malloc(num_temps * sizeof(double));

    if (!temperatures || !E_values || !M_values || !M_abs_values ||
        !C_values || !chi_values)
    {
        fprintf(stderr, "Memory allocation failed (observable arrays)\n");
        exit(EXIT_FAILURE);
    }

    /* ---- Optional storage ---- */
    if (store_final_config) {
        final_config = malloc(num_temps * sizeof(int*));
        if (!final_config) {
            fprintf(stderr, "Memory allocation failed (final_config)\n");
            exit(EXIT_FAILURE);
        }

        for (int t = 0; t < num_temps; t++) {
            final_config[t] = malloc(num_nodes * sizeof(int));
            if (!final_config[t]) {
                fprintf(stderr, "Memory allocation failed (final_config row)\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("# Memory allocation complete\n");
}

/**
 * @brief Recursively create directories (like `mkdir -p`).
 *
 * Creates all intermediate directories required for a given path.
 *
 * @param path Directory path to create.
 * @param mode Permission mode (e.g. 0700).
 *
 * @return 0 on success, -1 on error.
 */
int mkdir_p(const char *path, mode_t mode)
{
    char tmp[1024];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);

    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;

            if (mkdir(tmp, mode) != 0) {
                if (errno != EEXIST)
                    return -1;
            }

            *p = '/';
        }
    }

    if (mkdir(tmp, mode) != 0) {
        if (errno != EEXIST)
            return -1;
    }

    return 0;
}

/**
 * @brief Save simulation results to disk.
 *
 * Creates a unique run directory and writes:
 *
 *  - thermodynamic observables for each sweep point
 *  - optional final spin configurations
 *
 * The output file contains columns for
 *
 *   sweep_value  E  |M|  M  C  chi  Binder
 *
 * where the sweep value corresponds to either temperature
 * or external field depending on the selected mode.
 */
void save_results()
{
    struct stat st = {0};

    char unique_id[128];
    generate_unique_id(unique_id, sizeof(unique_id));

    char unique_result_folder[1024];

    snprintf(unique_result_folder, sizeof(unique_result_folder),
             "run_%s_h%.5f_meas%d_thrm%d_T0%.5f_Tf%.5f_dT%.5f",
             network_name,
             ext_field,
             meas_steps,
             thrm_steps,
             T0, Tf, temp_step);

    if (mkdir_p(output_path, 0700) != 0) {
        perror("mkdir base failed");
        exit(EXIT_FAILURE);
    }

    /* Now build full run directory */
    char full_path[1024];
    snprintf(full_path, sizeof(full_path),
            "%s/%s", output_path, unique_result_folder);

    if (mkdir_p(full_path, 0700) != 0) {
        perror("mkdir run folder failed");
        exit(EXIT_FAILURE);
    }

    char filepath[1024];
    snprintf(filepath, sizeof(filepath),
             "%s/thermodynamics_%s.dat",
             full_path, unique_id);

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        perror("fopen failed");
        exit(EXIT_FAILURE);
    }

    if (sweep_mode == 0)
        fprintf(fp, "# T E_per_node M_abs M C chi Binder\n");
    else if (sweep_mode == 1)
        fprintf(fp, "# h E_per_node M_abs M C chi Binder\n");
    else {
        fprintf(stderr, "Error: Invalid sweep mode %d\n", sweep_mode);
        exit(EXIT_FAILURE);
    }

    for (int t = 0; t < num_temps; t++) {
        fprintf(fp, "%.8f %.12f %.12f %.12f %.12f %.12f %.12f\n",
                temperatures[t],
                E_values[t],
                M_abs_values[t],
                M_values[t],
                C_values[t],
                chi_values[t],
                binder_values[t]);
    }

    fclose(fp);

    printf("# Saved thermodynamics to %s\n", filepath);

    /* ---- Save final configurations ---- */
    if (store_final_config) {

        for (int t = 0; t < num_temps; t++) {

            snprintf(filepath, sizeof(filepath),
                     "%s/final_config_T_%0.4f_%s.txt",
                     full_path, temperatures[t], unique_id);

            FILE *fc = fopen(filepath, "w");
            if (!fc) {
                fprintf(stderr, "Error writing config file\n");
                continue;
            }

            for (int i = 0; i < num_nodes; i++)
                fprintf(fc, "%d\n", final_config[t][i]);

            fclose(fc);
        }

        printf("# Saved final configurations\n");
    }
}

/**
 * @brief Run the Monte Carlo simulation.
 *
 * Executes either a temperature sweep or an external field sweep.
 *
 * For each sweep point:
 *  - thermalizes the system
 *  - performs measurement steps
 *  - accumulates thermodynamic observables
 *
 * Supports both Wolff and Metropolis update algorithms.
 *
 * Results are stored in global observable arrays and later
 * written to disk by save_results().
 */
void simulate() {

    if (sweep_mode == 0) {
        for (int t = 0; t < num_temps; t++) {
            double energy = compute_energy();

            double T = T0 - t * temp_step;
            double beta = 1.0 / T;

            temperatures[t] = T;

            /* ---- Thermalization ---- */
            if (algorithm == 0) {
                 for (int i = 0; i < thrm_steps; i++)
                    wolff_step(beta, &energy);
            } else if (algorithm == 1) {
                for (int i = 0; i < thrm_steps; i++)
                    metropolis_sweep(beta, &energy);
            } else {
                fprintf(stderr, "Error: Invalid algorithm %d\n", algorithm);
                exit(EXIT_FAILURE);
            }

            double E_sum = 0.0;
            double E2_sum = 0.0;
            double M_sum = 0.0;
            double M_abs_sum = 0.0;
            double M2_sum = 0.0;
            double M4_sum = 0.0;

            /* ---- Measurement ---- */
            for (int i = 0; i < meas_steps; i++) {

                if (algorithm == 0) {
                    wolff_step(beta, &energy);
                } else if (algorithm == 1) {
                    metropolis_sweep(beta, &energy);
                } else {
                    fprintf(stderr, "Error: Invalid algorithm %d\n", algorithm);
                    exit(EXIT_FAILURE);
                }

                double E = energy;
                double M = compute_magnetization();

                E_sum  += E;
                E2_sum += E * E;

                M_sum  += M;
                M_abs_sum += fabs(M);
                M2_sum += M * M;
                M4_sum += M * M * M * M;
            }

            double norm = 1.0 / meas_steps;

            double E_avg = E_sum * norm;
            double M_avg = M_sum * norm;
            double M_abs_avg = M_abs_sum * norm;

            double C = beta * beta *
                    (E2_sum * norm - E_avg * E_avg) / num_nodes;

            double chi = beta *
                        (M2_sum * norm - M_abs_avg * M_abs_avg) * num_nodes;

            double M2_avg = M2_sum * norm;
            double M4_avg = M4_sum * norm;

            double binder = 1.0 - M4_avg / (3.0 * M_abs_avg * M_abs_avg);

            /* ===== STORE RESULTS HERE ===== */

            E_values[t]  = E_avg / num_nodes;
            M_values[t]  = M_avg;
            M_abs_values[t]  = M_abs_avg;
            C_values[t]  = C;
            chi_values[t] = chi;
            binder_values[t] = binder;

            /* Optional: store final configuration */
            if (store_final_config) {
                for (int i = 0; i < num_nodes; i++)
                    final_config[t][i] = nodes[i].spin;
            }

            if (verbose) {
                printf("T = %f, E = %f, M = %f, C = %f, chi = %f\n",
                    temperatures[t], E_values[t], M_values[t], C_values[t], chi_values[t]);
            }
        }
    } else if (sweep_mode == 1) {

        double beta = 1.0 / T;

        for (int t = 0; t < num_temps; t++) {

            double energy = compute_energy();

            double h;

            if (ext_field0 < ext_fieldf)
                h = ext_field0 + t * ext_field_step;
            else
                h = ext_field0 - t * ext_field_step;

            ext_field = h;
            temperatures[t] = h;   // store field value

            /* ---- Thermalization ---- */
            if (algorithm == 0) {
                 for (int i = 0; i < thrm_steps; i++)
                    wolff_step(beta, &energy);
            } else if (algorithm == 1) {
                for (int i = 0; i < thrm_steps; i++)
                    metropolis_sweep(beta, &energy);
            } else {
                fprintf(stderr, "Error: Invalid algorithm %d\n", algorithm);
                exit(EXIT_FAILURE);
            }

            double E_sum = 0.0;
            double E2_sum = 0.0;
            double M_abs_sum = 0.0;
            double M_sum = 0.0;
            double M2_sum = 0.0;
            double M4_sum = 0.0;

            /* ---- Measurement ---- */
            for (int i = 0; i < meas_steps; i++) {

                if (algorithm == 0) {
                    wolff_step(beta, &energy);
                } else if (algorithm == 1) {
                    metropolis_sweep(beta, &energy);
                } else {
                    fprintf(stderr, "Error: Invalid algorithm %d\n", algorithm);
                    exit(EXIT_FAILURE);
                }

                double E = energy;
                double M = compute_magnetization();

                E_sum  += E;
                E2_sum += E * E;
                M_sum  += M;
                M_abs_sum  += fabs(M);
                M2_sum += M * M;
                M4_sum += M * M * M * M;
            }

            double norm = 1.0 / meas_steps;

            double E_avg = E_sum * norm;
            double M_abs_avg = M_abs_sum * norm;
            double M_avg = M_sum * norm;

            double C = beta * beta *
                    (E2_sum * norm - E_avg * E_avg) / num_nodes;

            double chi = beta *
                        (M2_sum * norm - M_abs_avg * M_abs_avg) * num_nodes;

            double M2_avg = M2_sum * norm;
            double M4_avg = M4_sum * norm;

            double binder = 1.0 - M4_avg / (3.0 * M2_avg * M2_avg);

            E_values[t]  = E_avg / num_nodes;
            M_abs_values[t]  = M_abs_avg;
            M_values[t]  = M_avg;
            C_values[t]  = C;
            chi_values[t] = chi;
            binder_values[t] = binder;

            if (store_final_config) {
                for (int i = 0; i < num_nodes; i++)
                    final_config[t][i] = nodes[i].spin;
            }

            if (verbose) {
                printf("h = %f, E = %f, M = %f\n",
                    temperatures[t], E_values[t], M_values[t]);
            }
        }
    } else {
        fprintf(stderr, "Error: Invalid sweep mode %d\n", sweep_mode);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Entry point of the program.
 *
 * Workflow:
 *
 * 1. Parse command line arguments
 * 2. Initialize the graph and spin system
 * 3. Allocate memory for simulation data
 * 4. Run the Monte Carlo simulation
 * 5. Save results to disk
 */
int main(int argc, char *argv[])
{
    read_cli_args(argc, argv);
    srand(seed);

    init_system();

    allocate_memory();

    simulate();     // must fill arrays

    save_results();

    return 0;
}
