/*
@author: Kristof Benedek
@date: 2025-02-26
* Implements an Ising model on a graph with edge list representation.
* The graph is represented as an array of edges, where each edge connects two nodes and has a weight.
* The nodes have spin values that can be either -1 or 1.

* Applies Wolff cluster algorithm to update the spins of the nodes based on their interactions with neighboring nodes.
*/

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>

/* ---- Structural variables ----*/

typedef struct Edge {
    int node_i; // node u
    int node_j; // node v
    double weight; // 
} Edge;

Edge* graph;

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
double ext_field = 0.0; // external magnetic field (not used in current implementation)
double temp_step = 0.01; // temperature step for annealing
char* input_file = NULL;
int thrm_steps = 1000;
int meas_steps = 1000;
int num_temps;
int store_spin_history = 0;     // store all measured configs
int store_final_config = 1;     // store final config per temperature


/* ---- Storage arrays ---- */
double *temperatures;

double *E_avg;
double *E2_avg;

double *M_avg;
double *M2_avg;
double *Mabs_avg;

double *C_values;
double *chi_values;

double *E_values;   // size = num_temps
double *M_values;   // size = num_temps

int ***spin_history;   // [temp][meas][node]
int **final_config;    // [temp][node]


int *cluster_stack;
int *cluster_tag;
int current_tag = 1;

char *output_path = "output";

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
    printf("  -directed=value      Set graph as directed (0 or 1, default: 0)\n");
    printf("  -init_random=value   Initialize spins randomly (0 or 1, default: 0)\n");
    printf("  -thrm_steps=value    Number of thermalization steps (default: 1000)\n");
    printf("  -meas_steps=value    Number of measurement steps (default: 1000)\n"); 
    printf("  -ext_field=value      External magnetic field (default: 0.0)\n");

    printf("  -store_spin_history=value    Store all measured spin configurations (0 or 1, default: 0)\n");
    printf("  -store_final_config=value    Store final spin configuration per temperature (0 or 1, default: 1)\n");
    printf("  -output=value          Output path (default: output)\n");
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
        else if (strcmp(argv[i], "-ext_field") == 0) ext_field = atof(value);
        else if (strcmp(argv[i], "-init_random") == 0) init_random = atoi(value);
        else if (strcmp(argv[i], "-thrm_steps") == 0) thrm_steps = atoi(value);
        else if (strcmp(argv[i], "-meas_steps") == 0) meas_steps = atoi(value);
        else if (strcmp(argv[i], "-store_spin_history") == 0) store_spin_history = atoi(value);
        else if (strcmp(argv[i], "-store_final_config") == 0) store_final_config = atoi(value);
        else if (strcmp(argv[i], "-output") == 0)  output_path = value;

        else {
            printf("Error: Unknown option '%s'\n", argv[i]);
            exit(1);
        }
    }
}

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
        nodes[i].spin = 1;
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

        if (format_with_spins) {
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

void allocate_observables() {

    num_temps = (int)(fabs(Tf - T0) / temp_step) + 1;

    temperatures = calloc(num_temps, sizeof(double));

    E_avg     = calloc(num_temps, sizeof(double));
    E2_avg    = calloc(num_temps, sizeof(double));
    M_avg     = calloc(num_temps, sizeof(double));
    M2_avg    = calloc(num_temps, sizeof(double));
    Mabs_avg  = calloc(num_temps, sizeof(double));

    if (!temperatures || !E_avg || !E2_avg ||
        !M_avg || !M2_avg || !Mabs_avg) {
        fprintf(stderr, "Error allocating observable arrays\n");
        exit(EXIT_FAILURE);
    }

    /* Fill temperature grid */
    double T = T0;

    for (int i = 0; i < num_temps; i++) {

        temperatures[i] = T;

        if (T0 > Tf)
            T -= temp_step;
        else
            T += temp_step;
    }

    printf("# Allocated observables for %d temperatures\n", num_temps);
}

void allocate_results()
{
    E_values   = malloc(num_temps * sizeof(double));
    M_values   = malloc(num_temps * sizeof(double));
    C_values   = malloc(num_temps * sizeof(double));
    chi_values = malloc(num_temps * sizeof(double));

    if (!E_values || !M_values || !C_values || !chi_values) {
        fprintf(stderr, "Error allocating result arrays\n");
        exit(EXIT_FAILURE);
    }

    if (store_final_config) {
        final_config = malloc(num_temps * sizeof(int *));
        for (int t = 0; t < num_temps; t++)
            final_config[t] = malloc(num_nodes * sizeof(int));

        printf("# Final configuration storage enabled\n");
    }
}

void allocate_wolff_buffers() {
    cluster_stack = malloc(num_nodes * sizeof(int));
    cluster_tag   = calloc(num_nodes, sizeof(int));

    if (!cluster_stack || !cluster_tag) {
        fprintf(stderr, "Error allocating Wolff buffers\n");
        exit(EXIT_FAILURE);
    }
}

int wolff_step(double beta)
{
    /* Pick random seed */
    int seed = rand() % num_nodes;
    int spin_seed = nodes[seed].spin;

    int stack_top = 0;

    /* Initialize cluster */
    cluster_stack[stack_top++] = seed;
    cluster_tag[seed] = current_tag;

    int cluster_size = 1;

    /* Grow cluster */
    while (stack_top > 0)
    {
        int current = cluster_stack[--stack_top];

        for (int k = 0; k < nodes[current].degree; k++)
        {
            int neighbor = nodes[current].neighbors[k];

            /* Already in cluster? */
            if (cluster_tag[neighbor] == current_tag)
                continue;

            /* Only same spin */
            if (nodes[neighbor].spin != spin_seed)
                continue;

            /* Retrieve weight from graph */
            int edge_id = nodes[current].edge_ids[k];
            double Jij = graph[edge_id].weight;

            /* Wolff bond probability */
            double p_add = 1.0 - exp(-2.0 * beta * Jij);

            if ((double)rand() / RAND_MAX < p_add)
            {
                cluster_tag[neighbor] = current_tag;
                cluster_stack[stack_top++] = neighbor;
                cluster_size++;
            }
        }
    }

    /* Flip entire cluster */
    for (int i = 0; i < cluster_size; i++)
    {
        int node = cluster_stack[i];
        nodes[node].spin *= -1;
    }

    /* Advance tag instead of clearing array */
    current_tag++;

    /* Optional: prevent overflow (extremely rare) */
    if (current_tag == INT_MAX) {
        memset(cluster_tag, 0, num_nodes * sizeof(int));
        current_tag = 1;
    }
    printf("# Size %d\n", cluster_size);

    return cluster_size;
}

void wolff_sweep(double beta)
{
    int flipped = 0;

    while (flipped < num_nodes)
    {
        int size = wolff_step(beta);
        flipped += size;
    }
}


double compute_energy()
{
    double E = 0.0;

    for (int e = 0; e < num_edges; e++)
    {
        int i = graph[e].node_i;
        int j = graph[e].node_j;
        double Jij = graph[e].weight;

        E -= Jij * (double)nodes[i].spin * (double)nodes[j].spin; 
        // printf("# Edge %d-%d: Jij=%.3f  S_i=%d  S_j=%d  Contribution=%.3f\n field = %.3f\n",
        //        i, j, Jij, nodes[i].spin, nodes[j].spin, -Jij * nodes[i].spin * nodes[j].spin, nodes[i].field);
    }

    for (int i = 0; i < num_nodes; i++)
        E -= ext_field * (double)nodes[i].spin * nodes[i].field; // external field contribution (if needed)

    return E;
}

double compute_magnetization()
{
    double M = 0.0;

    for (int i = 0; i < num_nodes; i++)
        M += nodes[i].spin;

    return M;
}

void write_output()
{
    printf("# Writing output to %s\n", output_path);

#ifdef _WIN32
    _mkdir(output_path);
#else
    mkdir(output_path, 0755);
#endif

    char filename[512];

    /* ----------------------------- */
    /* Energy file                   */
    /* ----------------------------- */

    snprintf(filename, sizeof(filename),
             "%s/energy.csv", output_path);

    FILE *fE = fopen(filename, "w");

    fprintf(fE, "# T  Energy\n");
    for (int t = 0; t < num_temps; t++)
        fprintf(fE, "%.10f %.12f\n",
                temperatures[t],
                E_values[t]);

    fclose(fE);

    /* ----------------------------- */
    /* Magnetization file            */
    /* ----------------------------- */

    snprintf(filename, sizeof(filename),
             "%s/magnetization.csv", output_path);

    FILE *fM = fopen(filename, "w");

    fprintf(fM, "# T  Magnetization\n");
    for (int t = 0; t < num_temps; t++)
        fprintf(fM, "%.10f %.12f\n",
                temperatures[t],
                M_values[t]);

    fclose(fM);

    snprintf(filename, sizeof(filename),
         "%s/heat_capacity.csv", output_path);

    FILE *fC = fopen(filename, "w");
    fprintf(fC, "# T  C\n");
    for (int t = 0; t < num_temps; t++)
        fprintf(fC, "%.10f %.12f\n",
                temperatures[t],
                C_values[t]);
    fclose(fC);

    snprintf(filename, sizeof(filename),
            "%s/susceptibility.csv", output_path);

    FILE *fX = fopen(filename, "w");
    fprintf(fX, "# T  chi\n");
    for (int t = 0; t < num_temps; t++)
        fprintf(fX, "%.10f %.12f\n",
                temperatures[t],
                chi_values[t]);
    fclose(fX);

    /* ----------------------------- */
    /* Final configurations (optional) */
    /* ----------------------------- */

    if (store_final_config)
    {
        for (int t = 0; t < num_temps; t++)
        {
            snprintf(filename, sizeof(filename),
                     "%s/final_config_T%.6f.csv",
                     output_path,
                     temperatures[t]);

            FILE *fC = fopen(filename, "w");

            for (int i = 0; i < num_nodes; i++)
                fprintf(fC, "%d\n", final_config[t][i]);

            fclose(fC);
        }
    }

    printf("# Output writing complete\n");
}

/*
* This is a simple implementation of the Ising model in C on any graph with edge list representation.
* 
*/
int main(int argc, char **argv)
{
    read_cli_args(argc, argv);
    srand(seed);
    
    init_system();
    printf("# Node 0 spin: %d field: %.3f degree %d\n", nodes[0].spin, nodes[0].field, nodes[0].degree);
    allocate_wolff_buffers();
    allocate_observables();
    allocate_results();

    printf("# Starting simulation\n");

    for (int t = 0; t < num_temps; t++)
    {
        double T = temperatures[t];
        double beta = 1.0 / T;

        printf("# T = %.5f\n", T);

        /* ---- Thermalization at this temperature ---- */
        for (int i = 0; i < thrm_steps; i++)
            // wolff_sweep(beta);
            wolff_step(beta); 

        double E_sum = 0.0;
        double E2_sum = 0.0;
        double M_sum = 0.0;
        double M2_sum = 0.0;
        double Mabs_sum = 0.0;

        for (int m = 0; m < meas_steps; m++)
        {
            // wolff_sweep(beta);
            wolff_step(beta); 

            double E = compute_energy();
            double M = compute_magnetization();

            E_sum  += E;
            E2_sum += E * E;
            // printf("# Measurement %d: E=%.6f  M=%.6f\n", m, E, fabs(M));
            M_sum  += M;
            M2_sum += M * M;
            Mabs_sum += fabs(M);
        }

        /* Compute averages */
        double E_avg  = E_sum  / meas_steps;
        double E2_avg = E2_sum / meas_steps;

        double M_avg  = M_sum  / meas_steps;
        double M2_avg = M2_sum / meas_steps;
        double Mabs_avg = Mabs_sum / meas_steps;

        /* Intensive quantities */
        E_values[t] = E_avg;
        M_values[t] = Mabs_avg;

        /* Heat capacity */
        C_values[t] = beta * beta * (E2_avg - E_avg * E_avg);

        /* Susceptibility */
        chi_values[t] = beta * (M2_avg - Mabs_avg * Mabs_avg);


        /* Store final configuration if requested */
        if (store_final_config)
        {
            for (int i = 0; i < num_nodes; i++)
                final_config[t][i] = nodes[i].spin;
        }

        printf("# Finished T=%.5f  <E>=%.6f  <M>=%.6f\n",
               T, E_values[t], M_values[t]);
    }

    printf("# Simulation complete\n");

    write_output();

    return 0;
}