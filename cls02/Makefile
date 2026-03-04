# ===============================
#  Ising Model Makefile
# ===============================

CC       = gcc
CFLAGS   = -O3 -Wall -Wextra -std=c11
LDFLAGS  = -lm
TARGET   = ising

# Optional OpenMP support
OMPFLAGS = -fopenmp

# Source files
SRCS = ising.c
OBJS = $(SRCS:.c=.o)

# ===============================
# Default build
# ===============================

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ===============================
# Debug build
# ===============================

debug: CFLAGS = -g -O0 -Wall -Wextra -std=c11
debug: clean $(TARGET)

# ===============================
# OpenMP build
# ===============================

omp: CFLAGS += $(OMPFLAGS)
omp: LDFLAGS += $(OMPFLAGS)
omp: clean $(TARGET)

# ===============================
# Clean
# ===============================

clean:
	rm -f $(OBJS) $(TARGET)

# ===============================
# Run example
# ===============================

run:
	./$(TARGET)

