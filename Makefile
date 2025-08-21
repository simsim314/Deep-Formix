################################################################################
# CUDA Chess-Engine Makefile (Integrated with Thrust Wrapper Support)
################################################################################

NVCC        := nvcc

# --- Source File Configuration ---
ALL_SRCS      := $(wildcard *.cu)

# Exclude the test main file and thrust wrapper from engine sources
ENGINE_SRCS   := $(filter-out main.cu test_main.cu eval_test.cu thrust_precompiled.cu, $(ALL_SRCS))
ENGINE_OBJS   := $(ENGINE_SRCS:.cu=.o)

# Thrust wrapper object file (precompiled)
THRUST_OBJ    := thrust_precompiled.o

# Collect headers so changes to headers force rebuild of objects
DEPS := $(wildcard *.h */*.h)

# --- Target Configuration ---
MAIN_TARGET   := engine
TEST_TARGET   := test_engine
EVAL_TARGET   := eval_test_runner

# --- Build Flags ---
CUDA_ARCH     ?= -arch=sm_61
STD_FLAG      := -std=c++11
QUIET_WARN    := -Wno-deprecated-gpu-targets

# Common flags used by all builds
BASE_CFLAGS   := $(CUDA_ARCH) $(STD_FLAG) $(QUIET_WARN)

# Release / Debug
REL_FLAGS     := -O3 -lineinfo
DBG_FLAGS     := -G -g -O0

# Optimized flags
OPT_FLAGS     := -O3 -use_fast_math -Xptxas -O3,-dlcm=ca -DNDEBUG -lineinfo \
                 -Xcompiler -fno-exceptions -Xcompiler -fno-rtti

# Default CFLAGS to release if not specified
CFLAGS        ?= $(BASE_CFLAGS) $(REL_FLAGS)

# --- Phony Targets ---
.PHONY: all release debug optimized optimized-test optimized-evaluate test evaluate clean clean-all help

# Default target
all: help

# Standard builds
release: CFLAGS := $(BASE_CFLAGS) $(REL_FLAGS)
release: $(MAIN_TARGET)

debug: CFLAGS := $(BASE_CFLAGS) $(DBG_FLAGS)
debug: $(MAIN_TARGET)

# Tests
test: CFLAGS := $(BASE_CFLAGS) $(DBG_FLAGS)
test: $(TEST_TARGET)

evaluate: CFLAGS := $(BASE_CFLAGS) $(DBG_FLAGS)
evaluate: $(EVAL_TARGET)

# Optimized builds
optimized: CFLAGS := $(BASE_CFLAGS) $(OPT_FLAGS)
optimized: $(MAIN_TARGET)

optimized-test: CFLAGS := $(BASE_CFLAGS) $(OPT_FLAGS)
optimized-test: $(TEST_TARGET)

optimized-evaluate: CFLAGS := $(BASE_CFLAGS) $(OPT_FLAGS)
optimized-evaluate: $(EVAL_TARGET)

# --- Build Rules ---

# Ensure all object files are rebuilt when any header changes.
# (This is a simple global header dependency rule; it's robust and portable.)
$(ENGINE_OBJS) main.o test_main.o eval_test.o: $(DEPS)

# Main engine build rule
$(MAIN_TARGET): $(filter-out main.o, $(ENGINE_OBJS)) main.o $(THRUST_OBJ)
	@echo "Linking main engine: $(MAIN_TARGET)..."
	$(NVCC) $(CFLAGS) -o $@ $^

# Test engine build rule
$(TEST_TARGET): $(filter-out main.o, $(ENGINE_OBJS)) test_main.o $(THRUST_OBJ)
	@echo "Linking test engine: $(TEST_TARGET)..."
	$(NVCC) $(CFLAGS) -o $@ $^

# Evaluation test build rule
$(EVAL_TARGET): $(ENGINE_OBJS) eval_test.o $(THRUST_OBJ)
	@echo "Linking evaluation test: $(EVAL_TARGET)..."
	$(NVCC) $(CFLAGS) -o $@ $^

# Generic compile rule for all .cu → .o
%.o: %.cu
	@echo "Compiling $<..."
	$(NVCC) $(CFLAGS) -dc -c $< -o $@

# --- Clean Rules ---

clean:
	@echo "Cleaning project (excluding thrust_precompiled.o)..."
	rm -f $(ENGINE_OBJS) main.o test_main.o eval_test.o $(MAIN_TARGET) $(TEST_TARGET) $(EVAL_TARGET) *.d

clean-all: clean
	@echo "Cleaning all (including thrust wrapper)..."
	rm -f $(THRUST_OBJ)

# --- Help Message ---

help:
	@echo "====================================================================="
	@echo "  CUDA Chess Engine Makefile"
	@echo "====================================================================="
	@echo "  Usage: make [target]"
	@echo ""
	@echo "  Targets:"
	@echo "    help                 - Show this help message."
	@echo "    release              - Build '$(MAIN_TARGET)' with -O3."
	@echo "    debug                - Build '$(MAIN_TARGET)' with debug flags."
	@echo "    optimized            - Build '$(MAIN_TARGET)' with aggressive GPU/PTX opts."
	@echo "    test                 - Build '$(TEST_TARGET)' (debug flags)."
	@echo "    evaluate             - Build '$(EVAL_TARGET)' (debug flags)."
	@echo "    optimized-test       - Build '$(TEST_TARGET)' with optimized flags."
	@echo "    optimized-evaluate   - Build '$(EVAL_TARGET)' with optimized flags."
	@echo "    clean                - Remove objects and executables (keep thrust obj)."
	@echo "    clean-all            - Remove all including '$(THRUST_OBJ)'."
	@echo ""
	@echo "  Examples:"
	@echo "    make optimized"
	@echo "    make optimized-evaluate"
	@echo "====================================================================="
