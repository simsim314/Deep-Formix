CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -I./eigen -I. -pthread
LDFLAGS =
SOURCES = engine.cpp estimator.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = nnue32_engine
all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
%.o: %.cpp estimator.hpp chess.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
.PHONY: all clean
