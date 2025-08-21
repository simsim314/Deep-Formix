// src/uci.cuh
#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <string>

namespace UCI {

// Declare global variables for network paths, accessible by other parts of the engine.
extern std::string g_nnue_model_path;
extern std::string g_nnue_mapping_path;

void loop(int argc, char** argv);

}  // namespace UCI

#endif  // #ifndef UCI_H_INCLUDED
