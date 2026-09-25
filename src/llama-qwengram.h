#pragma once

#include "llama.h"

#include <memory>
#include <string>
#include <unordered_map>

struct llama_ubatch;

struct llama_qwengram_state {
    std::unordered_map<llama_seq_id, std::unordered_map<llama_pos, llama_token>> tokens;
};

class llama_qwengram_ple {
public:
    explicit llama_qwengram_ple(const std::string & path);
    ~llama_qwengram_ple();

    void fill(const llama_ubatch & ubatch, llama_qwengram_state & state, float * dst) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
