#include "llama-qwengram.h"

#include "llama-batch.h"
#include "llama-mmap.h"

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

static uint64_t splitmix64(uint64_t v) {
    v += 0x9e3779b97f4a7c15ULL;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

static bool is_prime(uint64_t v) {
    if (v < 2 || v % 2 == 0) return v == 2;
    for (uint64_t d = 3; d * d <= v; d += 2) {
        if (v % d == 0) return false;
    }
    return true;
}

struct llama_qwengram_ple::impl {
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llama_mmap> map;
    std::array<uint64_t, 3> mult;
    std::array<uint64_t, 16> offsets;
    std::array<uint64_t, 16> sizes;
    const uint8_t * rows = nullptr;
    size_t row_bytes = 0;
    int64_t row_count = 0;
    ggml_to_float_t dequantize = nullptr;

    explicit impl(const std::string & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_from_file(path.c_str(), params), gguf_free);
        if (!ctx) throw std::runtime_error("QwenGram: invalid PLE GGUF");

        const int64_t weight = gguf_find_tensor(ctx.get(), "ple.weight");
        const int64_t mult_id = gguf_find_tensor(ctx.get(), "ple.layer_multipliers");
        const int64_t off_id = gguf_find_tensor(ctx.get(), "ple.ngram_heads_offsets");
        const int64_t size_id = gguf_find_tensor(ctx.get(), "ple.ngram_heads_vocab_sizes");
        if (weight < 0 || mult_id < 0 || off_id < 0 || size_id < 0) {
            throw std::runtime_error("QwenGram: PLE tensors missing");
        }
        const auto * ne = gguf_get_tensor_ne(ctx.get(), weight);
        if (gguf_get_tensor_type(ctx.get(), weight) != GGML_TYPE_Q4_1 || ne[0] != 160 || ne[1] != 320001536) {
            throw std::runtime_error("QwenGram: unexpected PLE row layout");
        }
        row_count = ne[1];
        row_bytes = 160 / ggml_blck_size(GGML_TYPE_Q4_1) * ggml_type_size(GGML_TYPE_Q4_1);
        const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_1);
        if (!traits || !traits->to_float) throw std::runtime_error("QwenGram: Q4_1 dequantization unavailable");
        dequantize = traits->to_float;

        file = std::make_unique<llama_file>(path.c_str(), "rb");
        map = std::make_unique<llama_mmap>(file.get(), 0, true);
        const auto * base = static_cast<const uint8_t *>(map->addr()) + gguf_get_data_offset(ctx.get());
        auto read_i64 = [&](int64_t id, int64_t * dst, size_t count) {
            if (gguf_get_tensor_type(ctx.get(), id) != GGML_TYPE_I64 || gguf_get_tensor_ne(ctx.get(), id)[0] != (int64_t) count) {
                throw std::runtime_error("QwenGram: invalid PLE address tensor");
            }
            const size_t off = gguf_get_data_offset(ctx.get()) + gguf_get_tensor_offset(ctx.get(), id);
            if (off + count * sizeof(int64_t) > map->size()) throw std::runtime_error("QwenGram: truncated PLE address tensor");
            std::memcpy(dst, static_cast<const uint8_t *>(map->addr()) + off, count * sizeof(int64_t));
        };
        std::array<int64_t, 3> raw_mult;
        std::array<int64_t, 16> raw_off;
        std::array<int64_t, 16> raw_size;
        read_i64(mult_id, raw_mult.data(), raw_mult.size());
        read_i64(off_id, raw_off.data(), raw_off.size());
        read_i64(size_id, raw_size.data(), raw_size.size());

        const uint64_t mmax = std::numeric_limits<int64_t>::max() / 248320;
        const uint64_t half = mmax / 2;
        for (int i = 0; i < 3; ++i) {
            mult[i] = 2 * (splitmix64(1234 + 0x9e3779b97f4a7c15ULL * (i + 1)) % half) + 1;
            if (raw_mult[i] != (int64_t) mult[i]) throw std::runtime_error("QwenGram: PLE hash multipliers differ");
        }
        uint64_t prime = 19999999;
        uint64_t next_off = 0;
        for (int i = 0; i < 16; ++i) {
            do { ++prime; } while (!is_prime(prime));
            sizes[i] = prime;
            offsets[i] = next_off;
            if (raw_size[i] != (int64_t) sizes[i] || raw_off[i] != (int64_t) offsets[i]) {
                throw std::runtime_error("QwenGram: PLE head layout differs");
            }
            next_off += prime;
        }
        if (next_off > (uint64_t) row_count) throw std::runtime_error("QwenGram: PLE rows too short");

        const size_t row_off = gguf_get_data_offset(ctx.get()) + gguf_get_tensor_offset(ctx.get(), weight);
        if (row_off + gguf_get_tensor_size(ctx.get(), weight) > map->size()) {
            throw std::runtime_error("QwenGram: truncated PLE weights");
        }
        rows = base + gguf_get_tensor_offset(ctx.get(), weight);
    }

    void row(uint64_t address, float * dst) const {
        dequantize(rows + address * row_bytes, dst, 160);
    }
};

llama_qwengram_ple::llama_qwengram_ple(const std::string & path) : pimpl(std::make_unique<impl>(path)) {}
llama_qwengram_ple::~llama_qwengram_ple() = default;

void llama_qwengram_ple::fill(const llama_ubatch & ubatch, llama_qwengram_state & state, float * dst) const {
    if (!ubatch.token || !ubatch.pos || !ubatch.seq_id) {
        throw std::runtime_error("QwenGram requires token IDs, positions, and sequence IDs");
    }
    constexpr llama_token eos = 248044;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos[i * ubatch.n_pos];
        for (int j = 0; j < ubatch.n_seq_id[i]; ++j) {
            auto & history = state.tokens[ubatch.seq_id[i][j]];
            if (pos == 0) history.clear();
            history[pos] = ubatch.token[i];
        }
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos[i * ubatch.n_pos];
        const auto & history = state.tokens.at(ubatch.seq_id[i][0]);
        const auto prev = history.find(pos - 1);
        const auto prev2 = history.find(pos - 2);
        const uint64_t t0 = ubatch.token[i];
        const uint64_t t1 = prev == history.end() || prev->second == eos ? eos : prev->second;
        const uint64_t t2 = t1 == eos || prev2 == history.end() || prev2->second == eos ? eos : prev2->second;
        const uint64_t h2 = (t0 * pimpl->mult[0]) ^ (t1 * pimpl->mult[1]);
        const uint64_t h3 = h2 ^ (t2 * pimpl->mult[2]);
        for (int head = 0; head < 16; ++head) {
            const int64_t mixed = (int64_t) (head < 8 ? h2 : h3);
            const int64_t size = (int64_t) pimpl->sizes[head];
            const uint64_t address = pimpl->offsets[head] + (mixed % size + size) % size;
            pimpl->row(address, dst + i * 2560 + head * 160);
        }
    }
}
