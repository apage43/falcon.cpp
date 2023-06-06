#include "ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <signal.h>
#endif

struct falcon_hparams {
    int32_t n_vocab = 65024;
    int32_t n_ctx = 512;  // this is provided as user input?
    int32_t n_embd = 8192;
    int32_t n_head = 128;
    int32_t n_layer = 60;
    int32_t f16 = 1;
};

struct falcon_layer {
    // normalization
    struct ggml_tensor* attention_norm;
    struct ggml_tensor* attention_norm_b;

    // attention
    struct ggml_tensor* query_key_value;
    struct ggml_tensor* wo;

    // ff
    struct ggml_tensor* ffn_up;
    struct ggml_tensor* ffn_down;
};

struct falcon_model {
    falcon_hparams hparams;

    struct ggml_tensor* tok_embeddings;
    struct ggml_tensor* output_norm;
    struct ggml_tensor* output_norm_b;
    struct ggml_tensor* lm_head;

    std::vector<falcon_layer> layers;

    // key + value memory
    struct ggml_tensor* memory_k;
    struct ggml_tensor* memory_v;

    struct ggml_context* ctx;
    std::map<std::string, struct ggml_tensor*> tensors;
};

// load the model's weights from a file
bool falcon_model_load(const std::string& fname,
                       falcon_model& model,
                       gpt_vocab& vocab,
                       int n_ctx) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__,
           fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char*)&magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n",
                    __func__, fname.c_str());
            return false;
        }
    }

    int n_ff = 0;
    int n_parts = 1;

    // load hparams
    {
        auto& hparams = model.hparams;

        fin.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char*)&hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char*)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char*)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char*)&hparams.f16, sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;
        n_ff = 4 * hparams.n_embd;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
        printf("%s: n_ff    = %d\n", __func__, n_ff);
        printf("%s: n_parts = %d\n", __func__, n_parts);
    }

    // load vocab
    {
        const int32_t n_vocab = model.hparams.n_vocab;

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr,
                    "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char*)&len, sizeof(len));

            word.resize(len);
            fin.read((char*)word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            // if (i < 30000) {
            //     printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
            // }
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit
    // floats or quantized in order to save memory and also to speed up the
    // computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16) {
        case 0:
            wtype = GGML_TYPE_F32;
            break;
        case 1:
            wtype = GGML_TYPE_F16;
            break;
        case 2:
            wtype = GGML_TYPE_Q4_0;
            break;
        case 3:
            wtype = GGML_TYPE_Q4_1;
            break;
        default: {
            fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                    __func__, fname.c_str(), model.hparams.f16);
            return false;
        }
    }

    const ggml_type wtype2 = GGML_TYPE_F32;

    auto& ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_head = hparams.n_head;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size +=
            n_embd * n_vocab * ggml_type_sizef(wtype);  // tok_embeddings

        // ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // norm
        // ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // norm_b

        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32);  // output_norm
        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32);  // output_norm_b

        ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype);  // lm_head

        ctx_size +=
            n_layer *
            (n_embd * ggml_type_sizef(GGML_TYPE_F32));  // attention_norm
        ctx_size +=
            n_layer *
            (n_embd * ggml_type_sizef(GGML_TYPE_F32));  // attention_norm_b

        ctx_size += n_layer * (n_embd * (n_embd + 2 * (n_embd / n_head)) *
                               ggml_type_sizef(wtype));  // query_key_value
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype));  // wo

        ctx_size +=
            n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));  // ffn_norm
        ctx_size +=
            n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32));  // ffn_norm_b

        ctx_size +=
            n_layer * (n_ff * n_embd * ggml_type_sizef(wtype));  // ffn_up
        ctx_size +=
            n_layer * (n_ff * n_embd * ggml_type_sizef(wtype));  // ffn_down

        ctx_size += n_ctx * n_layer * n_embd *
                    ggml_type_sizef(GGML_TYPE_F32);  // memory_k
        ctx_size += n_ctx * n_layer * n_embd *
                    ggml_type_sizef(GGML_TYPE_F32);  // memory_v

        ctx_size += (5 + 10 * n_layer) * 256;  // object overhead TODO:

        printf("%s: ggml ctx size = %6.2f MB\n", __func__,
               ctx_size / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_head = hparams.n_head;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.tok_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        model.output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.output_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.lm_head = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        // map by name
        model.tensors["transformer.word_embeddings.weight"] =
            model.tok_embeddings;

        model.tensors["transformer.ln_f.weight"] = model.output_norm;
        model.tensors["transformer.ln_f.bias"] = model.output_norm_b;
        model.tensors["lm_head.weight"] = model.lm_head;

        for (int i = 0; i < n_layer; ++i) {
            auto& layer = model.layers[i];

            layer.attention_norm =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.attention_norm_b =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // query_key_value shape for config.multi_query == True:
            layer.query_key_value = ggml_new_tensor_2d(
                ctx, wtype, n_embd, n_embd + 2 * (n_embd / n_head));
            layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ffn_up = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ff);
            layer.ffn_down = ggml_new_tensor_2d(ctx, wtype, n_ff, n_embd);

            // map by name
            model.tensors["transformer.h." + std::to_string(i) +
                          ".input_layernorm.weight"] = layer.attention_norm;
            model.tensors["transformer.h." + std::to_string(i) +
                          ".input_layernorm.bias"] = layer.attention_norm_b;

            model.tensors["transformer.h." + std::to_string(i) +
                          ".self_attention.query_key_value.weight"] =
                layer.query_key_value;
            model.tensors["transformer.h." + std::to_string(i) +
                          ".self_attention.dense.weight"] = layer.wo;

            model.tensors["transformer.h." + std::to_string(i) +
                          ".mlp.dense_h_to_4h.weight"] = layer.ffn_up;
            model.tensors["transformer.h." + std::to_string(i) +
                          ".mlp.dense_4h_to_h.weight"] = layer.ffn_down;
        }
    }

    // key + value memory
    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;

        const int n_mem = n_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size =
            ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__,
               memory_size / 1024.0 / 1024.0, n_mem);
    }

    const size_t file_offset = fin.tellg();

    fin.close();

    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    fin = std::ifstream(fname, std::ios::binary);
    fin.seekg(file_offset);

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char*>(&length), sizeof(length));
            fin.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n",
                        __func__, name.data());
                return false;
            }

            // split_type = 0: split by columns
            // split_type = 1: split by rows
            int split_type = 0;

            if (name.find("word_embeddings") != std::string::npos) {
                split_type = 0;
            } else if (name.find("layers") != std::string::npos) {
                if (name.find("query_key_value") != std::string::npos) {
                    split_type = 0;
                } else if (name.find("dense_h_to_4h") != std::string::npos) {
                    split_type = 0;
                } else {
                    split_type = 1;
                }
            } else if (name.find("lm_head") != std::string::npos) {
                split_type = 1;
            }

            auto tensor = model.tensors[name.data()];

            if (n_dims == 1) {
                printf("%s = %d\n", name.data(), ne[0]);
                if (ggml_nelements(tensor) != nelements) {
                    fprintf(stderr,
                            "%s: tensor '%s' has wrong size in model file\n",
                            __func__, name.data());
                    return false;
                }
            } else {
                printf("%s = %d x %d\n", name.data(), ne[0], ne[1]);
                if (ggml_nelements(tensor) / n_parts != nelements) {
                    fprintf(stderr,
                            "%s: tensor '%s' has wrong size in model file\n",
                            __func__, name.data());
                    return false;
                }
            }

            if (n_dims == 1) {
                if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                    fprintf(stderr,
                            "%s: tensor '%s' has wrong shape in model file: "
                            "got [%li, "
                            "%li], expected [%d, %d]\n",
                            __func__, name.data(), tensor->ne[0], tensor->ne[1],
                            ne[0], ne[1]);
                    return false;
                }
            } else {
                if (split_type == 0) {
                    if (tensor->ne[0] / n_parts != ne[0] ||
                        tensor->ne[1] != ne[1]) {
                        fprintf(stderr,
                                "%s: tensor '%s' has wrong shape in model "
                                "file: got [%li, "
                                "%li], expected [%d, %d]\n",
                                __func__, name.data(), tensor->ne[0] / n_parts,
                                tensor->ne[1], ne[0], ne[1]);
                        return false;
                    }
                } else {
                    if (tensor->ne[0] != ne[0] ||
                        tensor->ne[1] / n_parts != ne[1]) {
                        fprintf(stderr,
                                "%s: tensor '%s' has wrong shape in model "
                                "file: got [%li, "
                                "%li], expected [%d, %d]\n",
                                __func__, name.data(), tensor->ne[0],
                                tensor->ne[1] / n_parts, ne[0], ne[1]);
                        return false;
                    }
                }
            }

            if (0) {
                static const char* ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%24s - [%5d, %5d], type = %6s, split = %d\n",
                       name.data(), ne[0], ne[1], ftype_str[ftype], split_type);
            }

            size_t bpe = 0;

            switch (ftype) {
                case 0:
                    bpe = ggml_type_size(GGML_TYPE_F32);
                    break;
                case 1:
                    bpe = ggml_type_size(GGML_TYPE_F16);
                    break;
                case 2:
                    bpe = ggml_type_size(GGML_TYPE_Q4_0);
                    assert(ne[0] % 64 == 0);
                    break;
                case 3:
                    bpe = ggml_type_size(GGML_TYPE_Q4_1);
                    assert(ne[0] % 64 == 0);
                    break;
                default: {
                    fprintf(stderr, "%s: unknown ftype %d in model file\n",
                            __func__, ftype);
                    return false;
                }
            };

            if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
                ggml_nbytes(tensor)) {
                fprintf(
                    stderr,
                    "%s: tensor '%s' has wrong size in model file: got %zu, "
                    "expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor),
                    nelements * bpe);
                return false;
            }

            fin.read(reinterpret_cast<char*>(tensor->data),
                     ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);

            // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(),
            // ne[0], ne[1], ftype == 0 ? "float" : "f16",
            // ggml_nbytes(tensor)/1024.0/1024.0);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__,
               total_size / 1024.0 / 1024.0, n_tensors);

        fin.close();
    }

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool falcon_eval(const falcon_model& model,
                 const int n_threads,
                 const int n_past,
                 const std::vector<gpt_vocab::id>& embd_inp,
                 std::vector<float>& embd_w) {
    const int N = embd_inp.size();

    const auto& hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;
    const int n_vocab = hparams.n_vocab;

    const size_t head_dim = n_embd / n_head;

    static size_t buf_size = 512u * 1024 * 1024;
    static void* buf = malloc(buf_size);

    struct ggml_init_params params = {
        .mem_size = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context* ctx0 = ggml_init(params);
    ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N * ggml_element_size(embd));

    struct ggml_tensor* inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);
    struct ggml_tensor* repeat_dummy = ggml_new_tensor_3d(ctx0, inpL->type, head_dim, N + n_past, n_head);

    for (int il = 0;
         il < 1 /*TODO: replace 1 with n_layer after porting complete! */;
         ++il) {
        struct ggml_tensor* residual = inpL;  // TODO: copy?

        struct ggml_tensor* cur;

        // layernorm_output = self.input_layernorm(hidden_states)
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_mul(
                ctx0, ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                cur);
            cur = ggml_add(
                ctx0, ggml_repeat(ctx0, model.layers[il].attention_norm_b, cur),
                cur);
        }

        struct ggml_tensor* layernorm_output = cur;

        // fused_qkv = self.query_key_value(hidden_states)
        {
            cur = ggml_mul_mat(ctx0, model.layers[il].query_key_value, cur);

            // cur = ggml_add(ctx0,
            //         ggml_repeat(ctx0, model.layers[il].query_key_value_b,
            //         cur), cur);
        }

        // cur = ggml_debug(ctx0, cur);

        // self-attention
        {
            // fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads
            // + 2, self.head_dim)

            // struct ggml_tensor * fused_qkv_view = ggml_view_3d(ctx0, cur,
            //     n_embd/n_head, n_head+2, N,
            //     n_embd/n_head * sizeof(float), n_embd + 2 * (n_embd / n_head)
            //     * sizeof(float), 0);

            size_t fused_qkv_row_nb =
                (n_embd + 2 * (n_embd / n_head)) * sizeof(float);

            // (query_layer, key_layer, value_layer) =
            // self._split_heads(fused_qkv)

            struct ggml_tensor* Qcur =
                ggml_view_3d(ctx0, cur, head_dim, n_head, N,
                             head_dim * sizeof(float), fused_qkv_row_nb, 0);

            struct ggml_tensor* Kcur = ggml_view_3d(
                ctx0, cur, head_dim, 1, N, head_dim * sizeof(float),
                fused_qkv_row_nb, n_embd * sizeof(float));

            struct ggml_tensor* Vcur = ggml_view_3d(
                ctx0, cur, head_dim, 1, N, head_dim * sizeof(float),
                fused_qkv_row_nb, (n_embd + head_dim) * sizeof(float));

            // using mode = 2 for GPT-NeoX mode
            Qcur = ggml_rope_inplace(ctx0, Qcur, n_past, head_dim, 2);
            Kcur = ggml_rope_inplace(ctx0, Kcur, n_past, head_dim, 2);

            // Example of how we can dump a tensor (Vcur) to stdout:
            //    ggml_build_forward_expand(&gf, Vcur);
            //    ggml_graph_compute(ctx0, &gf);
            //    ggml_print_tensor_f32(Vcur);

            // TODO: verified to match layer 0 data from falcon/modelling_RW.py
            // up to this point That is, we stand here just before query_layer =
            // query_layer.transpose(1, 2).reshape from falcon/modelling_RW.py

            // store key and value to memory
            {
                // k,v tensors are just the size of a single head when using
                // multiquery attention

                struct ggml_tensor* k = ggml_view_1d(
                    ctx0, model.memory_k, N * head_dim,
                    (ggml_element_size(model.memory_k) * head_dim) *
                        (il * n_ctx + n_past));
                struct ggml_tensor* v = ggml_view_1d(
                    ctx0, model.memory_v, N * head_dim,
                    (ggml_element_size(model.memory_v) * head_dim) *
                        (il * n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // TODO: UNVERIFIED from here on:

            struct ggml_tensor* Q = ggml_permute(
                ctx0,
                ggml_cpy(ctx0, Qcur,
                         ggml_new_tensor_3d(ctx0, GGML_TYPE_F32,
                                            n_embd / n_head, n_head, N)),
                0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1,
            // 3)
            struct ggml_tensor* K = ggml_permute(
                ctx0,
                ggml_reshape_3d(
                    ctx0,
                    ggml_view_1d(ctx0, model.memory_k, (n_past + N) * head_dim,
                                 il * n_ctx *
                                     ggml_element_size(model.memory_k) *
                                     head_dim),
                    head_dim, 1, n_past + N),
                0, 2, 1, 3);

            K = ggml_repeat(ctx0, K, repeat_dummy);
            // K * Q
            struct ggml_tensor* KQ =
                ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor* KQ_scaled = ggml_scale(
                ctx0, KQ,
                ggml_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head)));

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor* KQ_masked =
                ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor* KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            struct ggml_tensor* V = ggml_permute(
                ctx0,
                ggml_reshape_3d(
                    ctx0,
                    ggml_view_1d(ctx0, model.memory_v, (n_past + N) * head_dim,
                                 il * n_ctx *
                                     ggml_element_size(model.memory_v) *
                                     head_dim),
                    head_dim, 1, n_past + N),
                0, 2, 1, 3);

            V = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_repeat(ctx0, V, repeat_dummy)));

            struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor* KQV_merged =
                ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0, KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection
            cur = ggml_mul_mat(ctx0, model.layers[il].wo, cur);
            // cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].wo_b,
            // cur), cur);
        }

        // struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        struct ggml_tensor* inpFF = layernorm_output;
        struct ggml_tensor* attn_out = ggml_cpy(
            ctx0, cur, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

        // feed-forward network
        {
            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_up, inpFF);
            // cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].w1_b,
            // cur), cur);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_mul_mat(ctx0, model.layers[il].ffn_down, cur);
            // cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].w2_b,
            // cur), cur);
        }

        // cur  = ggml_add(ctx0, cur, inpFF);
        cur = ggml_add(ctx0, cur, attn_out);
        cur = ggml_add(ctx0, cur, inpL);

        // input for next layer
        inpL = cur;
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);
        // inpL = norm*inpL
        inpL = ggml_mul(ctx0, ggml_repeat(ctx0, model.output_norm, inpL), inpL);

        inpL =
            ggml_add(ctx0, ggml_repeat(ctx0, model.output_norm_b, inpL), inpL);
    }

    // lm_head
    { inpL = ggml_mul_mat(ctx0, model.lm_head, inpL); }

    // logits -> probs
    // inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute(ctx0, &gf);

    // if (n_past%100 == 0) {
    //     ggml_graph_print   (&gf);
    //     ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    // }

    // embd_w.resize(n_vocab*N);
    // memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float*)ggml_get_data(inpL) + (n_vocab * (N - 1)),
           sizeof(float) * n_vocab);
    // printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main(int argc, char** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "./models/ggml-model-gpt4all-falcon-f16.bin";
    params.prompt = "The best part of waking up";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    //    params.prompt = R"(// this function checks if the number n is prime
    // bool is_prime(int n) {)";

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    falcon_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        if (!falcon_model_load(params.model, model, vocab, params.n_ctx)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
                    params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::bloom_tokenize(
        vocab, params.prompt, false);  // TODO: set bos to true?

    params.n_predict =
        std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

    printf("\n");
    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < (int)embd_inp.size(); i++) {
        printf("%6d -> '%s'\n", embd_inp[i],
               vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    printf("\n");
    printf(
        "sampling parameters: temp = %f, top_k = %d, top_p = %f, "
        "repeat_last_n = %i, repeat_penalty = %f\n",
        params.temp, params.top_k, params.top_p, params.repeat_last_n,
        params.repeat_penalty);
    printf("\n\n");

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    // size_t mem_per_token = 0;
    // falcon_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits,
    // mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!falcon_eval(model, params.n_threads, n_past, embd,
                             logits)) {  // update logits
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const float top_p = params.top_p;
            const float temp = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = bloom_sample_top_p(
                    vocab, logits.data() + (logits.size() - n_vocab),
                    last_n_tokens, repeat_penalty, top_p, temp, rng);

                // // print
                // printf("\ngenerated token: '%s' (%d)\n",
                // vocab.id_to_token[id].c_str(), id);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 2) {
            printf(" [end of text]\n");
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__,
               t_sample_us / 1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__,
               t_predict_us / 1000.0f, t_predict_us / 1000.0f / n_past);
        printf("%s:    total time = %8.2f ms\n", __func__,
               (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
