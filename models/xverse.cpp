#include "llama.h"
#include "deepseek.h"

namespace chatllm::xverse::dense
{
typedef llama::v2::Config Config;

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "";
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::BPEProcessor3(
            {
                "[0-9]",
            }
        );
        size_t size = tp->Load(buffer, n_vocab);
        return size;
    }

    void encode(const std::string &text, std::vector<int> &ids) const override
    {
        encode(text, ids, false, false);
    }

public:
    virtual void encode(const std::string &text, std::vector<int> &ids, bool add_bos, bool add_eos) const
    {
        if (add_bos)
            ids.push_back(bos_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_eos)
            ids.push_back(eos_token_id);
    }
};

class ConditionalGeneration : public llama::v2::GenericConditionalGeneration<LlamaBlock>
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_XVERSE)
        : ConditionalGeneration(config, runtime_config, type, config.num_attention_heads, config.max_length)
    {}

    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type,
                        int num_key_value_heads, int max_length)
        : GenericConditionalGeneration<LlamaBlock>(config, runtime_config, type, num_key_value_heads, max_length)
    {}
};

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids, false, true);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    //if (round_idx == 0)
    //    ids.push_back(tok->bos_token_id);

    oss_prompt << "Human: " << user << "\n\n";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, false);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    //if (round_idx == 0)
    //    ids.push_back(tok->bos_token_id);

    oss_prompt << "Assistant: ";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, false);
}
}

namespace chatllm::xverse::moe
{
    typedef deepseek::v1_moe::Config Config;

    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override
        {
            auto *tok = dynamic_cast<dense::Tokenizer *>(tokenizer);
            if (tok->get_system_prompt().size() < 1) return;

            std::ostringstream oss_prompt;
            oss_prompt << "system: " << tok->get_system_prompt() << "\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids);
        }

        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override
        {
            auto *tok = dynamic_cast<dense::Tokenizer *>(tokenizer);
            append_ai_opening(round_idx, ids);
            tok->encode(ai, ids);
            ids.push_back(tok->eos_token_id);
        }

        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override
        {
            auto *tok = dynamic_cast<dense::Tokenizer *>(tokenizer);

            std::ostringstream oss_prompt;
            oss_prompt << "user: " << user << "\n";
            auto text = oss_prompt.str();
            tok->encode(text, ids);
        }

        void append_ai_opening(int round_idx, std::vector<int> &ids) const override
        {
            auto *tok = dynamic_cast<dense::Tokenizer *>(tokenizer);
            tok->encode("assistant: ", ids);
        }
    };

    static ChatHistoryEncoder _chat_encoder;

    class Tokenizer : public dense::Tokenizer
    {
    public:
        Tokenizer(const Config &config)
            : dense::Tokenizer(config, &_chat_encoder)
        {}
    }
    typedef  Tokenizer;

    typedef deepseek::v1_moe::ConditionalGeneration ConditionalGeneration;
}

namespace chatllm
{
    REGISTER_MODEL_LOADER(XVERSEMOE,             xverse::moe, 1);
    REGISTER_MODEL_LOADER(XVERSE,                xverse::dense, 1);
}