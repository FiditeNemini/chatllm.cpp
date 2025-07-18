#include "deepseek.h"

namespace chatllm::numinamath
{
struct Config : public deepseek::v1::Config
{
};

class ChatHistoryEncoder : public BaseHistoryEncoder
{
public:
    void append_sys_prompt(std::vector<int> &ids) const override;
    void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
    void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
    void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
};

static ChatHistoryEncoder _chat_encoder;

class Tokenizer : public deepseek::v1::Tokenizer
{
public:
    Tokenizer(const Config &config)
        : deepseek::v1::Tokenizer(config, &_chat_encoder)
    {
        sys_prompt = "";
    }
};

void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    append_ai_opening(round_idx, ids);
    tok->encode(ai, ids, false, false);
}

void ChatHistoryEncoder::append_sys_prompt(std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    ids.push_back(tok->bos_token_id);
}

void ChatHistoryEncoder::append_user(int round_idx, const std::string &user, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    oss_prompt << "### Problem: " << user << "\n";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, false);
}

void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
{
    Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);
    std::ostringstream oss_prompt;

    oss_prompt << "### Solution: ";

    auto text = oss_prompt.str();
    tok->encode(text, ids, false, false);
}

class ConditionalGeneration : public llama::v2::ConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : llama::v2::ConditionalGeneration(config, runtime_config, MODEL_TYPE_NUMINAMATH)
    {
    }
};

REGISTER_MODEL_LOADER(NUMINAMATH,            numinamath, 1);
}