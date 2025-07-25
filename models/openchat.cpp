#include "openchat.h"

namespace chatllm::openchat
{
    class ChatHistoryEncoder : public BaseHistoryEncoder
    {
    public:
        void append_sys_prompt(std::vector<int> &ids) const override;
        void append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const override;
        void append_user(int round_idx, const std::string &user, std::vector<int> &ids) const override;
        void append_ai_opening(int round_idx, std::vector<int> &ids) const override;
    };

    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config &config)
        : Tokenizer(config, &_chat_encoder)
    {}

    Tokenizer::Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : mistral::mistral::Tokenizer(config, encoder)
    {
        sys_prompt = "GPT4";
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config)
        : ConditionalGeneration(config, runtime_config, MODEL_TYPE_OPENCHAT)
    {
    }

    ConditionalGeneration::ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type)
        : mistral::mistral::ConditionalGeneration(config, runtime_config, type)
    {
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string &ai, std::vector<int> &ids) const
    {
        // {{ bos_token }}
        // {% for message in messages %}
        //      {{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}
        // {% endfor %}
        // {% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids, false, true);
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

        oss_prompt << tok->get_system_prompt() << " Correct User: " << user;
        tok->encode(oss_prompt.str(), ids, false, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int> &ids) const
    {
        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        std::ostringstream oss_prompt;

        oss_prompt << tok->get_system_prompt() << " Correct Assistant: ";
        tok->encode(oss_prompt.str(), ids, false, false);
    }

    REGISTER_MODEL_LOADER(OPENCHAT,              openchat, 1);
}