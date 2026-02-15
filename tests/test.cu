#include "tests.h"

TestStats g_stats;

int main(int argc, char** argv) {
    bool do_correct = true;
    bool do_speed   = true;
    bool do_weight  = true;

    // 简单参数解析
    for (int i=1;i<argc;i++){
        std::string arg = argv[i];
        if (arg == "--correct-only") { do_speed = false; }
        else if (arg == "--speed-only") { do_correct = false; do_weight = false; }
        else if (arg == "--skip-load") { do_weight = false; }
        else if (arg == "--skip-speed") { do_speed = false; }
        else if (arg == "--skip-correct") { do_correct = false; }
    }

    std::cout << "=== CUDA Test Runner ===" << std::endl;
    std::cout << "correct=" << do_correct
              << " speed=" << do_speed
              << " load=" << do_weight << std::endl;

    test_linear(do_correct, do_speed);
    test_rmsnorm(do_correct, do_speed);
    test_rope(do_correct, do_speed);
    test_attn(do_correct, do_speed);
    test_embedding(do_correct, do_speed);
    test_attentionblock(do_correct, do_speed);
    test_swishglu(do_correct, do_speed);
    test_transformerblock(do_correct, do_speed);
    test_transformer(do_correct, do_speed, do_weight);

    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Passed: " << g_stats.passed
              << " Failed: " << g_stats.failed
              << " Skipped: " << g_stats.skipped << std::endl;

    return (g_stats.failed == 0) ? 0 : 1;
}
