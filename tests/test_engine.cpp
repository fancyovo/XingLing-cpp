#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include "inference.h"
#include <xmmintrin.h> // 必须引入这个
#include <pmmintrin.h>

// 计时辅助工具
class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double stop() { // 返回毫秒
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

int main() {
    _mm_setcsr(_mm_getcsr() | 0x8040);
    std::cout << "========== Inference Engine Benchmark ==========" << std::endl;

    // 1. 初始化引擎
    std::cout << "[1/4] Initializing InferenceEngine..." << std::endl;
    InferenceEngine engine;

    // 2. 加载模型
    // 假设编译出的可执行文件在 build/ 下，模型在 ../data/model
    std::string model_path = "../data/model"; 
    std::cout << "[2/4] Loading model from: " << model_path << std::endl;
    try {
        Timer load_timer;
        load_timer.start();
        engine.load(model_path);
        std::cout << "      Model loaded successfully. Time: " << load_timer.stop() / 1000.0 << " s" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "      [FATAL] Failed to load model: " << e.what() << std::endl;
        return -1;
    }

    // 3. 构造输入 (模拟 User Prompt)
    // 随意构造一些 Token ID，长度建议不要太短，以便测试 Prefill 性能
    // 这里构造一个长度为 32 的序列
    std::vector<int> input_ids = { engine.im_start_id, engine.role_id_user };
    for(int i = 0; i < 30; ++i) {
        input_ids.push_back(100 + i); // 随机填充一些常用词ID
    }
    input_ids.push_back(engine.im_end_id);
    input_ids.push_back(engine.role_id_assistant); // 触发生成

    std::cout << "[3/4] Starting Inference Test..." << std::endl;
    std::cout << "      Input Length: " << input_ids.size() << " tokens" << std::endl;

    // 4. Prefill 阶段测试 (计算 prompt 的 logits)
    Timer timer;
    timer.start();
    
    // 获取第一个预测 token
    int next = engine.next_token(input_ids);
    
    double prefill_time_ms = timer.stop();
    double prefill_tps = input_ids.size() / (prefill_time_ms / 1000.0);
    
    std::cout << "      [Prefill] Time: " << prefill_time_ms << " ms | Speed: " << prefill_tps << " tokens/s" << std::endl;

    // 5. Decode 阶段测试 (自回归生成)
    int generate_len = 50; // 测试生成 50 个 token
    std::cout << "      [Decode] Generating " << generate_len << " tokens..." << std::endl;

    std::vector<double> latencies;
    int current_token = next;
    
    // 预热 (Warmup) - 跑两步不计入时间，让CPU频率跑起来
    for(int i=0; i<2; ++i) {
        current_token = engine.next_token(current_token);
    }

    // 正式测速
    for (int i = 0; i < generate_len; i++) {
        timer.start();
        
        int output_token = engine.next_token(current_token);
        
        double step_ms = timer.stop();
        latencies.push_back(step_ms);
        
        // 简单的流式打印效果
        // std::cout << output_token << " " << std::flush; 
        
        current_token = output_token;
        if (current_token == engine.im_end_id) break;
    }
    std::cout << std::endl;

    // 6. 统计结果
    if (latencies.empty()) {
        std::cout << "      [Warning] No tokens generated." << std::endl;
    } else {
        double total_decode_time = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double avg_latency = total_decode_time / latencies.size();
        double decode_tps = 1000.0 / avg_latency;

        std::cout << "      [Decode] Average Latency: " << avg_latency << " ms/token" << std::endl;
        std::cout << "      [Decode] Speed: " << std::fixed << std::setprecision(2) << decode_tps << " tokens/s" << std::endl;
    }

    std::cout << "========== Done ==========" << std::endl;
    return 0;
}
