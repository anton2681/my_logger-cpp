#pragma once
#include <string>  
#include <iostream>
#include "logger_backend.h" 

class ConsoleBackend : public LoggerBackend {
public:
    void write(const std::string& level,  const std::string& message) override {
        std::cout << "[" << level << "] " << getCurrentTimestamp() << " " << message << "\n";
    }
private:
    std::string getCurrentTimestamp() {
        std::time_t now = std::time(nullptr);
        char buf[20];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buf);
    }
};