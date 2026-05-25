#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include "logger_backend.h"
#include <ctime>

class FileBackend :public LoggerBackend {
public:
    FileBackend(const std::string& filename) : filename_(filename) {}   
    std::string getCurrentTimestamp() {
        std::time_t now = std::time(nullptr);
        char buf[20];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buf);
    }
    void write(const std::string& level,const std::string& message) override {
        std::ofstream file(filename_, std::ios::app);
        if (file.is_open()) {
            file << "[" << level << "] "<<getCurrentTimestamp() << " " << message << "\n";
            file.close();
        } else {
            std::cerr << "Failed to open log file: " << filename_ << "\n";
        }
    }
private:
    std::string filename_;
};