#pragma once
#include <string>   


class LoggerBackend {
public:
    virtual ~LoggerBackend() = default;
    virtual void write(const std::string& level,const std::string& message) = 0;
};
