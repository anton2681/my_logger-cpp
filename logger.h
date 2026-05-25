#pragma once
#include <iostream>
#include <string>
#include "logger_backend.h"
#include <vector>
 

  class Logger {
  public:
      
      
       enum class LogLevel {
        DEBUG=0,
          INFO=1,
          WARN=2,
          ERROR=3
      };

      void info(const std::string& message) {
            
            write(LogLevel::INFO, "INFO", message);
      }

      void warn(const std::string& message) {
          write(LogLevel::WARN, "WARN", message);
      }

      void error(const std::string& message) {
          write(LogLevel::ERROR, "ERROR", message);
      }
      void setlevel(LogLevel level) {
          level_ = level;
      }
      void addBackend(std::unique_ptr<LoggerBackend> backend) {
          backends_.push_back(std::move(backend));
      }
  private:
        
      void write(LogLevel msgLevel,const std::string& level, const std::string& message) {
         if(msgLevel >= level_) {
             for(const auto& backend : backends_) {
                 backend->write(level, message);
             }
         }
                          
      }
std::vector<std::unique_ptr<LoggerBackend>> backends_;
     
      LogLevel level_=LogLevel::DEBUG;
  };

