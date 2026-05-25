#include "logger.h"
#include "console_backend.h"
#include "file_backend.h"


int main() {
    Logger logger;
    logger.addBackend(std::make_unique<ConsoleBackend>());
    logger.addBackend(std::make_unique<FileBackend>("log.txt"));
    logger.setlevel(Logger::LogLevel::INFO);
    logger.info("NONE PUT");
    logger.warn("This is a warning message.");
    logger.error("This is an error message.");
    return 0;
}