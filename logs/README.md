# Logs Directory

This directory contains application logs.

## Log Files:
- `app.log` - General application logs
- `error.log` - Error logs only

## Log Rotation:
- Maximum file size: 10MB
- Maximum number of files: 5
- Logs are automatically rotated when size limit is reached

## Log Levels:
- `error` - Error messages
- `warn` - Warning messages  
- `info` - Informational messages
- `debug` - Debug messages (development only)

## Log Format:
Logs are written in JSON format with the following fields:
- `timestamp` - ISO date string
- `level` - Log level
- `message` - Log message
- `service` - Always "vpbank-ml-api"
- Additional contextual fields based on the log entry
