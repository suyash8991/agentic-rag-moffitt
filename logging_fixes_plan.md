# Logging System Fixes Plan

## Current Issues

There are two primary issues with the logging system:

1. **Unicode Encoding Errors**: The system is failing to handle special characters like Greek letters (γδ) due to using Windows' default cp1252 encoding instead of UTF-8.
   - Error: `UnicodeEncodeError: 'charmap' codec can't encode characters in position 8302-8303: character maps to <undefined>`
   - Specific characters causing problems: Greek gamma (γ) and delta (δ) from research titles

2. **Duplicate Log Entries**: The same log messages are being repeated multiple times in the logs, making them difficult to read. Example:
   ```
   2025-10-21 11:57:06,567 [INFO] root (logging.py:265): Structured logging enabled with 7 component loggers
   2025-10-21 11:57:06,567 [INFO] root (logging.py:265): Structured logging enabled with 7 component loggers
   ...
   ```

## Root Causes

### Unicode Encoding Issue
- The log handlers are using Windows' default encoding (cp1252) which doesn't support all Unicode characters
- When the logger tries to write log messages containing special characters, it fails with an encoding error
- This affects all log files (app.log, errors.log, etc.) and console output

### Duplicate Log Entries Issue
- Likely caused by logger propagation or multiple initializations of the logging system
- The Streamlit framework may be initializing the application multiple times during hot-reload cycles
- Each initialization is creating new log handlers without properly cleaning up old ones
- Messages are being logged through multiple handlers resulting in duplicates

## Proposed Fixes

### 1. Unicode Encoding Fix
- Add UTF-8 encoding parameter to all file handlers:
  - Main log file handler
  - Query log file handler
  - Error log file handler
  - Structured log handlers
- Add UTF-8 reconfiguration to the console handler if possible

### 2. Duplicate Log Entries Fix
- Add a check to prevent multiple initializations of the logging system
- Use a global variable to track if logging has been initialized
- Remove all existing handlers before adding new ones during reinitialization
- Add a check to avoid duplicate logger initialization for each component in structured logging
- For Streamlit's hot-reload problem, make the logging configuration idempotent

### 3. Debug Level Adjustment
- Increase the default logging level for development to reduce verbosity
- Set default level to INFO instead of DEBUG for most log handlers
- Create a configuration setting to enable/disable debug logging

## Implementation Steps

1. **Fix Unicode Encoding**
   - Update all file handlers to use UTF-8 encoding
   - Update console handler to use UTF-8 reconfiguration if available

2. **Fix Duplicate Logs**
   - Add a global initialized flag to the logging module
   - Enhance init_logging with a check to prevent multiple initializations
   - Implement proper cleanup of existing handlers before adding new ones
   - Add single initialization guarantee for structured loggers

3. **Adjust Debug Level**
   - Modify LOG_LEVELS dictionary to use INFO as default for 'dev' environment
   - Add a debug_mode parameter to init_logging function

4. **Testing**
   - Test with Unicode characters in various locations
   - Verify proper logging without duplicates
   - Check proper encoding in all log files

5. **Documentation Update**
   - Update development_progress.md to note the bug fixes
   - Add comments in the code to explain the fixes

## Expected Results

- No more Unicode encoding errors in log files
- Single log entries instead of duplicates
- More readable logs with appropriate verbosity
- No changes to actual logging functionality other than fixing the issues